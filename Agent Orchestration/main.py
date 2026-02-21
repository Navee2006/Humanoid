"""
orin1/main.py
────────────────────────────────────────────────────────────────
VEDHA – Orin1 Perception Agent

Responsibilities:
  1. Run face recognition on every camera frame (FaceRecognitionDetector)
     • Known face  → KNOWN_FACE_DETECTED  event (name, title, image)
     • Unknown     → UNKNOWN_PERSON_DETECTED event (image)
  2. Listen for wake word with openWakeWord
  3. On wake word → record audio → Sarvam Saaras STT → STT_RESULT event
  4. Serve latest frame over HTTP for on-demand capture_image tool calls

Camera source:
  • If config.perception.camera.ros_topic is non-empty → use ROS subscriber
    (requires rclpy + camera publishing to that topic on the network)
  • Otherwise → use OpenCV VideoCapture directly

All events published over ZMQ PUB to Orin2.
"""

from __future__ import annotations
import asyncio, base64, io, logging, struct, time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyaudio
import requests
import yaml
import zmq
import zmq.asyncio

from orin1.face_detector import FaceRecognitionDetector, FaceDetection
from shared.protocol import EventType, PerceptionEvent

import openwakeword
from openwakeword.model import Model as WakeWordModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Orin1] %(levelname)s %(message)s"
)
log = logging.getLogger("orin1")

with open(Path(__file__).parent.parent / "config.yaml") as f:
    CFG = yaml.safe_load(f)

PCFG = CFG["perception"]
SCFG = CFG["stt"]
NET  = CFG["network"]


# ═══════════════════════════════════════════════════════════════
#  Shared camera frame store (written by camera loop, read by snapshot)
# ═══════════════════════════════════════════════════════════════
import threading
_frame_lock  = threading.Lock()
_latest_frame: Optional[np.ndarray] = None

def _set_frame(frame: np.ndarray):
    global _latest_frame
    with _frame_lock:
        _latest_frame = frame.copy()

def _get_frame() -> Optional[np.ndarray]:
    with _frame_lock:
        return _latest_frame.copy() if _latest_frame is not None else None


# ═══════════════════════════════════════════════════════════════
#  Audio helpers
# ═══════════════════════════════════════════════════════════════
class AudioCapture:
    def __init__(self):
        self.pa     = pyaudio.PyAudio()
        self.chunk  = PCFG["wake_word"]["chunk_size"]
        self.rate   = SCFG["sample_rate"]
        self.stream: Optional[pyaudio.Stream] = None

    def open(self):
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=SCFG["channels"],
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )
        log.info("Microphone stream opened")

    def read_chunk(self) -> bytes:
        return self.stream.read(self.chunk, exception_on_overflow=False)

    def record_until_silence(self) -> bytes:
        silence_thresh = 300
        frames         = []
        silent_chunks  = 0
        max_chunks     = int(SCFG["max_listen_seconds"] * self.rate / self.chunk)
        silence_limit  = int(SCFG["silence_timeout"]    * self.rate / self.chunk)

        log.info("Recording speech …")
        for _ in range(max_chunks):
            data = self.read_chunk()
            frames.append(data)
            rms  = self._rms(data)
            if rms < silence_thresh:
                silent_chunks += 1
                if silent_chunks >= silence_limit:
                    break
            else:
                silent_chunks = 0

        log.info(f"Recorded {len(frames)*self.chunk/self.rate:.1f}s")
        return b"".join(frames)

    @staticmethod
    def _rms(data: bytes) -> float:
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        return float(np.sqrt(np.mean(arr ** 2))) if len(arr) > 0 else 0.0

    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()


def _pcm_to_wav(pcm: bytes, rate: int, channels: int) -> bytes:
    bits = 16
    byte_rate   = rate * channels * bits // 8
    block_align = channels * bits // 8
    data_size   = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, channels, rate,
        byte_rate, block_align, bits,
        b"data", data_size,
    )
    return header + pcm


def sarvam_stt(pcm_bytes: bytes) -> str:
    wav  = _pcm_to_wav(pcm_bytes, SCFG["sample_rate"], SCFG["channels"])
    hdrs = {"api-subscription-key": SCFG["sarvam_api_key"]}
    data = {"model": SCFG["model"], "language_code": SCFG["language_code"],
            "with_timestamps": "false"}
    files = {"file": ("audio.wav", io.BytesIO(wav), "audio/wav")}
    try:
        r = requests.post(SCFG["sarvam_endpoint"], headers=hdrs,
                          files=files, data=data, timeout=20)
        r.raise_for_status()
        t = r.json().get("transcript", "").strip()
        log.info(f"STT → '{t}'")
        return t
    except Exception as e:
        log.error(f"Sarvam STT error: {e}")
        return ""


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


# ═══════════════════════════════════════════════════════════════
#  Wake word detector
# ═══════════════════════════════════════════════════════════════
class WakeWordDetector:
    def __init__(self):
        self.threshold = PCFG["wake_word"]["threshold"]
        model_path     = PCFG["wake_word"]["model_path"]
        openwakeword.utils.download_models()
        self.model = WakeWordModel(
            wakeword_models=[model_path] if Path(model_path).exists() else [],
            inference_framework="onnx",
        )
        log.info("Wake word model loaded")

    def predict(self, chunk: bytes) -> bool:
        arr    = np.frombuffer(chunk, dtype=np.int16)
        scores = self.model.predict(arr)
        return any(v >= self.threshold for v in scores.values())


# ═══════════════════════════════════════════════════════════════
#  ZMQ Event Publisher
# ═══════════════════════════════════════════════════════════════
class EventPublisher:
    def __init__(self):
        ctx  = zmq.asyncio.Context()
        host = NET["orin2"]["host"]
        port = NET["orin1"]["perception_port"]
        self.sock = ctx.socket(zmq.PUB)
        self.sock.connect(f"tcp://{host}:{port}")
        log.info(f"ZMQ PUB → tcp://{host}:{port}")

    async def publish(self, event: PerceptionEvent):
        await self.sock.send_multipart(
            [event.event_type.encode(), event.pack()]
        )
        log.debug(f"Published {event.event_type} [{event.session_id}]")


# ═══════════════════════════════════════════════════════════════
#  Camera source – OpenCV or ROS
# ═══════════════════════════════════════════════════════════════
class OpenCVCamera:
    """Direct OpenCV capture loop – runs in background thread."""

    def __init__(self, on_frame: callable):
        c          = PCFG["camera"]
        self._cb   = on_frame
        self._cap  = cv2.VideoCapture(c["device"])
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  c["width"])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
        self._cap.set(cv2.CAP_PROP_FPS,          c["fps"])
        self._running = False

    def start(self):
        self._running = True
        import threading
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                self._cb(frame)
        self._cap.release()

    def stop(self):
        self._running = False


class ROSCamera:
    """ROS subscriber for /camera/color/image_raw – runs rclpy in background."""

    def __init__(self, topic: str, on_frame: callable):
        self._topic = topic
        self._cb    = on_frame

    def start(self):
        import threading
        threading.Thread(target=self._spin, daemon=True).start()

    def _spin(self):
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import Image
            from cv_bridge import CvBridge

            rclpy.init()
            node   = rclpy.create_node("vedha_camera_subscriber")
            bridge = CvBridge()

            def _img_cb(msg):
                try:
                    frame = bridge.imgmsg_to_cv2(msg, "bgr8")
                    self._cb(frame)
                except Exception as e:
                    log.error(f"ROSCamera bridge error: {e}")

            node.create_subscription(Image, self._topic, _img_cb, 10)
            log.info(f"ROS camera subscriber → {self._topic}")
            rclpy.spin(node)
        except Exception as e:
            log.error(f"ROSCamera failed: {e}. Falling back to OpenCV camera.")
            # Fallback
            OpenCVCamera(self._cb).start()


# ═══════════════════════════════════════════════════════════════
#  Main Orin1 Agent
# ═══════════════════════════════════════════════════════════════
class Orin1Agent:
    def __init__(self):
        self.audio    = AudioCapture()
        self.detector = FaceRecognitionDetector()
        self.wakeword = WakeWordDetector()
        self.pub      = EventPublisher()

        # Active session for wake word (set when face seen)
        self._active_session: str = "default"
        self._stt_busy: bool = False

        # Wire face-detection trigger callback
        self.detector.set_trigger_callback(self._on_face_trigger)

        # Choose camera source
        ros_topic = PCFG["camera"].get("ros_topic", "")
        if ros_topic:
            self._camera = ROSCamera(ros_topic, self._on_frame)
        else:
            self._camera = OpenCVCamera(self._on_frame)

    # ── Camera frame callback (called from background thread) ───
    def _on_frame(self, frame: np.ndarray):
        """Called by camera source for every new frame."""
        _set_frame(frame)   # store for snapshot server

        # Run face recognition (sync – it's fast at 0.25 scale)
        annotated, detections = self.detector.process_frame(frame)

        # Optional: show live feed if display is attached
        # cv2.imshow("Vedha – Face Recognition", annotated)
        # cv2.waitKey(1)

    # ── Face trigger callback (called from detector after gating) ─
    def _on_face_trigger(self, det: FaceDetection, frame: np.ndarray):
        """
        Called when a face clears consecutive-frame threshold + cooldown.
        Publishes appropriate event to Orin2 (async-safe via run_coroutine_threadsafe).
        """
        img_b64 = encode_jpeg(frame)
        self._active_session = det.session_id

        if det.is_known:
            event = PerceptionEvent(
                event_type = EventType.KNOWN_FACE_DETECTED,
                session_id = det.session_id,
                label      = det.name,
                payload    = {
                    "face_name":  det.name,
                    "face_title": det.title,
                    "distance":   det.distance,
                    "image_b64":  img_b64,
                },
            )
        else:
            event = PerceptionEvent(
                event_type = EventType.UNKNOWN_PERSON_DETECTED,
                session_id = det.session_id,
                label      = "Unknown",
                payload    = {
                    "face_name":  "Unknown",
                    "face_title": "",
                    "image_b64":  img_b64,
                },
            )

        # Schedule coroutine on the running event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(self.pub.publish(event), loop)
        else:
            log.warning("Event loop not running – dropping trigger event")

    # ── Main async entry point ──────────────────────────────────
    async def run(self):
        self.audio.open()
        self._camera.start()
        log.info("Orin1 agent running – face recognition + wake word active")

        # Only async task: wake-word + STT loop
        await self._audio_loop()

    async def _audio_loop(self):
        while True:
            if self._stt_busy:
                await asyncio.sleep(0.01)
                continue

            chunk    = await asyncio.to_thread(self.audio.read_chunk)
            detected = await asyncio.to_thread(self.wakeword.predict, chunk)

            if detected:
                log.info("Wake word detected!")
                await self.pub.publish(PerceptionEvent(
                    event_type = EventType.WAKE_WORD,
                    session_id = self._active_session,
                    payload    = {},
                ))
                asyncio.create_task(self._record_and_transcribe())

    async def _record_and_transcribe(self):
        self._stt_busy = True
        try:
            pcm  = await asyncio.to_thread(self.audio.record_until_silence)
            text = await asyncio.to_thread(sarvam_stt, pcm)
            if text:
                frame   = _get_frame()
                img_b64 = encode_jpeg(frame) if frame is not None else ""
                await self.pub.publish(PerceptionEvent(
                    event_type = EventType.STT_RESULT,
                    session_id = self._active_session,
                    payload    = {"text": text, "image_b64": img_b64},
                ))
        finally:
            self._stt_busy = False


# ─────────────────────────────────────────────
if __name__ == "__main__":
    agent = Orin1Agent()
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        agent.audio.close()
        log.info("Orin1 shut down")
