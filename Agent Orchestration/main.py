"""
orin1/main.py
─────────────────────────────────────────────────────────────────
VEDHA – Orin1 Perception Agent
Responsibilities:
  1. Continuously detect persons with YOLOv8n
  2. Continuously listen for wake word with openWakeWord
  3. On person detection → fire PERSON_DETECTED event (with cooldown)
  4. On wake word → record audio → Sarvam STT → fire STT_RESULT event
  5. On demand: capture camera frame and publish CAMERA_FRAME event

All events are published over ZMQ to Orin2 VLM server.
"""

from __future__ import annotations
import asyncio, base64, io, logging, time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyaudio
import requests
import yaml
import zmq
import zmq.asyncio

from ultralytics import YOLO
import openwakeword
from openwakeword.model import Model as WakeWordModel

from shared.protocol import EventType, PerceptionEvent

# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [Orin1] %(levelname)s %(message)s")
log = logging.getLogger("orin1")

with open(Path(__file__).parent.parent / "config.yaml") as f:
    CFG = yaml.safe_load(f)

PCFG = CFG["perception"]
SCFG = CFG["stt"]
NET  = CFG["network"]


# ═══════════════════════════════════════════════════════════════
#  Audio helpers
# ═══════════════════════════════════════════════════════════════
class AudioCapture:
    """Thin wrapper around PyAudio for microphone input."""

    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.chunk = PCFG["wake_word"]["chunk_size"]
        self.rate  = SCFG["sample_rate"]
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

    def record_until_silence(self, max_secs: float = None, silence_timeout: float = None) -> bytes:
        """Record audio until silence is detected or max_secs reached."""
        max_secs     = max_secs or SCFG["max_listen_seconds"]
        silence_to   = silence_timeout or SCFG["silence_timeout"]
        silence_thresh = 300          # RMS threshold for silence
        frames       = []
        silent_chunks = 0
        max_chunks   = int(max_secs * self.rate / self.chunk)
        silence_limit = int(silence_to * self.rate / self.chunk)

        log.info("Recording speech ...")
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

        log.info(f"Recorded {len(frames)} chunks ({len(frames)*self.chunk/self.rate:.1f}s)")
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


# ═══════════════════════════════════════════════════════════════
#  Sarvam Saaras STT
# ═══════════════════════════════════════════════════════════════
def sarvam_stt(pcm_bytes: bytes) -> str:
    """Send raw PCM to Sarvam Saaras and return transcript."""
    endpoint = SCFG["sarvam_endpoint"]
    api_key  = SCFG["sarvam_api_key"]
    lang     = SCFG["language_code"]
    model    = SCFG["model"]
    rate     = SCFG["sample_rate"]

    # Sarvam expects WAV – wrap PCM in a minimal WAV header
    wav_bytes = _pcm_to_wav(pcm_bytes, rate, SCFG["channels"])

    files   = {"file": ("audio.wav", io.BytesIO(wav_bytes), "audio/wav")}
    headers = {"api-subscription-key": api_key}
    data    = {"model": model, "language_code": lang, "with_timestamps": "false"}

    try:
        r = requests.post(endpoint, headers=headers, files=files, data=data, timeout=20)
        r.raise_for_status()
        transcript = r.json().get("transcript", "").strip()
        log.info(f"STT → '{transcript}'")
        return transcript
    except Exception as e:
        log.error(f"Sarvam STT error: {e}")
        return ""


def _pcm_to_wav(pcm: bytes, rate: int, channels: int) -> bytes:
    """Wrap raw 16-bit PCM in a WAV container."""
    import struct
    bits_per_sample = 16
    byte_rate = rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, channels, rate,
        byte_rate, block_align, bits_per_sample,
        b"data", data_size,
    )
    return header + pcm


# ═══════════════════════════════════════════════════════════════
#  Camera
# ═══════════════════════════════════════════════════════════════
class Camera:
    def __init__(self):
        c = PCFG["camera"]
        self.cap = cv2.VideoCapture(c["device"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  c["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
        self.cap.set(cv2.CAP_PROP_FPS,          c["fps"])
        self._last_frame: Optional[np.ndarray] = None

    def grab(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if ret:
            self._last_frame = frame
        return self._last_frame if ret else self._last_frame

    def encode_jpeg(self, frame: np.ndarray, quality: int = 80) -> str:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf.tobytes()).decode()

    def release(self):
        self.cap.release()


# ═══════════════════════════════════════════════════════════════
#  Person Detector (YOLOv8n)
# ═══════════════════════════════════════════════════════════════
class PersonDetector:
    PERSON_CLASS = 0   # COCO class 0 = person

    def __init__(self):
        self.model      = YOLO(PCFG["person_detection"]["model"])
        self.conf       = PCFG["person_detection"]["confidence"]
        self.cooldown   = PCFG["person_detection"]["cooldown_seconds"]
        self.min_frames = PCFG["person_detection"]["min_consecutive_frames"]
        self._last_trigger = 0.0
        self._consec    = 0

    def detect(self, frame: np.ndarray) -> bool:
        """Returns True if a person is confidently present."""
        results = self.model(frame, classes=[self.PERSON_CLASS],
                             conf=self.conf, verbose=False)
        found   = any(len(r.boxes) > 0 for r in results)

        if found:
            self._consec += 1
        else:
            self._consec = 0

        return self._consec >= self.min_frames

    def should_trigger(self) -> bool:
        """True only if cooldown has elapsed since last trigger."""
        now = time.time()
        if now - self._last_trigger >= self.cooldown:
            self._last_trigger = now
            return True
        return False


# ═══════════════════════════════════════════════════════════════
#  Wake Word Detector (openWakeWord)
# ═══════════════════════════════════════════════════════════════
class WakeWordDetector:
    def __init__(self):
        model_path = PCFG["wake_word"]["model_path"]
        self.threshold = PCFG["wake_word"]["threshold"]
        # openWakeWord supports custom ONNX models
        openwakeword.utils.download_models()
        self.model = WakeWordModel(
            wakeword_models=[model_path] if Path(model_path).exists() else [],
            inference_framework="onnx",
        )
        log.info("Wake word model loaded")

    def predict(self, chunk: bytes) -> bool:
        """Returns True if wake word is detected in chunk."""
        arr = np.frombuffer(chunk, dtype=np.int16)
        scores = self.model.predict(arr)
        # scores is dict {model_name: float}
        return any(v >= self.threshold for v in scores.values())


# ═══════════════════════════════════════════════════════════════
#  ZMQ Publisher
# ═══════════════════════════════════════════════════════════════
class EventPublisher:
    def __init__(self):
        ctx  = zmq.asyncio.Context()
        host = NET["orin2"]["host"]
        port = NET["orin1"]["perception_port"]
        self.sock = ctx.socket(zmq.PUB)
        self.sock.connect(f"tcp://{host}:{port}")
        log.info(f"Event publisher connected → tcp://{host}:{port}")

    async def publish(self, event: PerceptionEvent):
        topic = event.event_type.encode()
        await self.sock.send_multipart([topic, event.pack()])
        log.debug(f"Published {event.event_type}")


# ═══════════════════════════════════════════════════════════════
#  Main Agent Loop
# ═══════════════════════════════════════════════════════════════
class Orin1Agent:
    def __init__(self):
        self.audio    = AudioCapture()
        self.camera   = Camera()
        self.detector = PersonDetector()
        self.wakeword = WakeWordDetector()
        self.pub      = EventPublisher()
        self._listening_for_speech = False   # guard: only 1 STT session at a time
        self._session_id: Optional[str] = None

    async def run(self):
        self.audio.open()
        log.info("Orin1 agent running – listening for persons and wake word")

        await asyncio.gather(
            self._perception_loop(),
            self._audio_loop(),
        )

    # ── Camera / person detection loop ─────────────────────────
    async def _perception_loop(self):
        """Runs at camera FPS; detects persons and fires events."""
        while True:
            frame = await asyncio.to_thread(self.camera.grab)
            if frame is None:
                await asyncio.sleep(0.05)
                continue

            person_present = await asyncio.to_thread(self.detector.detect, frame)

            if person_present and self.detector.should_trigger():
                img_b64 = self.camera.encode_jpeg(frame)
                event = PerceptionEvent(
                    event_type=EventType.PERSON_DETECTED,
                    payload={"image_b64": img_b64},
                )
                self._session_id = event.session_id
                log.info(f"Person detected → session {event.session_id}")
                await self.pub.publish(event)

            await asyncio.sleep(1.0 / PCFG["camera"]["fps"])

    # ── Audio / wake-word / STT loop ───────────────────────────
    async def _audio_loop(self):
        """Continuously reads mic chunks, checks for wake word."""
        while True:
            if self._listening_for_speech:
                await asyncio.sleep(0.01)
                continue

            chunk = await asyncio.to_thread(self.audio.read_chunk)
            detected = await asyncio.to_thread(self.wakeword.predict, chunk)

            if detected:
                log.info("Wake word detected!")
                # Publish wake word event immediately
                await self.pub.publish(PerceptionEvent(
                    event_type=EventType.WAKE_WORD,
                    session_id=self._session_id or "default",
                    payload={},
                ))
                # Record and transcribe in background
                asyncio.create_task(self._record_and_transcribe())

    async def _record_and_transcribe(self):
        self._listening_for_speech = True
        try:
            pcm = await asyncio.to_thread(self.audio.record_until_silence)
            text = await asyncio.to_thread(sarvam_stt, pcm)
            if text:
                # Also grab a fresh camera frame for context
                frame   = self.camera.grab()
                img_b64 = self.camera.encode_jpeg(frame) if frame is not None else ""

                await self.pub.publish(PerceptionEvent(
                    event_type=EventType.STT_RESULT,
                    session_id=self._session_id or "default",
                    payload={"text": text, "image_b64": img_b64},
                ))
        finally:
            self._listening_for_speech = False


# ─────────────────────────────────────────────
if __name__ == "__main__":
    agent = Orin1Agent()
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        agent.audio.close()
        agent.camera.release()
        log.info("Orin1 shut down")
