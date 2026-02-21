"""
ros_bridge/tool_executor.py
────────────────────────────────────────────────────────────────
VEDHA – i7 Tool Executor

Receives tool calls from Orin2 VLM over HTTP and dispatches them to:
  • ROS2 topics / actions for locomotion and gestures
  • Sarvam Bulbul v3 TTS REST API for speaker output

Run on the i7 machine alongside the ROS stack.
"""

from __future__ import annotations
import asyncio, base64, io, logging, os, struct, tempfile
from pathlib import Path
from typing import Optional

import httpx
import yaml
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# ROS2 imports – only available on i7
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    logging.warning("rclpy not found – running in mock mode (no ROS)")

# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [i7-tools] %(levelname)s %(message)s",
)
log = logging.getLogger("tool_executor")

with open(Path(__file__).parent.parent / "config.yaml") as f:
    CFG = yaml.safe_load(f)

RCFG = CFG["ros"]
TCFG = CFG["tts"]

app  = FastAPI(title="Vedha Tool Executor")


# ═══════════════════════════════════════════════════════════════
#  Sarvam Bulbul v3 TTS
# ═══════════════════════════════════════════════════════════════
async def speak_bulbul(text: str) -> bool:
    """
    Call Sarvam Bulbul v3 TTS API, receive WAV bytes, play via aplay.

    API ref: https://docs.sarvam.ai/api-reference-docs/text-to-speech
    Request:  POST /text-to-speech
              { "inputs": ["text"], "target_language_code": "en-IN",
                "speaker": "meera", "model": "bulbul:v3",
                "pitch": 0, "pace": 1.1, "loudness": 1.5,
                "speech_sample_rate": 22050,
                "enable_preprocessing": true, "eng_interpolation_wt": 123 }
    Response: { "audios": ["<base64_wav>"], "request_id": "..." }
    """
    if not text:
        return False

    headers = {
        "api-subscription-key": TCFG["sarvam_api_key"],
        "Content-Type": "application/json",
    }
    payload = {
        "inputs":                 [text],
        "target_language_code":   TCFG["language_code"],
        "speaker":                TCFG["speaker"],
        "model":                  TCFG["model"],          # "bulbul:v3"
        "pitch":                  TCFG.get("pitch", 0),
        "pace":                   TCFG.get("pace", 1.1),
        "loudness":               TCFG.get("loudness", 1.5),
        "speech_sample_rate":     22050,
        "enable_preprocessing":   TCFG.get("enable_preprocessing", True),
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                TCFG["sarvam_tts_endpoint"],
                headers=headers,
                json=payload,
            )
            r.raise_for_status()
            data = r.json()

        audios = data.get("audios", [])
        if not audios:
            log.error("Bulbul returned no audio")
            return False

        # audios[0] is a base64-encoded WAV string
        wav_bytes = base64.b64decode(audios[0])

        # Write to temp file and play
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            wav_path = f.name

        proc = await asyncio.create_subprocess_shell(
            f"aplay -D {TCFG['output_device']} {wav_path}",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()
        os.unlink(wav_path)
        log.info(f"Bulbul spoke: '{text[:60]}'")
        return True

    except httpx.HTTPStatusError as e:
        log.error(f"Bulbul HTTP error {e.response.status_code}: {e.response.text[:200]}")
        return False
    except Exception as e:
        log.error(f"Bulbul TTS error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
#  ROS2 Node
# ═══════════════════════════════════════════════════════════════
class VedhaRosNode:
    def __init__(self):
        if not ROS_AVAILABLE:
            return
        rclpy.init()
        self._node = rclpy.create_node("vedha_tool_executor")

        self.cmd_vel_pub = self._node.create_publisher(Twist,  RCFG["cmd_vel_topic"],    10)
        self.gesture_pub = self._node.create_publisher(String, RCFG["gesture_action"],   10)
        self.head_pub    = self._node.create_publisher(String, RCFG["head_control_topic"],10)
        log.info("ROS2 node ready")

    def move(self, linear_x: float, angular_z: float, duration: float):
        if not ROS_AVAILABLE:
            log.info(f"[mock] move linear={linear_x} angular={angular_z} for {duration:.1f}s")
            return
        import time
        twist = Twist()
        twist.linear.x  = linear_x
        twist.angular.z = angular_z
        start = time.time()
        while time.time() - start < duration:
            self.cmd_vel_pub.publish(twist)
            rclpy.spin_once(self._node, timeout_sec=0.05)
        self.cmd_vel_pub.publish(Twist())  # stop

    def gesture(self, name: str):
        if not ROS_AVAILABLE:
            log.info(f"[mock] gesture: {name}")
            return
        msg = String(); msg.data = name
        self.gesture_pub.publish(msg)

    def head_cmd(self, cmd: str):
        if not ROS_AVAILABLE:
            log.info(f"[mock] head: {cmd}")
            return
        msg = String(); msg.data = cmd
        self.head_pub.publish(msg)


ros_node: Optional[VedhaRosNode] = None


# ═══════════════════════════════════════════════════════════════
#  Request model
# ═══════════════════════════════════════════════════════════════
class ToolRequest(BaseModel):
    call_id:   str
    arguments: dict = {}


# ═══════════════════════════════════════════════════════════════
#  Tool endpoints
# ═══════════════════════════════════════════════════════════════
@app.post("/tools/welcome_person")
async def tool_welcome_person(req: ToolRequest):
    greeting = req.arguments.get(
        "greeting_text",
        "Vanakkam! I'm Vedha. Welcome! Feel free to ask me anything."
    )
    if ros_node:
        await asyncio.to_thread(ros_node.gesture, "wave")
        await asyncio.to_thread(ros_node.head_cmd, "look_forward")
    ok = await speak_bulbul(greeting)
    return {"success": ok, "result": "welcomed"}


@app.post("/tools/speak")
async def tool_speak(req: ToolRequest):
    text = req.arguments.get("text", "")
    if not text:
        return {"success": False, "result": None, "error": "No text provided"}
    ok = await speak_bulbul(text)
    return {"success": ok, "result": f"spoke: {text[:60]}"}


@app.post("/tools/move_forward")
async def tool_move_forward(req: ToolRequest):
    dist  = min(max(float(req.arguments.get("distance_meters", 0.3)), 0.05), 2.0)
    speed = CFG["ros"]["linear_speed"]
    if ros_node:
        await asyncio.to_thread(ros_node.move, speed, 0.0, dist / speed)
    return {"success": True, "result": f"moved forward {dist}m"}


@app.post("/tools/move_backward")
async def tool_move_backward(req: ToolRequest):
    dist  = min(max(float(req.arguments.get("distance_meters", 0.3)), 0.05), 2.0)
    speed = CFG["ros"]["linear_speed"]
    if ros_node:
        await asyncio.to_thread(ros_node.move, -speed, 0.0, dist / speed)
    return {"success": True, "result": f"moved backward {dist}m"}


@app.post("/tools/turn_left")
async def tool_turn_left(req: ToolRequest):
    import math
    angle_rad = math.radians(float(req.arguments.get("angle_degrees", 45)))
    ang_speed = CFG["ros"]["angular_speed"]
    if ros_node:
        await asyncio.to_thread(ros_node.move, 0.0, ang_speed, angle_rad / ang_speed)
    return {"success": True, "result": f"turned left"}


@app.post("/tools/turn_right")
async def tool_turn_right(req: ToolRequest):
    import math
    angle_rad = math.radians(float(req.arguments.get("angle_degrees", 45)))
    ang_speed = CFG["ros"]["angular_speed"]
    if ros_node:
        await asyncio.to_thread(ros_node.move, 0.0, -ang_speed, angle_rad / ang_speed)
    return {"success": True, "result": f"turned right"}


@app.post("/tools/wave_hand")
async def tool_wave_hand(req: ToolRequest):
    if ros_node: await asyncio.to_thread(ros_node.gesture, "wave")
    return {"success": True, "result": "waved"}


@app.post("/tools/nod_head")
async def tool_nod_head(req: ToolRequest):
    if ros_node: await asyncio.to_thread(ros_node.head_cmd, "nod")
    return {"success": True, "result": "nodded"}


@app.post("/tools/shake_head")
async def tool_shake_head(req: ToolRequest):
    if ros_node: await asyncio.to_thread(ros_node.head_cmd, "shake")
    return {"success": True, "result": "shook head"}


@app.post("/tools/look_at_person")
async def tool_look_at_person(req: ToolRequest):
    if ros_node: await asyncio.to_thread(ros_node.head_cmd, "look_at_person")
    return {"success": True, "result": "looking at person"}


# ─────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global ros_node
    ros_node = await asyncio.to_thread(VedhaRosNode)
    log.info("Tool executor ready. TTS=Sarvam Bulbul v3")


@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "ros":     ROS_AVAILABLE and ros_node is not None,
        "tts":     TCFG["model"],
        "speaker": TCFG["speaker"],
    }


if __name__ == "__main__":
    uvicorn.run(
        "ros_bridge.tool_executor:app",
        host="0.0.0.0",
        port=CFG["network"]["i7"]["tool_executor_port"],
        log_level="info",
    )
