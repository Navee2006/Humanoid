"""
orin1/snapshot_server.py
────────────────────────────────────────────────────────────────
Tiny FastAPI server that serves the latest camera frame as base64.
Called by Orin2 when the VLM executes the `capture_image` tool.
Run alongside orin1/main.py.
"""

import asyncio, base64, logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from fastapi import FastAPI
import uvicorn

log = logging.getLogger("snapshot_server")
with open(Path(__file__).parent.parent / "config.yaml") as f:
    CFG = yaml.safe_load(f)

PCFG = CFG["perception"]["camera"]

app = FastAPI(title="Vedha Snapshot Server")

_cap: Optional[cv2.VideoCapture] = None
_last_frame: Optional[np.ndarray] = None
_lock = asyncio.Lock()


@app.on_event("startup")
async def startup():
    global _cap
    _cap = cv2.VideoCapture(PCFG["device"])
    _cap.set(cv2.CAP_PROP_FRAME_WIDTH,  PCFG["width"])
    _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PCFG["height"])
    asyncio.create_task(_capture_loop())


async def _capture_loop():
    """Background task: grab frames at camera FPS."""
    global _last_frame
    while True:
        ret, frame = await asyncio.to_thread(_cap.read)
        if ret:
            async with _lock:
                _last_frame = frame
        await asyncio.sleep(1.0 / PCFG["fps"])


@app.get("/snapshot")
async def snapshot(quality: int = 80):
    async with _lock:
        frame = _last_frame
    if frame is None:
        return {"image_b64": "", "error": "No frame available"}

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    b64 = base64.b64encode(buf.tobytes()).decode()
    return {"image_b64": b64, "width": PCFG["width"], "height": PCFG["height"]}


if __name__ == "__main__":
    uvicorn.run("orin1.snapshot_server:app", host="0.0.0.0", port=5553)
