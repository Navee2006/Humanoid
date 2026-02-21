"""
shared/protocol.py
─────────────────
Typed message envelopes exchanged between Orin1 ↔ Orin2 ↔ i7.
Uses msgpack for compact binary framing over ZMQ.
"""

from __future__ import annotations
import time, uuid
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import msgpack


# ──────────────────────────────────────────────
#  Event types published by Orin1
# ──────────────────────────────────────────────
class EventType(str, Enum):
    # Camera detected a registered face
    KNOWN_FACE_DETECTED   = "known_face_detected"
    # Camera detected a person but face is not in the registry
    UNKNOWN_PERSON_DETECTED = "unknown_person_detected"
    # A previously visible person / face is no longer in frame
    PERSON_LOST           = "person_lost"
    # Wake word ("Hey Vedha") detected by openWakeWord
    WAKE_WORD             = "wake_word"
    # Sarvam Saaras STT result ready
    STT_RESULT            = "stt_result"
    # On-demand camera frame (used internally)
    CAMERA_FRAME          = "camera_frame"

    # ── backward-compat alias (kept so vlm_server imports don't break) ──
    PERSON_DETECTED       = "unknown_person_detected"


# ──────────────────────────────────────────────
#  Tool names
# ──────────────────────────────────────────────
class ToolName(str, Enum):
    WELCOME_PERSON   = "welcome_person"
    SPEAK            = "speak"
    CAPTURE_IMAGE    = "capture_image"
    MOVE_FORWARD     = "move_forward"
    MOVE_BACKWARD    = "move_backward"
    TURN_LEFT        = "turn_left"
    TURN_RIGHT       = "turn_right"
    WAVE_HAND        = "wave_hand"
    NOD_HEAD         = "nod_head"
    SHAKE_HEAD       = "shake_head"
    LOOK_AT_PERSON   = "look_at_person"


# ──────────────────────────────────────────────
#  Wire messages
# ──────────────────────────────────────────────
@dataclass
class PerceptionEvent:
    event_type: str
    timestamp: float  = field(default_factory=time.time)
    session_id: str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    payload: dict     = field(default_factory=dict)
    # Convenience: human-readable label for logging / VLM context
    label: str        = ""

    def pack(self) -> bytes:
        return msgpack.packb(asdict(self), use_bin_type=True)

    @classmethod
    def unpack(cls, data: bytes) -> "PerceptionEvent":
        d = msgpack.unpackb(data, raw=False)
        return cls(**d)


@dataclass
class ToolCall:
    call_id: str
    tool_name: str
    arguments: dict = field(default_factory=dict)

    def pack(self) -> bytes:
        return msgpack.packb(asdict(self), use_bin_type=True)

    @classmethod
    def unpack(cls, data: bytes) -> "ToolCall":
        d = msgpack.unpackb(data, raw=False)
        return cls(**d)


@dataclass
class ToolResult:
    call_id: str
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None

    def pack(self) -> bytes:
        return msgpack.packb(asdict(self), use_bin_type=True)

    @classmethod
    def unpack(cls, data: bytes) -> "ToolResult":
        d = msgpack.unpackb(data, raw=False)
        return cls(**d)
