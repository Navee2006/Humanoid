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
    PERSON_DETECTED   = "person_detected"
    PERSON_LOST       = "person_lost"
    WAKE_WORD         = "wake_word"
    STT_RESULT        = "stt_result"
    CAMERA_FRAME      = "camera_frame"   # base64 JPEG


# ──────────────────────────────────────────────
#  Tool call / result types
# ──────────────────────────────────────────────
class ToolName(str, Enum):
    WELCOME_PERSON  = "welcome_person"
    SPEAK           = "speak"
    CAPTURE_IMAGE   = "capture_image"
    MOVE_FORWARD    = "move_forward"
    MOVE_BACKWARD   = "move_backward"
    TURN_LEFT       = "turn_left"
    TURN_RIGHT      = "turn_right"
    WAVE_HAND       = "wave_hand"
    NOD_HEAD        = "nod_head"
    SHAKE_HEAD      = "shake_head"
    LOOK_AT_PERSON  = "look_at_person"


# ──────────────────────────────────────────────
#  Wire messages
# ──────────────────────────────────────────────
@dataclass
class PerceptionEvent:
    event_type: str
    timestamp: float = field(default_factory=time.time)
    session_id: str  = field(default_factory=lambda: str(uuid.uuid4())[:8])
    payload: dict    = field(default_factory=dict)

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
