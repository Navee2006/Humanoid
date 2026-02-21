"""
orin2/vlm_server.py
────────────────────────────────────────────────────────────────
VEDHA – Orin2 VLM Agent (gemma3:4b + JSON tool calling)

Handles three trigger types from Orin1:
  KNOWN_FACE_DETECTED      → personalized greeting by name/title
  UNKNOWN_PERSON_DETECTED  → generic welcome sequence
  STT_RESULT               → conversational reply to user speech

Session IDs:
  Known face  → "face_<name_lower>"   (stable, history persists across re-visits)
  Unknown     → random uuid           (fresh each encounter)
  STT         → inherits from last face trigger for that session
"""

from __future__ import annotations
import asyncio, json, logging, re, time
from pathlib import Path
from typing import Optional

import httpx
import yaml
import zmq
import zmq.asyncio
from fastapi import FastAPI
import uvicorn

from orin2.tools import execute_tool, tools_json_block
from shared.protocol import EventType, PerceptionEvent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Orin2] %(levelname)s %(message)s",
)
log = logging.getLogger("orin2")

with open(Path(__file__).parent.parent / "config.yaml") as f:
    CFG = yaml.safe_load(f)

VCFG = CFG["vlm"]
NET  = CFG["network"]

_PROMPT_BASE  = (Path(__file__).parent / "system_prompt.txt").read_text().strip()
SYSTEM_PROMPT = _PROMPT_BASE + "\n\n" + tools_json_block()

app = FastAPI(title="Vedha VLM Server")


# ═══════════════════════════════════════════════════════════════
#  Session
# ═══════════════════════════════════════════════════════════════
class SessionState:
    def __init__(self, session_id: str, face_name: str = "", face_title: str = ""):
        self.session_id   = session_id
        self.face_name    = face_name    # "" for unknown
        self.face_title   = face_title
        self.welcome_done = False
        self.history: list[dict] = []
        self.last_activity = time.time()

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        max_pairs = CFG["vlm"]["history_turns"] * 2
        if len(self.history) > max_pairs:
            self.history = self.history[-max_pairs:]
        self.last_activity = time.time()

    @property
    def display_name(self) -> str:
        if self.face_title and self.face_name:
            return f"{self.face_title} {self.face_name}"
        return self.face_name or "the visitor"


sessions: dict[str, SessionState] = {}


def get_or_create(session_id: str, face_name: str = "", face_title: str = "") -> SessionState:
    if session_id not in sessions:
        sessions[session_id] = SessionState(session_id, face_name, face_title)
        log.info(f"New session: {session_id} (name='{face_name}')")
    else:
        # Update face info if newly provided (e.g., face recognised after unknown start)
        s = sessions[session_id]
        if face_name and not s.face_name:
            s.face_name  = face_name
            s.face_title = face_title
    return sessions[session_id]


# ═══════════════════════════════════════════════════════════════
#  Tool-call parser
# ═══════════════════════════════════════════════════════════════
_TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

def extract_tool_calls(text: str) -> list[dict]:
    calls = []
    for m in _TOOL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            if "name" in obj:
                calls.append(obj)
        except json.JSONDecodeError as e:
            log.warning(f"Bad tool_call JSON: {e}")
    return calls

def strip_tool_calls(text: str) -> str:
    return _TOOL_RE.sub("", text).strip()


# ═══════════════════════════════════════════════════════════════
#  Ollama client
# ═══════════════════════════════════════════════════════════════
async def call_gemma(messages: list[dict]) -> str:
    payload = {
        "model":    VCFG["model"],
        "messages": messages,
        "stream":   False,
        "options": {
            "temperature": VCFG["temperature"],
            "num_ctx":     VCFG["context_window"],
            "num_predict": VCFG["max_tokens"],
        },
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        r = await client.post(f"{VCFG['ollama_host']}/api/chat", json=payload)
        r.raise_for_status()
    return r.json().get("message", {}).get("content", "").strip()


# ═══════════════════════════════════════════════════════════════
#  User message builder per trigger type
# ═══════════════════════════════════════════════════════════════
def _build_user_message(
    trigger_type: str,
    session: SessionState,
    user_text: str,
    image_b64: str,
) -> str:
    if trigger_type == EventType.KNOWN_FACE_DETECTED:
        name    = session.display_name
        context = (
            f"[SYSTEM EVENT: A registered person has been detected by the camera. "
            f"Their name is {name}. "
            f"Begin the personalised welcome sequence: "
            f"call look_at_person, then welcome_person with a greeting that uses their name, "
            f"then speak to invite them to interact.]"
        )

    elif trigger_type in (EventType.UNKNOWN_PERSON_DETECTED, EventType.PERSON_DETECTED):
        context = (
            "[SYSTEM EVENT: An unregistered person has been detected by the camera. "
            "Begin the standard welcome sequence: "
            "call look_at_person, then welcome_person, then speak to invite them.]"
        )

    else:  # STT_RESULT
        context = user_text or "(Empty transcript. Ask the user to repeat.)"
        if session.face_name:
            context = f"[Speaking with {session.display_name}] {context}"

    if image_b64:
        context += "\n[Camera frame available – call capture_image if visual context is needed.]"

    return context


# ═══════════════════════════════════════════════════════════════
#  Agentic tool-calling loop
# ═══════════════════════════════════════════════════════════════
async def agent_loop(
    session: SessionState,
    trigger_type: str,
    user_text: str = "",
    image_b64: str = "",
) -> str:
    user_content = _build_user_message(trigger_type, session, user_text, image_b64)
    session.add("user", user_content)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + session.history

    last_spoken = ""
    MAX_ROUNDS  = 8

    for rnd in range(MAX_ROUNDS):
        log.info(f"Gemma round {rnd + 1} [{session.session_id}]")

        raw = await call_gemma(messages)
        log.debug(f"Raw: {raw[:300]}")

        tool_calls = extract_tool_calls(raw)
        plain_text = strip_tool_calls(raw)

        if not tool_calls:
            session.add("assistant", plain_text or raw)
            log.info(f"Done. plain='{plain_text[:80]}'")
            break

        session.add("assistant", raw)
        messages.append({"role": "assistant", "content": raw})

        result_parts = []
        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            # ── welcome_person: inject person name into greeting ──
            if name == "welcome_person":
                if session.welcome_done:
                    log.warning("welcome_person blocked – already done this session")
                    result = {
                        "success": False,
                        "result":  "Already welcomed. Do NOT call welcome_person again this session.",
                    }
                else:
                    session.welcome_done = True
                    # If VLM didn't write a greeting_text, inject one with the name
                    if not args.get("greeting_text") and session.face_name:
                        name_str = session.display_name
                        args["greeting_text"] = (
                            f"Vanakkam {name_str}! Welcome back. "
                            f"It's great to see you. How can I help you today?"
                        )
                    result = await execute_tool(name, args)
            else:
                result = await execute_tool(name, args)

            if name == "speak":
                last_spoken = args.get("text", "")

            result_parts.append(
                f'<tool_result name="{name}">{json.dumps(result)}</tool_result>'
            )

        tool_msg = {
            "role": "user",
            "content": (
                "[Tool results]\n" + "\n".join(result_parts) +
                "\nContinue. Use plain text to close the turn, or more <tool_call> blocks if needed."
            ),
        }
        messages.append(tool_msg)
        session.add("user", tool_msg["content"])

    return last_spoken


# ═══════════════════════════════════════════════════════════════
#  ZMQ subscriber
# ═══════════════════════════════════════════════════════════════
async def zmq_subscriber_loop():
    ctx  = zmq.asyncio.Context()
    sock = ctx.socket(zmq.SUB)
    port = NET["orin1"]["perception_port"]
    sock.bind(f"tcp://*:{port}")

    # Subscribe to all relevant topics
    for topic in [
        EventType.KNOWN_FACE_DETECTED,
        EventType.UNKNOWN_PERSON_DETECTED,
        EventType.STT_RESULT,
        # backward-compat
        "person_detected",
    ]:
        sock.setsockopt(zmq.SUBSCRIBE, topic.encode())

    log.info(f"ZMQ SUB bound on port {port}")
    running: set[str] = set()

    while True:
        try:
            parts = await sock.recv_multipart()
            if len(parts) < 2:
                continue

            event = PerceptionEvent.unpack(parts[1])
            sid   = event.session_id

            if sid in running:
                log.debug(f"Session {sid} busy – skipping")
                continue
            running.add(sid)

            face_name  = event.payload.get("face_name",  "")
            face_title = event.payload.get("face_title", "")
            s = get_or_create(sid, face_name, face_title)

            async def _handle(ev=event, sess=s):
                try:
                    await agent_loop(
                        session      = sess,
                        trigger_type = ev.event_type,
                        user_text    = ev.payload.get("text", ""),
                        image_b64    = ev.payload.get("image_b64", ""),
                    )
                finally:
                    running.discard(sess.session_id)

            asyncio.create_task(_handle())

        except Exception as e:
            log.error(f"ZMQ error: {e}", exc_info=True)
            await asyncio.sleep(0.5)


# ═══════════════════════════════════════════════════════════════
#  FastAPI
# ═══════════════════════════════════════════════════════════════
@app.on_event("startup")
async def startup():
    asyncio.create_task(zmq_subscriber_loop())
    log.info(f"VLM server ready – model={VCFG['model']}")


@app.get("/health")
async def health():
    return {"status": "ok", "model": VCFG["model"], "sessions": len(sessions)}


@app.get("/sessions")
async def list_sessions():
    return {
        sid: {
            "face_name":    s.face_name,
            "face_title":   s.face_title,
            "welcome_done": s.welcome_done,
            "history_len":  len(s.history),
            "last_activity": s.last_activity,
        }
        for sid, s in sessions.items()
    }


@app.delete("/sessions/{session_id}")
async def reset_session(session_id: str):
    sessions.pop(session_id, None)
    return {"deleted": session_id}


@app.delete("/sessions")
async def reset_all_sessions():
    sessions.clear()
    return {"cleared": True}


if __name__ == "__main__":
    uvicorn.run(
        "orin2.vlm_server:app",
        host="0.0.0.0",
        port=NET["orin2"]["vlm_api_port"],
        log_level="info",
    )
