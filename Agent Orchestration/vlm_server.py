"""
orin2/vlm_server.py
────────────────────────────────────────────────────────────────
VEDHA – Orin2 VLM Agent (gemma3:4b edition)

gemma3:4b has NO native tool-calling in Ollama's protocol.
Strategy: inject tool schemas as JSON in the system prompt,
parse the model's raw text output for JSON tool-call blocks,
execute them, inject results, and loop until the model emits
plain text with no tool call.
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

from orin2.tools import TOOL_SCHEMAS, execute_tool, tools_json_block
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

app = FastAPI(title="Vedha VLM Server (gemma3:4b)")


# ═══════════════════════════════════════════════════════════════
#  Session state
# ═══════════════════════════════════════════════════════════════
class SessionState:
    def __init__(self, session_id: str):
        self.session_id      = session_id
        self.welcome_done    = False
        self.history: list[dict] = []
        self.last_activity   = time.time()

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        max_pairs = CFG["vlm"]["history_turns"] * 2
        if len(self.history) > max_pairs:
            self.history = self.history[-max_pairs:]
        self.last_activity = time.time()


sessions: dict[str, SessionState] = {}

def get_or_create(session_id: str) -> SessionState:
    if session_id not in sessions:
        sessions[session_id] = SessionState(session_id)
        log.info(f"New session: {session_id}")
    return sessions[session_id]


# ═══════════════════════════════════════════════════════════════
#  JSON tool-call parser
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
#  Ollama client (gemma3:4b – text only)
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
#  Agentic loop
# ═══════════════════════════════════════════════════════════════
async def agent_loop(
    session: SessionState,
    trigger_type: str,
    user_text: str = "",
    image_b64: str = "",
) -> str:
    if trigger_type == EventType.PERSON_DETECTED:
        user_content = (
            "[SYSTEM EVENT: A person has just been detected. "
            "Begin the welcome sequence: call look_at_person, "
            "then welcome_person, then speak to invite them.]"
        )
    else:
        user_content = user_text or "(Empty transcript. Ask user to repeat.)"

    if image_b64:
        user_content += (
            "\n[Camera frame available – call capture_image if visual context is needed.]"
        )

    session.add("user", user_content)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + session.history

    last_spoken = ""
    MAX_ROUNDS  = 8

    for round_num in range(MAX_ROUNDS):
        log.info(f"Gemma round {round_num + 1} (session={session.session_id})")

        raw = await call_gemma(messages)
        log.debug(f"Raw: {raw[:300]}")

        tool_calls = extract_tool_calls(raw)
        plain_text = strip_tool_calls(raw)

        if not tool_calls:
            session.add("assistant", plain_text or raw)
            log.info(f"Loop complete. plain='{plain_text[:80]}'")
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

            if name == "welcome_person":
                if session.welcome_done:
                    log.warning("welcome_person blocked (already done)")
                    result = {"success": False,
                              "result": "Already welcomed. Do not call welcome_person again."}
                else:
                    session.welcome_done = True
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
                "\nContinue. Output plain text if done, or more <tool_call> blocks if needed."
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

    for topic in [EventType.PERSON_DETECTED, EventType.STT_RESULT]:
        sock.setsockopt(zmq.SUBSCRIBE, topic.encode())

    log.info(f"ZMQ bound on port {port}")
    running: set[str] = set()

    while True:
        try:
            parts = await sock.recv_multipart()
            if len(parts) < 2:
                continue
            event = PerceptionEvent.unpack(parts[1])
            sid   = event.session_id

            if sid in running:
                continue
            running.add(sid)
            s = get_or_create(sid)

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
            "welcome_done":  s.welcome_done,
            "history_len":   len(s.history),
            "last_activity": s.last_activity,
        }
        for sid, s in sessions.items()
    }


@app.delete("/sessions/{session_id}")
async def reset_session(session_id: str):
    sessions.pop(session_id, None)
    return {"deleted": session_id}


if __name__ == "__main__":
    uvicorn.run(
        "orin2.vlm_server:app",
        host="0.0.0.0",
        port=NET["orin2"]["vlm_api_port"],
        log_level="info",
    )
