"""
orin2/tools.py
────────────────────────────────────────────────────────────────
Tool schema (Ollama / OpenAI format) + executor that routes
each tool call to the appropriate service (i7 REST API or local).
"""

from __future__ import annotations
import logging, uuid
from typing import Any

import httpx
import yaml
from pathlib import Path

log = logging.getLogger("orin2.tools")
with open(Path(__file__).parent.parent / "config.yaml") as f:
    CFG = yaml.safe_load(f)

I7_BASE = f"http://{CFG['network']['i7']['host']}:{CFG['network']['i7']['tool_executor_port']}"


# ═══════════════════════════════════════════════════════════════
#  TOOL SCHEMAS  (Ollama JSON-tool format)
# ═══════════════════════════════════════════════════════════════
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "welcome_person",
            "description": (
                "Play the standard Vedha welcome greeting and perform a welcoming wave. "
                "Call this EXACTLY ONCE when a person is first detected in a session. "
                "Do NOT call again until the person leaves and re-enters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "greeting_text": {
                        "type": "string",
                        "description": "Optional custom greeting. Leave empty to use default.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "speak",
            "description": "Speak the given text aloud through the robot speaker using TTS.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to speak. Keep natural and conversational.",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_image",
            "description": (
                "Capture the current camera frame to gain visual context. "
                "Use when you need to see what is in front of you before responding."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_forward",
            "description": "Move the robot forward by a specified distance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "distance_meters": {
                        "type": "number",
                        "description": "Distance to move forward in meters (0.1 – 2.0).",
                    }
                },
                "required": ["distance_meters"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_backward",
            "description": "Move the robot backward by a specified distance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "distance_meters": {"type": "number"}
                },
                "required": ["distance_meters"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "turn_left",
            "description": "Turn the robot left by a given angle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle_degrees": {
                        "type": "number",
                        "description": "Angle to turn in degrees (5 – 180).",
                    }
                },
                "required": ["angle_degrees"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "turn_right",
            "description": "Turn the robot right by a given angle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle_degrees": {"type": "number"}
                },
                "required": ["angle_degrees"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wave_hand",
            "description": "Perform a friendly waving gesture with the robot arm.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nod_head",
            "description": "Nod the robot head to indicate yes or acknowledgement.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shake_head",
            "description": "Shake the robot head to indicate no or disagreement.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look_at_person",
            "description": "Turn head/gaze toward the detected person for engagement.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# Index for quick lookup
TOOL_MAP = {t["function"]["name"]: t for t in TOOL_SCHEMAS}


# ═══════════════════════════════════════════════════════════════
#  TOOL EXECUTOR  (routes to i7 REST API)
# ═══════════════════════════════════════════════════════════════
async def execute_tool(tool_name: str, arguments: dict) -> dict:
    """
    Route tool call to appropriate endpoint on i7 tool executor.
    Returns {"success": bool, "result": Any}
    """
    call_id = str(uuid.uuid4())[:8]
    log.info(f"[{call_id}] Tool call: {tool_name}({arguments})")

    # capture_image is handled locally on Orin2 (camera snapshot from Orin1)
    if tool_name == "capture_image":
        return await _capture_image_local()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{I7_BASE}/tools/{tool_name}",
                json={"call_id": call_id, "arguments": arguments},
            )
            r.raise_for_status()
            result = r.json()
            log.info(f"[{call_id}] Result: {result}")
            return result
    except httpx.HTTPError as e:
        log.error(f"[{call_id}] HTTP error calling {tool_name}: {e}")
        return {"success": False, "result": None, "error": str(e)}
    except Exception as e:
        log.error(f"[{call_id}] Unexpected error: {e}")
        return {"success": False, "result": None, "error": str(e)}


async def _capture_image_local() -> dict:
    """
    Request latest frame from Orin1 via its HTTP snapshot endpoint.
    Returns base64-encoded JPEG.
    """
    orin1_host = CFG["network"]["orin1"]["host"]
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"http://{orin1_host}:5553/snapshot")
            r.raise_for_status()
            return {"success": True, "result": r.json().get("image_b64", "")}
    except Exception as e:
        log.error(f"Snapshot error: {e}")
        return {"success": False, "result": "", "error": str(e)}


# ═══════════════════════════════════════════════════════════════
#  JSON-in-prompt tool schema renderer
#  (used because gemma3:4b has no native tool-calling)
# ═══════════════════════════════════════════════════════════════
import json as _json

def tools_json_block() -> str:
    """
    Render all tool schemas as a system-prompt block.
    The model emits:  <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    Results come back: <tool_result name="...">{"success": true}</tool_result>
    """
    compact = []
    for t in TOOL_SCHEMAS:
        fn = t["function"]
        compact.append({
            "name":        fn["name"],
            "description": fn["description"],
            "parameters":  fn.get("parameters", {}).get("properties", {}),
            "required":    fn.get("parameters", {}).get("required", []),
        })

    schema_json = _json.dumps(compact, indent=2)

    return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL CALLING FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
To call a tool, output EXACTLY this (one per line, no other text in the same response):
  <tool_call>{{"name": "tool_name", "arguments": {{"key": "value"}}}}</tool_call>

Tool results are returned as:
  <tool_result name="tool_name">{{"success": true, "result": "..."}}</tool_result>

Rules:
• Only call tools from the list below.
• One tool call per line; multiple allowed if truly parallel.
• Do NOT mix plain text and tool calls in the same response.
• After all results arrive, continue with plain text to close the turn.
• No ```json fences. No markdown. Raw XML tags only.

AVAILABLE TOOLS:
{schema_json}
"""
