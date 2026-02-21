# VEDHA – Humanoid Robot AI System

> Humanoid robot assistant powered by on-device VLM, wake word detection,
> person detection, and Sarvam Saaras STT — all running across a distributed
> Jetson Orin Nano + i7 cluster.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        VEDHA System Architecture                      │
└──────────────────────────────────────────────────────────────────────┘

 ┌─────────────────────┐         ZMQ PUB/SUB         ┌────────────────────────┐
 │     ORIN 1          │ ──────── events ──────────▶  │       ORIN 2           │
 │  (Perception)       │                               │    (VLM / Brain)       │
 │                     │◀─ HTTP /snapshot ──────────── │                        │
 │  ┌───────────────┐  │                               │  ┌──────────────────┐  │
 │  │  YOLOv8n      │  │                               │  │  Ollama          │  │
 │  │  Person Det.  │  │                               │  │  llama3.2-vision │  │
 │  └───────────────┘  │                               │  └──────────────────┘  │
 │  ┌───────────────┐  │                               │  ┌──────────────────┐  │
 │  │  openWakeWord │  │                               │  │  Tool-call loop  │  │
 │  │  "Hey Vedha"  │  │                               │  │  Session manager │  │
 │  └───────────────┘  │                               │  └──────────────────┘  │
 │  ┌───────────────┐  │                               │                        │
 │  │ Sarvam Saaras │  │                               └──────────┬─────────────┘
 │  │  STT (cloud)  │  │                                          │
 │  └───────────────┘  │                                          │ HTTP tool calls
 │  ┌───────────────┐  │                                          ▼
 │  │  Camera       │  │                               ┌────────────────────────┐
 │  │  /dev/video0  │  │                               │       i7 Machine       │
 │  └───────────────┘  │                               │    (ROS Locomotion)    │
 └─────────────────────┘                               │                        │
                                                        │  ┌──────────────────┐  │
                                                        │  │  FastAPI         │  │
                                                        │  │  Tool Executor   │  │
                                                        │  └──────────────────┘  │
                                                        │  ┌──────────────────┐  │
                                                        │  │  ROS2 Node       │  │
                                                        │  │  /cmd_vel        │  │
                                                        │  │  /gesture        │  │
                                                        │  │  /head/cmd       │  │
                                                        │  └──────────────────┘  │
                                                        │  ┌──────────────────┐  │
                                                        │  │  Piper TTS       │  │
                                                        │  │  Speaker output  │  │
                                                        │  └──────────────────┘  │
                                                        └────────────────────────┘
```

---

## Interaction Flow

### Person Detection Trigger
```
Camera → YOLOv8n detects person (3 consecutive frames)
       → Cooldown check (30s since last trigger)
       → Publish PERSON_DETECTED event (with camera frame)
       → Orin2 receives event
       → VLM tool sequence:
            look_at_person  →  welcome_person (once!)  →  speak("Vanakkam! ...")
```

### Wake Word Trigger
```
Mic → openWakeWord detects "Hey Vedha"
    → Publish WAKE_WORD event
    → Record audio (until silence, max 10s)
    → Sarvam Saaras STT → transcript
    → Publish STT_RESULT event (text + camera frame)
    → Orin2 receives event
    → VLM processes text + image
    → Tool sequence: [capture_image?] → speak(response) → [movement?]
```

---

## File Structure

```
vedha/
├── config.yaml                  ← All configuration
├── requirements.txt
├── docker-compose.yml           ← Orin2 deployment
│
├── orin1/
│   ├── main.py                  ← Perception agent (wake word + person + STT)
│   └── snapshot_server.py       ← HTTP endpoint for on-demand frame capture
│
├── orin2/
│   ├── vlm_server.py            ← VLM agent + ZMQ subscriber + tool loop
│   ├── tools.py                 ← Tool schemas + executor router
│   └── system_prompt.txt        ← Vedha identity + behavior rules
│
├── ros_bridge/
│   └── tool_executor.py         ← FastAPI server bridging tools → ROS2
│
└── shared/
    └── protocol.py              ← ZMQ message types
```

---

## Setup

### 1. Configure `config.yaml`
Edit IPs, API keys, and model paths:
```yaml
network:
  orin1.host: "192.168.1.10"
  orin2.host: "192.168.1.11"
  i7.host:    "192.168.1.12"

stt:
  sarvam_api_key: "your-key-here"
```

### 2. Install on each machine
```bash
# Orin1
bash scripts/setup.sh orin1

# Orin2  
bash scripts/setup.sh orin2

# i7
bash scripts/setup.sh i7
```

### 3. Train / download wake word model
Option A – Use pretrained openWakeWord model (hey_mycroft, etc.) and rename:
```bash
# In config.yaml, point to any .onnx from openWakeWord
wake_word.model_path: "models/hey_mycroft_v0.1.onnx"
```

Option B – Train custom "Hey Vedha":
```bash
# Follow: https://github.com/dscripka/openWakeWord#training-new-models
# Place output at: models/hey_vedha.onnx
```

---

## Running

```bash
# On Orin1 (two processes)
python -m orin1.main &
python -m orin1.snapshot_server &

# On Orin2
python -m orin2.vlm_server

# On i7 (with ROS2 sourced)
source /opt/ros/humble/setup.bash
python -m ros_bridge.tool_executor
```

---

## VLM Model Options (Orin2)

| Model | Size (q4) | Vision | Tool calls | Notes |
|-------|-----------|--------|------------|-------|
| `llama3.2-vision:11b-instruct-q4_K_M` | ~7GB | ✅ | ✅ Native | **Recommended** |
| `llava:7b-q4_K_M` | ~4GB | ✅ | JSON prompt | Good fallback |
| `moondream2` | ~2GB | ✅ | JSON prompt | Fastest, less capable |
| `minicpm-v:8b-q4_K_M` | ~5GB | ✅ | JSON prompt | Good balance |

> **Note:** llama3.2-vision supports native tool calling via Ollama's `/api/chat` tools parameter.
> For other models, tool calling falls back to JSON-in-prompt parsing (see `tools.py`).

---

## Sarvam Saaras STT

Sign up at [https://sarvam.ai](https://sarvam.ai) for an API key.

Supports languages:
- `en-IN` – Indian English
- `ta-IN` – Tamil
- `hi-IN` – Hindi
- (and 10+ other Indian languages)

Set `stt.language_code` in `config.yaml` accordingly.

---

## Customizing Vedha's Personality

Edit `orin2/system_prompt.txt`. Key sections:
- **IDENTITY** – Name, personality, language mix
- **TRIGGER MODES** – How she behaves on person detection vs wake word
- **TOOL USAGE RULES** – Which tools to chain in which order
- **CONVERSATION STYLE** – Response length, tone, phrasing

---

## Adding New Tools

1. Add schema to `orin2/tools.py → TOOL_SCHEMAS`
2. Add endpoint to `ros_bridge/tool_executor.py`
3. Add description to `system_prompt.txt` tool table
4. Add to `config.yaml → tools.enabled`
