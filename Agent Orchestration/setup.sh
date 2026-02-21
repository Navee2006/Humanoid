#!/bin/bash
# scripts/setup.sh
# Run once on each machine to download models and configure services.
# Usage: bash scripts/setup.sh [orin1|orin2|i7]

set -e
MACHINE=${1:-"orin1"}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")

echo "=== VEDHA Setup: $MACHINE ==="

# ─── Common ───────────────────────────────────────────────────
pip install -r "$ROOT_DIR/requirements.txt" --upgrade

if [ "$MACHINE" = "orin1" ]; then
    echo "--- Setting up Orin1 (Perception) ---"

    # YOLOv8 nano (person detection)
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
    echo "YOLOv8n downloaded"

    # openWakeWord – download pretrained models
    python -c "import openwakeword; openwakeword.utils.download_models()"
    echo "openWakeWord models downloaded"

    # Custom "Hey Vedha" wake word
    # Either train with: https://github.com/dscripka/openWakeWord#training-new-models
    # Or use nearest pretrained and adjust config threshold
    mkdir -p "$ROOT_DIR/models"
    echo "Place custom hey_vedha.onnx in $ROOT_DIR/models/"
    echo "Or set wake_word.model_path to a pretrained openWakeWord model path"

    # Test microphone
    echo "Testing microphone..."
    python -c "
import pyaudio
p = pyaudio.PyAudio()
info = p.get_default_input_device_info()
print(f'Default mic: {info[\"name\"]} @ {int(info[\"defaultSampleRate\"])}Hz')
p.terminate()
"

    # Test camera
    echo "Testing camera..."
    python -c "
import cv2
cap = cv2.VideoCapture(0)
ret, _ = cap.read()
print('Camera OK' if ret else 'Camera FAILED')
cap.release()
"

elif [ "$MACHINE" = "orin2" ]; then
    echo "--- Setting up Orin2 (VLM) ---"

    # Install Ollama
    if ! command -v ollama &> /dev/null; then
        echo "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        # Start Ollama service
        systemctl --user enable ollama
        systemctl --user start ollama
        sleep 3
    fi

    echo "Pulling VLM model (this may take a while on first run)..."
    # llama3.2-vision:11b-instruct-q4_K_M is ~7GB
    # For constrained RAM: try moondream2 (~2GB) or llava:7b-q4_K_M
    ollama pull llama3.2-vision:11b-instruct-q4_K_M

    echo "Testing Ollama..."
    ollama run llama3.2-vision:11b-instruct-q4_K_M "Hello, respond with just OK" --nowordwrap

elif [ "$MACHINE" = "i7" ]; then
    echo "--- Setting up i7 (ROS Bridge) ---"

    # Check ROS2
    if ! command -v ros2 &> /dev/null; then
        echo "ERROR: ROS2 not found. Install ROS2 Humble first."
        echo "https://docs.ros.org/en/humble/Installation.html"
        exit 1
    fi
    echo "ROS2 found: $(ros2 --version 2>&1 | head -1)"

    # Install Piper TTS
    if ! command -v piper &> /dev/null; then
        echo "Installing Piper TTS..."
        pip install piper-tts
    fi

    echo "Downloading Piper voice model..."
    python -c "
import piper
# Download voice
from piper import PiperVoice
voice = PiperVoice.load('en_US-lessac-medium', download=True)
print('Voice downloaded')
"

    echo ""
    echo "i7 setup complete. Start the tool executor with:"
    echo "  python -m ros_bridge.tool_executor"
fi

echo ""
echo "=== Setup complete for $MACHINE ==="
echo ""
echo "Start commands:"
echo "  Orin1: python -m orin1.main & python -m orin1.snapshot_server"
echo "  Orin2: python -m orin2.vlm_server"
echo "  i7:    python -m ros_bridge.tool_executor"
