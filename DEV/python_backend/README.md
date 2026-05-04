# Python Backend: Documentation and Hardware Implementation

This folder contains the complete backend stack for face tracking, serial communication, and optional local STT + LLM + TTS response generation.

## Contents

- `face_tracking_test.py`: Main backend (camera, MediaPipe face tracking, serial protocol, optional LLM pipeline)
- `serial_io_test_client.py`: Test client (serial monitor, bridge mode, text prompt injection for LLM tests)
- `api_key_openrouterai`: OpenRouter API key file (plain text key, one line)

## System Overview

The project supports two main functions:

1. Face tracking loop:
- Detect largest face in webcam frame
- Send normalized error vector via serial: `POS,<x_error>,<y_error>,<confidence>`

2. LLM/audio loop (optional):
- Backend receives `TEXT,<message>` (or `AUDIO,<base64>`)
- Uses OpenRouter LLM
- Produces structured response (`animation`, `text`)
- Synthesizes speech to WAV (espeak first, pyttsx3 fallback)
- Sends back `ANIM,...` and `AUDIO,...` packets

## Requirements

Use Python 3.10+ on Linux.

### Python packages

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install opencv-contrib-python==4.10.0.84 mediapipe==0.10.14 numpy==1.26.4
pip install pyserial requests openai-whisper pyttsx3
```

### System packages (recommended)

```bash
sudo apt-get update
sudo apt-get install -y python3-tk espeak ffmpeg
```

Notes:
- `python3-tk` is required for backend startup GUI.
- `espeak` is preferred TTS backend (most reliable on Linux).
- `pyttsx3` may install successfully but still fail at runtime depending on speech backend availability.

## OpenRouter API Setup

Create `api_key_openrouterai` in this folder with your key:

```text
sk-or-v1-...
```

Backend endpoint in use:
- `https://openrouter.ai/api/v1/chat/completions`

If you see DNS/network errors, verify internet access and DNS configuration.

## Backend Startup

Run from `DEV/python_backend`:

```bash
python face_tracking_test.py
```

The GUI allows selecting:
- Camera
- Resolution
- Distance mode (short-range / full-range)
- Serial port or manual tty path
- Baudrate
- Enable LLM checkbox
- Terminal log display preset

### Log display presets (backend GUI)

Same style as test client presets:
- `All`
- `Tracking`
- `LLM/Audio`
- `Custom` (comma-separated kinds)

Log output is color-coded in terminal for readability.

### CLI flags (backend)

```bash
python face_tracking_test.py \
  --camera-index 0 \
  --model-selection 0 \
  --min-detection-confidence 0.5 \
  --serial-port /dev/ttyUSB0 \
  --serial-baudrate 115200 \
  --enable-llm
```

Available flags:
- `--max-cameras`
- `--camera-index`
- `--model-selection {0,1}`
- `--min-detection-confidence`
- `--serial-port`
- `--serial-baudrate`
- `--enable-llm`
- `--help-backend`

## Test Client Startup

Run:

```bash
python serial_io_test_client.py
```

Interactive flow:
1. Select mode (`Real serial` or `Internal bridge`)
2. Select display preset (`All`, `Tracking`, `LLM/Audio`, `Custom`)
3. Optional text input prompt enable

When text input is enabled:
- Type at `[tx TEXT] >`
- Client shows `[input] ...`
- Packet is sent directly to backend
- Receive and decode backend responses in same terminal

Output is color-coded for readability.

## Serial Protocol

All packets are ASCII lines ending with `\n`.

### Backend -> Arduino/Test Client

1. Face tracking:

```text
POS,<x_error>,<y_error>,<confidence>
```

2. LLM response metadata:

```text
ANIM,<animation>,<text>
```

3. Audio payload chunks:

```text
AUDIO,<base64_chunk>
```

### Arduino/Test Client -> Backend

1. Text prompt:

```text
TEXT,<user_message>
```

2. Optional audio input:

```text
AUDIO,<base64_audio>
```

### Optional status/debug from Arduino

```text
STAT,<servo_us>,<millis>
[debug text...]
```

## Hardware Implementation Guide (Arduino + Servos)

The backend is hardware-agnostic, but expected behavior is:

1. Arduino opens serial at same baudrate as backend (default `115200`).
2. Arduino parses incoming `POS` continuously.
3. Convert `x_error` and `y_error` to servo target offsets with clamp/rate limits.
4. Optional: send `STAT` periodically for monitoring.
5. For interaction mode, Arduino sends `TEXT,<message>` to backend.
6. Arduino handles returned `ANIM` and `AUDIO` packets (playback pipeline is device-specific).

### Recommended control logic on hardware

- Apply smoothing/filtering to reduce jitter.
- Clamp command range before servo write.
- Add deadzone around zero to avoid servo hunting.
- Keep watchdog timeout for lost `POS` packets.

## Recommended Workflows

### A) Full software test without hardware (bridge mode)

1. Start test client:

```bash
python serial_io_test_client.py
```

2. Choose `Internal bridge`; copy printed `/dev/pts/X`.

3. Start backend and set serial port to that `/dev/pts/X` in GUI (or via CLI):

```bash
python face_tracking_test.py --serial-port /dev/pts/X --enable-llm
```

4. In test client LLM mode, type text and verify `ANIM` + `AUDIO` replies.

### B) Real hardware run

1. Connect Arduino and servos.
2. Identify serial device (`/dev/ttyUSB0`, `/dev/ttyACM0`, etc.).
3. Start backend with that port and optional `--enable-llm`.
4. Use test client only when needed (do not open same serial port in two apps simultaneously).

## Troubleshooting

### 1) No Start button in backend GUI

- Use latest version from this folder.
- Ensure `python3-tk` is installed.
- If GUI unavailable, script falls back to terminal selection.

### 2) `TEXT` sent but no response

- Ensure backend `Enable LLM` is active.
- Ensure backend and test client are connected to the same tty/bridge.
- Check API key file exists and is valid.
- Verify network access to OpenRouter endpoint.

### 3) `[TTS] No local TTS engine available`

- Install `espeak` and retest.
- If using `pyttsx3`, inspect detailed runtime error now printed by backend.

### 4) Port busy / cannot connect

- Only one process can open a serial device at a time.
- Close other serial monitors before starting backend.

## Quick Command Reference

```bash
# Activate env
source .venv/bin/activate

# Backend GUI mode
python face_tracking_test.py

# Backend CLI mode
python face_tracking_test.py --camera-index 0 --serial-port /dev/ttyUSB0 --enable-llm

# Test client interactive
python serial_io_test_client.py

# Test client one-shot text send
python serial_io_test_client.py --port /dev/pts/X --send-text "hello robot"
```
