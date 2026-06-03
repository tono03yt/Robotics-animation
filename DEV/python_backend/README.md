# Robotics Face Tracking Backend

Real-time face tracking, LLM chat, wakeword detection, and German TTS pipeline for a robotics animatronics project. Communicates with a C program via Unix Domain Socket (IPC).

***

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Layout](#project-layout)
- [Setup: Python Virtual Environment](#setup-python-virtual-environment)
- [Setup: API Key](#setup-api-key)
- [Setup: C Receiver](#setup-c-receiver)
- [Quick Start](#quick-start)
- [GUI Walkthrough](#gui-walkthrough)
- [IPC Communication](#ipc-communication)
- [Log Output Reference](#log-output-reference)
- [CLI Reference](#cli-reference)
- [Troubleshooting](#troubleshooting)

***

## Overview

The backend runs as a Python process that:

1. Detects and tracks the largest face in a webcam frame using **MediaPipe**
2. Computes normalized X/Y error vectors from the frame centre
3. Sends positional data and LLM animation commands to a **C receiver** over a **Unix Domain Socket**
4. Optionally listens for the **"Alexa" wake word**, transcribes speech with **Faster-Whisper**, queries **OpenRouter LLM**, and speaks the reply via **Piper TTS**

```
┌─────────────────────────────────┐        Unix Domain Socket
│        Python Backend           │  ──────────────────────────►  C Program
│                                 │  newline-delimited JSON         (robot)
│  Camera → MediaPipe → Tracking  │
│  Mic → Wakeword → STT → LLM    │
│  LLM → TTS → Speaker            │
└─────────────────────────────────┘
```

> **Auto-reconnect:** The Python backend retries the socket connection every 2 seconds if the C receiver is not yet running. Start them in any order.

***

## Features

| Feature | Technology | Notes |
|---|---|---|
| Face detection | MediaPipe `FaceDetection` | Short-range (0) or full-range (1) mode |
| Positional output | Unix Domain Socket JSON | Normalized −1.0 … +1.0 per axis |
| Wake word | openWakeWord (`alexa`) | Optional, toggled in GUI |
| Speech-to-text | Faster-Whisper `large-v3-turbo` | German (`de`) language |
| LLM | OpenRouter (`gpt-4o-mini`) | Reads key from `api_key_openrouterai` |
| Text-to-speech | Piper TTS (`de_DE-thorsten-high`) | ~65 MB model, auto-downloaded on first run |
| IPC | Unix Domain Socket (`AF_UNIX`) | Auto-reconnect, no restart required |

***

## Project Layout

This is the exact directory layout of the project as it stands:

```
python_backend/
│
├── backend_tracking.py        ← Main Python pipeline (auto-bootstraps .venv)
├── api_key_openrouterai       ← OpenRouter API key — plain text, one line, NO extension
│
├── ipc_reciever.c             ← C receiver source code
├── ipc_reciever               ← C receiver binary (compiled from above)
│
├── tts_german.py              ← Standalone TTS test script
├── wakeword_detection.py      ← Standalone wakeword test script
├── face_tracking_test.py      ← Standalone face tracking test script
├── recording_test.py          ← Standalone audio recording test
├── recording_test_cont.py     ← Continuous recording test variant
├── serial_io_test_client.py   ← Legacy serial I/O test (replaced by IPC)
│
├── README.md                  ← This file
│
├── tts_models/                ← Auto-created on first TTS use
│   ├── de_DE-thorsten-high.onnx         Piper voice model (~65 MB, auto-downloaded)
│   └── de_DE-thorsten-high.onnx.json    Model config (auto-downloaded)
│
├── wakeword_moddel_creation/  ← Custom wakeword model training resources
│
├── .venv/                     ← Python virtual environment (create with steps below)
│   └── bin/python             The interpreter the backend re-launches into
│
└── __pycache__/               ← Python bytecode cache (auto-generated, ignore)
```

### Key files explained

| File | Purpose |
|---|---|
| `backend_tracking.py` | The main entry point — run this to start the full pipeline |
| `api_key_openrouterai` | Your OpenRouter API key (see [Setup: API Key](#setup-api-key)) |
| `ipc_reciever.c` / `ipc_reciever` | C socket receiver — compile once, run before or alongside the Python backend |
| `tts_models/` | Piper TTS voice model files — created automatically, do not edit |
| `.venv/` | Isolated Python environment — created once during setup |

***

## Setup: Python Virtual Environment

### 1. Install Python 3.12

```bash
# Ubuntu / Debian (Ubuntu 24.04 ships Python 3.12 by default)
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev -y

# Verify
python3.12 --version
# Python 3.12.x
```

### 2. Navigate to the project folder

```bash
cd ~/DEV/Robotics-facial-animation/DEV/python_backend
```

### 3. Create the virtual environment

```bash
python3.12 -m venv .venv
```

This creates a `.venv/` folder inside the project. The backend script **automatically detects and re-launches itself** inside this venv — you do not need to activate it to run the script.

### 4. Activate (for installing packages only)

```bash
source .venv/bin/activate
# Prompt becomes: (.venv) tono03@machine:~/DEV/.../python_backend$
```

### 5. Upgrade pip

```bash
pip install --upgrade pip
```

### 6. Install all dependencies

```bash
# Core vision
pip install opencv-python mediapipe numpy

# Audio pipeline
pip install sounddevice faster-whisper openwakeword

# ML backends
pip install torch onnxruntime

# LLM + TTS
pip install requests piper-tts

# Suppress MediaPipe protobuf deprecation warning
pip install "protobuf>=4.21"
```

> **GPU (NVIDIA CUDA):** If you have a CUDA GPU, use these instead:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> pip install onnxruntime-gpu    # replaces onnxruntime
> ```
> The backend automatically uses the GPU for Whisper STT when CUDA is available.

### 7. Install system audio libraries

```bash
sudo apt install portaudio19-dev libportaudio2 ffmpeg -y
```

### 8. Verify the installation

```bash
python -c "import cv2, mediapipe, faster_whisper, openwakeword, sounddevice, piper; print('All imports OK')"
# All imports OK
```

### 9. Deactivate when done

```bash
deactivate
```

***

## Setup: API Key

The LLM feature requires a free API key from **OpenRouter**.

### 1. Get a key

Go to [openrouter.ai/keys](https://openrouter.ai/keys), sign in, and create a new key. Copy the key — it looks like:

```
sk-or-v1-abc123def456...
```

### 2. Create the key file

The file **must** be named exactly `api_key_openrouterai` — no extension, no quotes.

```bash
# Make sure you are in the project folder
cd ~/DEV/Robotics-facial-animation/DEV/python_backend

# Create the file with your key
echo "sk-or-v1-YOUR_KEY_HERE" > api_key_openrouterai
```

### 3. Verify the file

```bash
cat api_key_openrouterai
# sk-or-v1-YOUR_KEY_HERE

ls -la api_key_openrouterai
# -rw-r--r-- 1 tono03 tono03 44 Jun  3 14:00 api_key_openrouterai
```

The file must contain **exactly one line** — the raw key string with no extra spaces, no quotes, no newline at the end (the `echo` command adds a newline which is fine — the backend strips it automatically).

### 4. Confirm project folder looks correct

After completing all setup steps, your directory should look like this:

```bash
ls
# api_key_openrouterai   recording_test_cont.py
# backend_tracking.py    recording_test.py
# face_tracking_test.py  serial_io_test_client.py
# ipc_reciever           tts_german.py
# ipc_reciever.c         tts_models/
# __pycache__/           wakeword_detection.py
# README.md              wakeword_moddel_creation/
# .venv/                 ← created during venv setup
```

If `.venv/` is present and `api_key_openrouterai` exists, setup is complete.

***

## Setup: C Receiver

The C receiver has no external dependencies. Compile it once:

```bash
cd ~/DEV/Robotics-facial-animation/DEV/python_backend

gcc -Wall -o ipc_reciever ipc_reciever.c

# Verify binary exists
ls -lh ipc_reciever
# -rwxr-xr-x 1 tono03 tono03 18K Jun  3 14:00 ipc_reciever
```

***

## Quick Start

After completing all setup steps above:

### Terminal 1 — Start the C receiver

```bash
cd ~/DEV/Robotics-facial-animation/DEV/python_backend
./ipc_reciever
```

```
  ╔══════════════════════════════════════════════╗
  ║     IPC Receiver — Socket Path Setup         ║
  ╚══════════════════════════════════════════════╝

  Default path : /tmp/robot_pipeline.sock

  Enter socket path (or press Enter for default):
  > [Enter]

[IPC] ✓ Listening on: /tmp/robot_pipeline.sock
[IPC]   Start the Python backend now — it will auto-connect.
```

### Terminal 2 — Start the Python backend

```bash
cd ~/DEV/Robotics-facial-animation/DEV/python_backend
python backend_tracking.py
```

A GUI window appears. Select your camera, verify the socket path matches, click **Start Tracking**.

### Expected output

**Python terminal:**
```
[IPC] Connected to Unix socket: /tmp/robot_pipeline.sock
[Tracking] POS 0.0938,0.2667 conf=0.96
[Tracking] POS 0.0922,0.2583 conf=0.95
```

**C receiver terminal:**
```
[IPC] Python connected.
[POS]  x=0.0938    y=0.2667    conf=0.96
[POS]  x=0.0922    y=0.2583    conf=0.95
```

***

## GUI Walkthrough

### Start-up window

| Section | What to set |
|---|---|
| **Camera** | Select webcam from the list; choose resolution and distance mode |
| **Unix Socket IPC** | Socket path (default `/tmp/robot_pipeline.sock`); leave blank to disable IPC |
| **Log Display** | Filter terminal output: All / Tracking / LLM / Custom |
| **Audio & Chat** | Enable wakeword mic mode; enable/disable TTS voice output |

### Chat window (default mode)

- Type text and press **Enter** (or *Send to LLM*) to query the LLM
- Toggle **Enable Mic (Wakeword)** to activate voice — say *"Alexa"* then your command
- Toggle **Enable TTS** to have replies spoken aloud in German

### Voice Pipeline window (wakeword mode)

Opened when *"Use Wakeword & Mic"* is ticked in the start-up GUI:

| Indicator | Meaning |
|---|---|
| Mic Level bar | Live microphone input volume |
| Speech ON/OFF | Silero VAD detected voice activity |
| Wake Word YES/NO | "Alexa" keyword confirmed |
| WW Score bar | Confidence score of wake word detection (threshold: 0.5) |
| Last Transcription | Faster-Whisper output text |

***

## IPC Communication

All messages are **newline-delimited JSON** sent over `AF_UNIX SOCK_STREAM`.

### Socket settings

| Setting | Value |
|---|---|
| Default path | `/tmp/robot_pipeline.sock` |
| Protocol | `AF_UNIX`, `SOCK_STREAM` |
| Encoding | UTF-8, one JSON object per line |
| Auto-reconnect | Python retries every 2 s if C receiver is not ready |

***

### Message: `pos` — Face position

Sent every frame where a face is detected (~25–30 Hz).

```json
{"type": "pos", "x": 0.0938, "y": 0.2667, "conf": 0.96}
```

| Field | Type | Range | Meaning |
|---|---|---|---|
| `type` | string | `"pos"` | Message discriminator |
| `x` | float | −1.0 … +1.0 | Horizontal error from centre (negative = left, positive = right) |
| `y` | float | −1.0 … +1.0 | Vertical error from centre (negative = above, positive = below) |
| `conf` | float | 0.0 … 1.0 | MediaPipe detection confidence |

**Coordinate system:**

```
        x = −1.0 (left)
             │
 y = −1.0 ───┼─── y = +1.0
 (top)       │       (bottom)
             │
        x = +1.0 (right)

 Origin (0, 0) = frame centre
```

No face detected → zero vector:

```json
{"type": "pos", "x": 0.0, "y": 0.0, "conf": 0.0}
```

**Real values from a live session:**

```json
{"type": "pos", "x":  0.0938, "y":  0.2667, "conf": 0.96}
{"type": "pos", "x":  0.0922, "y":  0.2583, "conf": 0.95}
{"type": "pos", "x":  0.0844, "y":  0.2639, "conf": 0.96}
{"type": "pos", "x":  0.0797, "y":  0.2639, "conf": 0.96}
{"type": "pos", "x":  0.0813, "y":  0.2556, "conf": 0.96}
{"type": "pos", "x":  0.0797, "y":  0.2528, "conf": 0.95}
{"type": "pos", "x":  0.0828, "y":  0.2528, "conf": 0.95}
{"type": "pos", "x":  0.0781, "y":  0.2528, "conf": 0.94}
```

***

### Message: `anim` — Animation command

Sent once after the LLM produces a response.

```json
{"type": "anim", "animation": "waving", "text": "Hello! How can I assist you today?"}
```

| Field | Type | Values | Meaning |
|---|---|---|---|
| `type` | string | `"anim"` | Message discriminator |
| `animation` | string | `"speech"` or `"waving"` | Robot animation to trigger |
| `text` | string | any | LLM reply text (spoken by Piper TTS if enabled) |

| `animation` | When triggered | Suggested robot behaviour |
|---|---|---|
| `"speech"` | General verbal reply | Mouth/jaw movement, idle animation |
| `"waving"` | Greeting detected by LLM | Raise and wave arm |

***

### Full message flow — live session trace

```
Python terminal                          C receiver terminal
──────────────────────────────────────   ──────────────────────────────────
[Tracking] POS 0.0938,0.2667 conf=0.96  →  [POS]  x=0.0938   y=0.2667   conf=0.96
[Tracking] POS 0.0922,0.2583 conf=0.95  →  [POS]  x=0.0922   y=0.2583   conf=0.95
[Tracking] POS 0.0844,0.2639 conf=0.96  →  [POS]  x=0.0844   y=0.2639   conf=0.96
[LLM] querying for: wave
[LLM] animation=waving reply=Hello! How can I assist you today?
[ANIM] waving                            →  [ANIM] animation=waving  text=Hello! How can I assist you today?
[IPC] Disconnected                          [IPC] Python disconnected — waiting for reconnect...
[Tracking] POS 0.0813,0.2500 conf=0.94  →  [IPC] Python connected.
                                            [POS]  x=0.0813   y=0.2500   conf=0.94
```

***

### C receiver — robot logic hooks

```c
static void handle_pos(float x, float y, float conf)
{
    // x, y: normalized error from frame centre (−1.0 to +1.0)
    if (conf < 0.5f) return;           // ignore low-confidence frames
    if (x >  0.05f) pan_right(x);
    if (x < -0.05f) pan_left(-x);
    if (y >  0.05f) tilt_down(y);
    if (y < -0.05f) tilt_up(-y);
}

static void handle_anim(const char *animation, const char *text)
{
    if (strcmp(animation, "waving") == 0) wave_arm();
    if (strcmp(animation, "speech") == 0) open_mouth_sync(text);
}
```

***

## Log Output Reference

| Prefix | Colour | Meaning |
|---|---|---|
| `[Tracking] POS x,y conf=c` | Blue | Face position vector sent via IPC |
| `[Tracking] No face detected` | Cyan | Zero vector sent |
| `[LLM] querying for: …` | Magenta | Input dispatched to OpenRouter |
| `[LLM] animation=… reply=…` | Magenta | LLM response parsed |
| `[ANIM] …` | Magenta | Animation command sent via IPC |
| `[STT] …` | Green | Faster-Whisper transcription |
| `[WW] Wake word detected!` | Green | "Alexa" heard, recording started |
| `[IPC] Connected to …` | Cyan | Socket connected to C receiver |
| `[IPC] Disconnected` | Cyan | C receiver closed connection |
| `[IPC] Socket not ready … will retry` | Cyan | C receiver not running yet, retrying in 2 s |

***

## CLI Reference

```bash
python backend_tracking.py [options]

Options:
  --camera-index N                    Use camera N directly, skips GUI
  --model-selection 0|1               0 = short-range (default), 1 = full-range
  --min-detection-confidence FLOAT    Detection threshold 0.0–1.0 (default: 0.5)
  --socket-path PATH                  Unix socket path (default: /tmp/robot_pipeline.sock)
  --max-cameras N                     Camera indices to probe (default: 10)
  --help-backend                      Print feature summary and exit
```

**Headless example** — no GUI:

```bash
python backend_tracking.py \
  --camera-index 0 \
  --socket-path /tmp/robot_pipeline.sock \
  --model-selection 0 \
  --min-detection-confidence 0.6
```

***

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `WARN: can't open camera by index` | OpenCV probing `/dev/video1` which doesn't exist | Harmless — select the correct camera in the GUI |
| `[IPC] Socket not ready — will retry` | C receiver not started yet | Start `./ipc_reciever` — Python reconnects automatically |
| `[LLM] No API key found` | Key file missing or wrong name | File must be named exactly `api_key_openrouterai` with no extension |
| `[LLM] no response` | Invalid or expired API key | Check key at openrouter.ai/keys; re-paste into `api_key_openrouterai` |
| `AudioFeatures unexpected keyword` | Outdated openwakeword | `source .venv/bin/activate && pip install --upgrade openwakeword` |
| No TTS audio | `piper-tts` missing or wrong audio device | `pip install piper-tts`; check `aplay -l` for output devices |
| Black camera window | Camera in use by another app | Close other apps using the webcam |
| `ModuleNotFoundError` on startup | Packages installed outside `.venv` | Run `source .venv/bin/activate` and re-run `pip install` steps |