# Real-Time Face Tracking (OpenCV + MediaPipe)

This backend script implements real-time face tracking with OpenCV and MediaPipe.

File:
- `DEV/python_backend/face_tracking_test.py`

What it does:
- Detects available webcams on startup.
- Opens a camera-selection window where you choose one connected webcam.
- Runs real-time MediaPipe face detection and tracks the largest detected face.
- Displays face-to-center tracking error values that can later be mapped to robot pan/tilt control.

## Install with `.venv` (Linux)

From project root (`/home/tono03/DEV/Robotics-facial-animation`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install opencv-contrib-python==4.10.0.84 mediapipe==0.10.14 numpy==1.26.4 pyserial
```

Dependencies used:
- `opencv-contrib-python==4.10.0.84`
- `mediapipe==0.10.14`
- `numpy==1.26.4`
- `pyserial` (for Arduino serial communication)

## Run

From project root:

```bash
source .venv/bin/activate
python DEV/python_backend/face_tracking_test.py
```

If you run the script with system `python3`, it will relaunch itself with the local `.venv` when available.

Startup window controls:
- **Camera list**: select which webcam to open
- **Resolution menu**: choose the capture resolution before start
- **Distance mode button**: toggles short-range vs full-range MediaPipe face detection
- **Serial Port selector** (new): choose USB port for Arduino (or "None" for tracking only)
- **Servo Control toggle** (new): enable/disable real-time servo commands to Arduino
- **Audio Recognition toggle** (new): enable/disable microphone input and audio playback (future)

## Useful options

- `--max-cameras 12`
	- Probe more camera indices during startup scan.
- `--camera-index 0`
	- Skip the selection window and use a fixed camera index.
- `--model-selection 0`
	- Short-range MediaPipe face detector for closer faces.
- `--model-selection 1`
	- Full-range MediaPipe face detector for farther faces.
- `--min-detection-confidence 0.6`
	- Increase or decrease detector strictness.
- `--serial-port /dev/ttyUSB0`
	- Enable servo control: connect to Arduino via USB serial.
	- When set, the script sends face-tracking X error values to Arduino automatically.
	- Without this option, servo control is disabled (tracking display only).
- `--serial-baudrate 115200`
	- Set serial communication baud rate (default 115200).

Examples:

```bash
python DEV/python_backend/face_tracking_test.py --max-cameras 12
python DEV/python_backend/face_tracking_test.py --camera-index 0 --model-selection 1

# With Arduino servo control
python DEV/python_backend/face_tracking_test.py --serial-port /dev/ttyUSB0
python DEV/python_backend/face_tracking_test.py --camera-index 0 --serial-port /dev/ttyUSB0 --serial-baudrate 115200
```

## Arduino Integration

The script includes two classes for communicating with Arduino Nano:

### SerialController
Manages bidirectional serial communication (USB/UART):
- Automatically sends face-tracking error (X-axis normalized: -1.0 to +1.0) to Arduino.
- When a face is detected, sends: `SERVO,<X_error>,<confidence>`
- When no face detected, sends: `SERVO,0.0,0.0`
- This allows Arduino to drive servos to center the detected face in the camera frame.

Usage:
```python
from face_tracking_test.py import SerialController

serial_ctrl = SerialController(port="/dev/ttyUSB0", baudrate=115200)
if serial_ctrl.connect():
    serial_ctrl.send_servo_command(x_error=-0.15, confidence=0.92)
    serial_ctrl.disconnect()
```

### AudioHandler
Background threads for receiving/sending audio data:
- Receives audio samples from Arduino microphone.
- Sends audio to Arduino for speaker playback (e.g., TTS output).
- Non-blocking queue-based design to avoid blocking the vision loop.

Usage:
```python
audio_handler = AudioHandler(serial_ctrl)
audio_handler.start()  # Start background threads

# Queue audio to send to Arduino speaker
audio_handler.queue_audio_to_send("PLAY", hex_encoded_audio_samples)

# Retrieve audio received from Arduino microphone
msg = audio_handler.get_received_message(block=False)

audio_handler.stop()   # Stop threads
```

### Serial Protocol Details

All messages are **newline-terminated** (`\n`) and use comma-delimited fields. Maximum serial latency is typically < 10 ms at 115200 baud.

#### Message Format
```
<COMMAND>,<field_1>,<field_2>,...,<field_N>\n
```

#### Python → Arduino Commands

##### `SERVO` — Face Tracking Error (sent every frame when servo enabled)
```
SERVO,<X_error_normalized>,<confidence>\n
```
- **X_error_normalized**: Float -1.0 to +1.0
  - `-1.0` = face far left
  - `0.0` = face centered
  - `+1.0` = face far right
- **confidence**: Float 0.0 to 1.0 (MediaPipe detection confidence)

_Examples:_
```
SERVO,-0.15,0.92     # Face 15% to left, 92% confidence
SERVO,0.0,0.0        # No face detected
SERVO,+0.28,0.87     # Face 28% to right, 87% confidence
```

##### `PLAY` — Audio Playback Command (audio recognition disabled for now)
```
PLAY,<duration_ms>,<sample_rate>,<hex_audio_data>\n
```
- **duration_ms**: Milliseconds of audio to play
- **sample_rate**: Audio sample rate (e.g., 8000, 16000)
- **hex_audio_data**: Hex-encoded PCM16 samples or MP3 chunk

_Example:_
```
PLAY,500,16000,ffc0ffc0ffc0...    # 500 ms @ 16 kHz
```

##### `REC` — Audio Recording Control (future)
```
REC,START\n
REC,STOP\n
```

#### Arduino → Python Telemetry

##### `AUDIO` — Microphone Audio Data (audio handler receives)
```
AUDIO,<sample_count>,<hex_samples>\n
```
- **sample_count**: Number of audio samples in hex string
- **hex_samples**: Hex-encoded PCM16 or PCM8 samples

_Example:_
```
AUDIO,256,ffc0ffc0ffc0...      # 256 samples in hex
```

##### `STAT` — System Status (optional telemetry)
```
STAT,<servo_pos_us>,<battery_mv>\n
```
- **servo_pos_us**: Current servo pulse width (1000–2000 μs)
- **battery_mv**: Battery voltage in millivolts

_Example:_
```
STAT,1500,5000    # Servo at 1500 us (center), 5.0 V battery
```

### Reference: Arduino Nano Serial Implementation

Typical Arduino Nano setup (pseudocode):
```cpp
#include <Servo.h>

Servo panServo;
volatile float gServoX = 0.0;
volatile float gConfidence = 0.0;

void setup() {
  Serial.begin(115200);  // Must match Python baudrate
  panServo.attach(9);    // Servo on pin 9
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    parseCommand(line);
  }
}

void parseCommand(const String& line) {
  if (line.startsWith("SERVO,")) {
    // Extract fields: SERVO,<X_error>,<confidence>
    int comma1 = line.indexOf(',');
    int comma2 = line.indexOf(',', comma1 + 1);
    
    float x_error = line.substring(comma1 + 1, comma2).toFloat();
    float confidence = line.substring(comma2 + 1).toFloat();
    
    // Map X error to servo position (e.g., -1.0 = 1000us, +1.0 = 2000us)
    int servo_us = 1500 + (int)(x_error * 500);  // Center ± 500 us
    panServo.writeMicroseconds(servo_us);
    
    gServoX = x_error;
    gConfidence = confidence;
  }
  else if (line.startsWith("PLAY,")) {
    // Future: decode and play audio
  }
}
```

### Architecture

```
Face Tracking Loop (Python)
  ├─ MediaPipe detects face
  ├─ Compute X error: err_x = (face_center_x - image_center_x) / image_center_x
  ├─ Every frame (if servo enabled):
  │   └─ SerialController.send_servo_command(err_x, confidence)
  │       └─ Transmit: SERVO,<err_x>,<confidence>\n
  │
  └─ Arduino Nano (receives each frame)
      ├─ Parse SERVO message
      ├─ Map error to servo PWM
      ├─ Pan servo to center face

AudioHandler (background threads, if audio enabled)
  ├─ _recv_loop:
  │   ├─ Listen for AUDIO,<samples> from Arduino microphone
  │   └─ Queue for Speech-to-Text (future)
  └─ _send_loop:
      ├─ Retrieve queued PLAY commands
      └─ Send to Arduino speaker amplifier (future)
```

### GUI Feature Toggles

The startup window now includes:
1. **Serial Port Selector** — auto-detects `/dev/ttyUSB*` and `/dev/ttyACM*` ports
2. **Servo Control Toggle** — enables/disables frame-by-frame servo commands
3. **Audio Recognition Toggle** — enables/disables microphone and speaker I/O
4. With GUI, all settings persist across restart (unlike CLI flags)

### To Implement in Arduino Firmware

1. **Servo Control** (required):
   - Parse `SERVO,<X>,<C>` messages
   - Map X error (−1.0 to +1.0) to servo range (typically 1000–2000 µs)
   - Send servo command via PWM or servo library
   - Optional: send back `STAT,<pos>,<volt>` for feedback

2. **Audio I/O** (future):
   - Implement microphone ADC sampling at configurable rate (8 kHz, 16 kHz)
   - Transmit audio in `AUDIO,<count>,<hex_data>` format periodically
   - Implement audio playback via DAC or PWM speaker driver
   - Receive `PLAY` commands from Python and play PCM data

### To Extend in Python

After Arduino has audio working:
1. Install speech-to-text: `pip install google-cloud-speech` or similar
2. In `run_face_tracking`, feed `audio_handler.get_received_message()` to STT
3. Send STT output to LLM via API
4. Generate TTS reply and queue playback: `audio_handler.queue_audio_to_send("PLAY", tts_hex_data)`

## Controls


Camera selection window:
- Native OS dialog opens with a camera list.
- Double-click camera or select one and press `Open`.
- Press `Cancel` to exit.

Tracking window:
- Press `q` or `ESC` to quit.

## Good Practice Note

Current approach is good for a lightweight prototype and fast local testing.

To make it better practice for production robotics use:
- Keep the vision pipeline isolated from the robot-control layer.
- Add structured logging and a config file for detector settings.
- Add camera warm-up/retry logic.
- Consider recording calibration data for the face-to-servo mapping.

## Troubleshooting (Linux)

If you saw warnings like:
- `Camera index out of range`
- `Ignoring XDG_SESSION_TYPE=wayland`
- `QFontDatabase: Cannot find font directory ... cv2/qt/fonts`

What they mean and what changed:
- `Camera index out of range`
	- Usually caused by probing non-existing camera indices.
	- Script now uses `/dev/video*` discovery first, which is better practice on Linux and reduces noisy probes.
- Wayland/X11 warning
	- OpenCV Qt windows often run with XCB by default.
	- Script now sets `QT_QPA_PLATFORM=xcb` by default to avoid backend ambiguity.
- Qt font directory warning
	- Some OpenCV wheels do not bundle Qt fonts.
	- Script now sets `QT_QPA_FONTDIR` to system font directories when available.

If GUI still fails, install system fonts once:

```bash
sudo apt update
sudo apt install -y fonts-dejavu-core
```

If MediaPipe is missing, reinstall the pinned package set with the install command above.
