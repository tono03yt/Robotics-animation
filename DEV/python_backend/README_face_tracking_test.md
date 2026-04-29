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
pip install opencv-contrib-python==4.10.0.84 mediapipe==0.10.14 numpy==1.26.4
```

Dependencies used:
- `opencv-contrib-python==4.10.0.84`
- `mediapipe==0.10.14`
- `numpy==1.26.4`

## Run

From project root:

```bash
source .venv/bin/activate
python DEV/python_backend/face_tracking_test.py
```

If you run the script with system `python3`, it will relaunch itself with the local `.venv` when available.

Startup window controls:
- Camera list
	- select which webcam to open
- Resolution menu
	- choose the capture resolution before start
- Distance mode button
	- toggles short-range vs full-range MediaPipe face detection

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

Examples:

```bash
python DEV/python_backend/face_tracking_test.py --max-cameras 12
python DEV/python_backend/face_tracking_test.py --camera-index 0 --model-selection 1
```

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
