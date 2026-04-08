# Real-Time Face Tracking (OpenCV)

This backend script implements real-time face tracking inspired by the OpenCV robotics demo workflow.

File:
- `DEV/python_backend/face_tracking_test.py`

What it does:
- Detects available webcams on startup.
- Opens a camera-selection window where you choose one connected webcam.
- Runs real-time face detection and tracks the largest detected face.
- Displays face-to-center tracking error values that can later be mapped to robot pan/tilt control.

## Install with `.venv` (Linux)

From project root (`/home/tono03/DEV/Robotics-facial-animation`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install opencv-python numpy
```

Dependencies used:
- `opencv-python`
- `numpy`

## Run

From project root:

```bash
source .venv/bin/activate
python DEV/python_backend/face_tracking_test.py --mode gui
```

## Useful options

- `--max-cameras 12`
	- Probe more camera indices during startup scan.
- `--camera-index 0`
	- Skip the selection window and use a fixed camera index.

Examples:

```bash
python DEV/python_backend/face_tracking_test.py --mode gui --max-cameras 12
python DEV/python_backend/face_tracking_test.py --mode gui --camera-index 0
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
- Replace Haar cascades with a DNN face detector for higher robustness.
- Add camera warm-up/retry logic and structured logging.
- Separate vision tracking output from robot control into clear modules.

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
