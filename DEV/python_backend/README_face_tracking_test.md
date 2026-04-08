# face_tracking_test.py

TensorFlow-based face tracking app that:
- reads camera input (laptop webcam or external camera URL),
- detects faces with MTCNN,
- draws a green bounding box and confidence label,
- runs locally in a desktop window,
- captures camera frames at fixed 480p (`640x480`).
- keeps display smooth by running detection on a lightweight, rate-limited inference path.
- now supports an OpenCV-native detector backend for lower-latency tracking.

## File Location

`DEV/python_backend/face_tracking_test.py`

## Requirements

Recommended setup:

```bash
# from repository root
cd DEV/python_backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "opencv-python>=4.9.0" "tensorflow>=2.13.0" "mtcnn>=0.1.1"
```

If you are already in `DEV/python_backend`, do not run `cd DEV/python_backend` again.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "opencv-python>=4.9.0" "tensorflow>=2.13.0" "mtcnn>=0.1.1"
```

## Run

### 1) GUI camera selection mode (recommended)

```bash
python face_tracking_test.py --mode gui
```

- A small GUI opens and lists connected cameras.
- Select a camera and click `Start`.
- Tracking opens in an OpenCV window.
- Press `q` or `ESC` to quit.

### 2) Local mode with fixed webcam index

```bash
python face_tracking_test.py --mode local --camera-index 0
```

### 3) Local mode with external camera URL

```bash
python face_tracking_test.py --mode local --camera-url "http://<camera-ip>:<port>/video"
```

RTSP example:

```bash
python face_tracking_test.py --mode local --camera-url "rtsp://user:pass@<camera-ip>:554/stream"
```

## Useful Flags

- `--mode` : `gui` or `local` (default `gui`)
- `--camera-index` : webcam index (default `0`)
- `--camera-url` : external camera URL; overrides camera index
- `--min-confidence` : minimum face confidence for drawing boxes (default `0.90`)
- `--probe-max-index` : max camera index to scan in GUI mode (default `8`)
- `--detect-width` : width used for face inference (default `320`, lower is faster)
- `--detect-interval-ms` : minimum delay between detections (default `45`, higher is smoother/lower CPU)
- `--detector-backend` : `opencv` (fast default) or `mtcnn` (higher accuracy, slower)
- `--bbox-smoothing` : box smoothing factor `0.0-0.95` (default `0.45`)
- `--hold-last-ms` : keeps last valid box briefly through misses (default `250`)
- `--max-missed-detections` : tolerated missed detections before hide (default `5`)

## Real-Time Tuning

For maximum smoothness on weaker hardware:

```bash
python face_tracking_test.py --mode local --camera-index 0 --detector-backend opencv --detect-width 256 --detect-interval-ms 55 --bbox-smoothing 0.55
```

If detection still flickers, increase persistence:

```bash
python face_tracking_test.py --mode local --camera-index 0 --detector-backend opencv --hold-last-ms 350 --max-missed-detections 8 --bbox-smoothing 0.60
```

For better detection precision on stronger hardware:

```bash
python face_tracking_test.py --mode local --camera-index 0 --detector-backend mtcnn --detect-width 384 --detect-interval-ms 30 --bbox-smoothing 0.35
```

## Comparison With OpenCV Example

Compared with the OpenCV UR face-tracking example, this script now adopts the same speed-first principles:

- OpenCV-first detection path (`--detector-backend opencv`) for low latency
- continuous closed-loop updates using latest-frame semantics (drop stale frames)
- center-offset feedback (`dx`, `dy`) from frame center to face center
- smoothing to reduce jitter and improve visual stability

Main difference:

- this project tracks in a local camera window only (no robot control mapping)

## Troubleshooting

- Camera does not open:
  - try another `--camera-index` (`0`, `1`, `2`)
  - verify no other app is using the webcam
  - test your external URL in VLC/ffplay first
- `ModuleNotFoundError: No module named 'cv2'`:
  - install dependencies in the same interpreter you use to run:
  - `python -m pip install "opencv-python>=4.9.0" "tensorflow>=2.13.0" "mtcnn>=0.1.1"`
- Tracking window lags:
  - 480p is already fixed; close other camera-heavy apps and reduce background load
- No boxes detected:
  - improve lighting
  - move closer to camera
  - reduce threshold, for example `--min-confidence 0.75`

## Good Practice Notes

Current implementation is good for prototyping and local testing.
For production-grade deployment, improve by:
- pinning dependency versions exactly (for reproducibility)
- adding structured logging and health checks
- separating capture, inference, and GUI into independent components
- adding tests for argument parsing and camera probing
