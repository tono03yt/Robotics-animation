"""TensorFlow-based face tracking for local desktop use.

Features:
- Detects faces with MTCNN (TensorFlow-backed) and draws colored boxes.
- Local desktop mode with camera selection GUI and OpenCV preview window.
- Supports local webcam and optional external camera URLs (HTTP/RTSP).
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from typing import Optional, Union

import cv2
from mtcnn import MTCNN

try:
	import tkinter as tk
	from tkinter import messagebox, ttk
except Exception:  # pragma: no cover - platform/runtime dependent
	tk = None
	messagebox = None
	ttk = None


CameraSource = Union[int, str]


@dataclass
class TrackingConfig:
	"""Runtime configuration for face tracking modes."""

	mode: str = "gui"
	camera_index: int = 0
	camera_url: Optional[str] = None
	frame_width: int = 640
	frame_height: int = 480
	min_confidence: float = 0.90
	probe_max_index: int = 8
	detect_width: int = 320
	detect_interval_ms: int = 45
	detector_backend: str = "opencv"
	bbox_smoothing: float = 0.45
	hold_last_ms: int = 250
	max_missed_detections: int = 5


class FaceTracker:
	"""Face detection helper that returns only the largest face box."""

	def __init__(self, min_confidence: float, backend: str = "opencv") -> None:
		self.min_confidence = min_confidence
		self.backend = backend
		self.detector = None
		self.haar = None

		if self.backend == "mtcnn":
			self.detector = MTCNN()
		else:
			haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
			self.haar = cv2.CascadeClassifier(haar_path)
			if self.haar.empty():
				raise RuntimeError(f"Unable to load Haar cascade from '{haar_path}'")

	def _detect_mtcnn(self, frame_bgr):
		frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
		detections = self.detector.detect_faces(frame_rgb)

		best_detection = None
		best_area = -1
		for detection in detections:
			confidence = detection.get("confidence", 0.0)
			if confidence < self.min_confidence:
				continue

			x, y, w, h = detection["box"]
			w = max(0, w)
			h = max(0, h)
			area = w * h
			if area > best_area:
				best_area = area
				best_detection = detection

		if best_detection is None:
			return None

		x, y, w, h = best_detection["box"]
		return {
			"confidence": float(best_detection.get("confidence", 0.0)),
			"box": (max(0, x), max(0, y), max(0, w), max(0, h)),
		}

	def _detect_opencv(self, frame_bgr):
		gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
		gray = cv2.equalizeHist(gray)
		faces = self.haar.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(40, 40),
			flags=cv2.CASCADE_SCALE_IMAGE,
		)

		if len(faces) == 0:
			return None

		x, y, w, h = max(faces, key=lambda b: int(b[2]) * int(b[3]))
		return {
			# Haar cascade does not provide confidence; keep a synthetic score for UI consistency.
			"confidence": 1.0,
			"box": (int(x), int(y), int(w), int(h)),
		}

	def detect_largest_face(self, frame_bgr):
		if self.backend == "mtcnn":
			return self._detect_mtcnn(frame_bgr)
		return self._detect_opencv(frame_bgr)


class LatestFrameCamera:
	"""Read camera frames continuously and keep only the latest one."""

	def __init__(self, cap: cv2.VideoCapture) -> None:
		self.cap = cap
		self._lock = threading.Lock()
		self._latest_frame = None
		self._running = False
		self._thread: Optional[threading.Thread] = None

	def start(self) -> None:
		self._running = True
		self._thread = threading.Thread(target=self._reader, daemon=True)
		self._thread.start()

	def _reader(self) -> None:
		while self._running:
			ok, frame = self.cap.read()
			if not ok or frame is None:
				time.sleep(0.001)
				continue
			with self._lock:
				self._latest_frame = frame

	def read(self):
		with self._lock:
			if self._latest_frame is None:
				return False, None
			return True, self._latest_frame.copy()

	def stop(self) -> None:
		self._running = False
		if self._thread is not None:
			self._thread.join(timeout=0.5)


class LatestInferenceWorker:
	"""Detect faces in a dedicated thread and expose the latest result."""

	def __init__(
		self,
		tracker: FaceTracker,
		detect_width: int,
		detect_interval_ms: int,
		hold_last_ms: int,
		max_missed_detections: int,
	) -> None:
		self.tracker = tracker
		self.detect_width = max(160, detect_width)
		self.detect_interval_sec = max(0.01, detect_interval_ms / 1000.0)
		self.hold_last_sec = max(0.0, hold_last_ms / 1000.0)
		self.max_missed_detections = max(0, max_missed_detections)
		self._lock = threading.Lock()
		self._latest_input = None
		self._latest_detection = None
		self._running = False
		self._thread: Optional[threading.Thread] = None
		self._last_detect_ts = 0.0
		self._last_good_detection_ts = 0.0
		self._missed_count = 0

	def start(self) -> None:
		self._running = True
		self._thread = threading.Thread(target=self._worker, daemon=True)
		self._thread.start()

	def submit(self, frame_bgr) -> None:
		with self._lock:
			self._latest_input = frame_bgr

	def read(self):
		with self._lock:
			return self._latest_detection

	def _build_detection_frame(self, frame_bgr):
		h, w = frame_bgr.shape[:2]
		if w <= self.detect_width:
			return frame_bgr, 1.0

		scale = self.detect_width / float(w)
		new_h = max(1, int(h * scale))
		resized = cv2.resize(frame_bgr, (self.detect_width, new_h), interpolation=cv2.INTER_LINEAR)
		return resized, scale

	def _worker(self) -> None:
		while self._running:
			with self._lock:
				if self._latest_input is None:
					frame = None
				else:
					frame = self._latest_input
					self._latest_input = None

			if frame is None:
				time.sleep(0.001)
				continue

			now = time.perf_counter()
			if (now - self._last_detect_ts) < self.detect_interval_sec:
				continue

			detect_frame, scale = self._build_detection_frame(frame)
			detection = self.tracker.detect_largest_face(detect_frame)

			if detection is not None and scale > 0.0 and scale != 1.0:
				x, y, w, h = detection["box"]
				inv_scale = 1.0 / scale
				detection["box"] = (
					int(x * inv_scale),
					int(y * inv_scale),
					int(w * inv_scale),
					int(h * inv_scale),
				)

			if detection is not None:
				self._missed_count = 0
				self._last_good_detection_ts = now
				with self._lock:
					self._latest_detection = detection
			else:
				self._missed_count += 1
				keep_last = (
					(now - self._last_good_detection_ts) <= self.hold_last_sec
					and self._missed_count <= self.max_missed_detections
				)
				if not keep_last:
					with self._lock:
						self._latest_detection = None
			self._last_detect_ts = now

	def stop(self) -> None:
		self._running = False
		if self._thread is not None:
			self._thread.join(timeout=0.5)


def open_camera(source: CameraSource, config: TrackingConfig) -> cv2.VideoCapture:
	cap = cv2.VideoCapture(source)
	cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

	# Request a high capture FPS. Camera/driver will clamp to supported maximum.
	cap.set(cv2.CAP_PROP_FPS, 120)
	if isinstance(source, int):
		cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

	if not cap.isOpened():
		raise RuntimeError(
			f"Unable to open camera source '{source}'. "
			"Use --camera-index for webcam or --camera-url for external stream."
		)

	return cap


def find_connected_cameras(max_index: int = 8) -> list[int]:
	connected: list[int] = []
	for index in range(max_index + 1):
		cap = cv2.VideoCapture(index)
		ok, _ = cap.read()
		cap.release()
		if ok:
			connected.append(index)
	return connected


def select_camera_with_gui(max_index: int = 8) -> Optional[int]:
	if tk is None or ttk is None:
		print("Tkinter GUI is not available. Falling back to camera index 0.")
		return 0

	selected: dict[str, Optional[int]] = {"index": None}
	cameras = find_connected_cameras(max_index)

	root = tk.Tk()
	root.title("Face Tracking Camera Selector")
	root.geometry("420x180")
	root.resizable(False, False)

	frame = ttk.Frame(root, padding=14)
	frame.pack(fill="both", expand=True)

	ttk.Label(frame, text="Select connected camera:").pack(anchor="w", pady=(0, 8))

	values = [f"Camera {idx} (index {idx})" for idx in cameras]
	combo = ttk.Combobox(frame, values=values, state="readonly")
	combo.pack(fill="x")
	if values:
		combo.current(0)

	status = ttk.Label(frame, text=f"Detected cameras: {cameras if cameras else 'none'}")
	status.pack(anchor="w", pady=(8, 8))

	button_row = ttk.Frame(frame)
	button_row.pack(fill="x")

	def refresh() -> None:
		new_cameras = find_connected_cameras(max_index)
		combo["values"] = [f"Camera {idx} (index {idx})" for idx in new_cameras]
		status.config(text=f"Detected cameras: {new_cameras if new_cameras else 'none'}")
		if new_cameras:
			combo.current(0)

	def start() -> None:
		current = combo.get()
		if not current:
			if messagebox:
				messagebox.showerror("No Camera", "No camera selected.")
			return
		idx_text = current.split("index ")[-1].rstrip(")")
		selected["index"] = int(idx_text)
		root.destroy()

	def cancel() -> None:
		root.destroy()

	ttk.Button(button_row, text="Refresh", command=refresh).pack(side="left")
	ttk.Button(button_row, text="Start", command=start).pack(side="right", padx=(8, 0))
	ttk.Button(button_row, text="Cancel", command=cancel).pack(side="right")

	root.mainloop()
	return selected["index"]


def run_local_tracking(config: TrackingConfig, source: CameraSource) -> None:
	tracker = FaceTracker(config.min_confidence, config.detector_backend)
	cap = open_camera(source, config)
	stream = LatestFrameCamera(cap)
	stream.start()
	inference = LatestInferenceWorker(
		tracker,
		config.detect_width,
		config.detect_interval_ms,
		config.hold_last_ms,
		config.max_missed_detections,
	)
	inference.start()
	cv2.setUseOptimized(True)

	fps_ema = 0.0
	prev_ts = time.perf_counter()
	smoothed_box = None
	beta = min(0.95, max(0.0, config.bbox_smoothing))

	window_name = "TensorFlow Face Tracking (press q to quit)"
	try:
		while True:
			ok, frame_bgr = stream.read()
			if not ok or frame_bgr is None:
				time.sleep(0.001)
				continue

			inference.submit(frame_bgr)
			annotated = frame_bgr
			detection = inference.read()
			if detection is not None:
				x, y, w, h = detection["box"]
				if w < 24 or h < 24:
					detection = None

			if detection is not None:
				x, y, w, h = detection["box"]
				if smoothed_box is None:
					smoothed_box = [float(x), float(y), float(w), float(h)]
				else:
					smoothed_box[0] = beta * smoothed_box[0] + (1.0 - beta) * float(x)
					smoothed_box[1] = beta * smoothed_box[1] + (1.0 - beta) * float(y)
					smoothed_box[2] = beta * smoothed_box[2] + (1.0 - beta) * float(w)
					smoothed_box[3] = beta * smoothed_box[3] + (1.0 - beta) * float(h)

				x = int(smoothed_box[0])
				y = int(smoothed_box[1])
				w = int(smoothed_box[2])
				h = int(smoothed_box[3])
				confidence = detection["confidence"]
				cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 210, 0), 2)
				label = f"Nearest Face {confidence:.2f}"
				cv2.putText(
					annotated,
					label,
					(x, max(24, y - 10)),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.6,
					(0, 210, 0),
					2,
					cv2.LINE_AA,
				)

				cx = x + (w // 2)
				cy = y + (h // 2)
				frame_cx = annotated.shape[1] // 2
				frame_cy = annotated.shape[0] // 2
				dx = cx - frame_cx
				dy = cy - frame_cy
				cv2.line(annotated, (frame_cx, frame_cy), (cx, cy), (0, 255, 255), 2)
				offset_text = f"dx={dx} dy={dy}"
				cv2.putText(
					annotated,
					offset_text,
					(10, 26),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.6,
					(0, 255, 255),
					2,
					cv2.LINE_AA,
				)
			else:
				smoothed_box = None

			cv2.drawMarker(
				annotated,
				(annotated.shape[1] // 2, annotated.shape[0] // 2),
				(255, 255, 0),
				markerType=cv2.MARKER_CROSS,
				markerSize=16,
				thickness=1,
			)

			now = time.perf_counter()
			delta = now - prev_ts
			prev_ts = now
			if delta > 0:
				inst_fps = 1.0 / delta
				fps_ema = inst_fps if fps_ema == 0.0 else (0.90 * fps_ema + 0.10 * inst_fps)

			fps_text = f"{fps_ema:.1f} FPS"
			font = cv2.FONT_HERSHEY_SIMPLEX
			scale = 0.65
			thickness = 2
			(text_w, text_h), baseline = cv2.getTextSize(fps_text, font, scale, thickness)
			x = max(10, annotated.shape[1] - text_w - 12)
			y = max(text_h + 10, 24)
			cv2.rectangle(
				annotated,
				(x - 6, y - text_h - 6),
				(x + text_w + 6, y + baseline + 4),
				(0, 0, 0),
				-1,
			)
			cv2.putText(
				annotated,
				fps_text,
				(x, y),
				font,
				scale,
				(60, 255, 60),
				thickness,
				cv2.LINE_AA,
			)

			cv2.imshow(window_name, annotated)

			key = cv2.waitKey(1) & 0xFF
			if key in (ord("q"), 27):  # q or ESC
				break
	finally:
		inference.stop()
		stream.stop()
		cap.release()
		cv2.destroyAllWindows()


def parse_args() -> TrackingConfig:
	parser = argparse.ArgumentParser(
		description="TensorFlow face tracking (local GUI/window only)."
	)
	parser.add_argument(
		"--mode",
		choices=["gui", "local"],
		default="gui",
		help="Run mode: gui (camera picker + local window) or local (window only)",
	)
	parser.add_argument(
		"--camera-index",
		type=int,
		default=0,
		help="OpenCV webcam index (used when --camera-url is not provided)",
	)
	parser.add_argument(
		"--camera-url",
		default=None,
		help="External camera stream URL (HTTP/RTSP)",
	)
	parser.add_argument(
		"--min-confidence",
		type=float,
		default=0.90,
		help="Minimum face confidence for drawing boxes",
	)
	parser.add_argument(
		"--probe-max-index",
		type=int,
		default=8,
		help="Highest camera index to scan in GUI mode",
	)
	parser.add_argument(
		"--detect-width",
		type=int,
		default=320,
		help="Inference frame width (smaller is faster; default 320)",
	)
	parser.add_argument(
		"--detect-interval-ms",
		type=int,
		default=45,
		help="Minimum milliseconds between detections (default 45)",
	)
	parser.add_argument(
		"--detector-backend",
		choices=["opencv", "mtcnn"],
		default="opencv",
		help="Face detector backend: opencv (faster) or mtcnn (more accurate)",
	)
	parser.add_argument(
		"--bbox-smoothing",
		type=float,
		default=0.45,
		help="Bounding box EMA smoothing factor 0.0-0.95 (higher is steadier)",
	)
	parser.add_argument(
		"--hold-last-ms",
		type=int,
		default=250,
		help="Keep last valid face for this many ms on brief misses (default 250)",
	)
	parser.add_argument(
		"--max-missed-detections",
		type=int,
		default=5,
		help="How many consecutive missed detections are tolerated (default 5)",
	)

	args = parser.parse_args()
	return TrackingConfig(
		mode=args.mode,
		camera_index=args.camera_index,
		camera_url=args.camera_url,
		min_confidence=args.min_confidence,
		probe_max_index=args.probe_max_index,
		detect_width=args.detect_width,
		detect_interval_ms=args.detect_interval_ms,
		detector_backend=args.detector_backend,
		bbox_smoothing=args.bbox_smoothing,
		hold_last_ms=args.hold_last_ms,
		max_missed_detections=args.max_missed_detections,
	)


def main() -> None:
	config = parse_args()

	if config.mode == "gui":
		selected_index = select_camera_with_gui(config.probe_max_index)
		if selected_index is None:
			print("No camera selected. Exiting.")
			return
		run_local_tracking(config, selected_index)
		return

	source: CameraSource = config.camera_index
	if config.camera_url:
		source = config.camera_url
	run_local_tracking(config, source)


if __name__ == "__main__":
	main()
