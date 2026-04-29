"""Real-time face tracking using OpenCV + MediaPipe.

Features:
- Discover connected webcams.
- Startup camera selection window.
- Real-time MediaPipe face detection and tracking visualization.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import site
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

# OpenCV HighGUI uses Qt in many Linux wheels. Set sensible defaults before
# importing cv2 to avoid repeated runtime warnings in common desktop setups.
if os.name == "posix":
	os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
	os.environ.setdefault("PYTHONNOUSERSITE", "1")
	if "QT_QPA_FONTDIR" not in os.environ:
		for font_dir in (
			"/usr/share/fonts/truetype/dejavu",
			"/usr/share/fonts/truetype",
			"/usr/share/fonts",
		):
			if os.path.isdir(font_dir):
				os.environ["QT_QPA_FONTDIR"] = font_dir
				break


def _remove_user_site_from_path() -> None:
	"""Avoid importing broken packages from the user's site-packages directory."""
	try:
		user_site = Path(site.getusersitepackages()).resolve()
	except Exception:
		return

	filtered: List[str] = []
	for entry in sys.path:
		try:
			resolved = Path(entry).resolve()
		except Exception:
			filtered.append(entry)
			continue
		if resolved == user_site or str(resolved).startswith(str(user_site)):
			continue
		filtered.append(entry)
	sys.path[:] = filtered


def _bootstrap_project_venv() -> None:
	"""Relaunch with the local project venv when not already running inside it."""
	if ".venv" in Path(sys.executable).parts:
		return

	venv_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
	if not venv_python.exists():
		return

	os.environ.setdefault("PYTHONNOUSERSITE", "1")
	os.execv(str(venv_python), [str(venv_python), *sys.argv])


_remove_user_site_from_path()
_bootstrap_project_venv()

import cv2


@dataclass
class CameraInfo:
	index: int
	width: int
	height: int


@dataclass
class CameraSelection:
	camera: CameraInfo
	resolution: Tuple[int, int]
	model_selection: int


RESOLUTION_PRESETS: List[Tuple[str, Tuple[int, int]]] = [
	("640 x 480", (640, 480)),
	("1280 x 720", (1280, 720)),
	("1920 x 1080", (1920, 1080)),
	("320 x 240", (320, 240)),
]


def list_linux_video_indices(max_cameras: int) -> List[int]:
	"""Return numeric indices from /dev/video* when available."""
	indices: List[int] = []
	for path in Path("/dev").glob("video*"):
		match = re.fullmatch(r"video(\d+)", path.name)
		if match:
			indices.append(int(match.group(1)))

	indices = sorted(set(indices))
	return [idx for idx in indices if idx < max_cameras]


def discover_cameras(max_cameras: int = 10) -> List[CameraInfo]:
	"""Probe camera indices and return working webcams."""
	cameras: List[CameraInfo] = []

	if os.name == "posix":
		candidate_indices = list_linux_video_indices(max_cameras=max_cameras)
		if not candidate_indices:
			candidate_indices = list(range(max_cameras))
	else:
		candidate_indices = list(range(max_cameras))

	for idx in candidate_indices:
		# CAP_V4L2 avoids some backend auto-probing noise on Linux.
		if os.name == "posix":
			cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
		else:
			cap = cv2.VideoCapture(idx)
		if not cap.isOpened():
			cap.release()
			continue

		ok, frame = cap.read()
		if ok and frame is not None:
			h, w = frame.shape[:2]
			cameras.append(CameraInfo(index=idx, width=w, height=h))
		cap.release()
	return cameras


def select_camera_window(cameras: List[CameraInfo]) -> Optional[CameraSelection]:
	"""Open a native OS dialog and let user select a camera and capture settings."""
	try:
		import tkinter as tk
		from tkinter import messagebox
	except Exception:
		print("Tkinter not available. Falling back to terminal selection.")
		for i, cam in enumerate(cameras, start=1):
			print(f"[{i}] Camera {cam.index} ({cam.width}x{cam.height})")
		choice = input("Select camera number (empty to cancel): ").strip()
		if not choice:
			return None
		if not choice.isdigit():
			return None
		idx = int(choice) - 1
		if 0 <= idx < len(cameras):
			selected_resolution = RESOLUTION_PRESETS[1][1]
			print("Resolution options:")
			for i, (label, _) in enumerate(RESOLUTION_PRESETS, start=1):
				print(f"[{i}] {label}")
			resolution_choice = input("Select resolution number (empty for 1280x720): ").strip()
			if resolution_choice.isdigit() and 1 <= int(resolution_choice) <= len(RESOLUTION_PRESETS):
				selected_resolution = RESOLUTION_PRESETS[int(resolution_choice) - 1][1]
			distance_choice = input("Use full-range face detection? [y/N]: ").strip().lower()
			model_selection = 1 if distance_choice in ("y", "yes", "1", "full") else 0
			return CameraSelection(
				camera=cameras[idx],
				resolution=selected_resolution,
				model_selection=model_selection,
			)
		return None

	selected_index = {"value": None}
	selected_resolution = {"value": RESOLUTION_PRESETS[1][1]}
	model_state = {"value": 0}

	root = tk.Tk()
	root.title("Select Webcam")
	root.resizable(False, False)
	root.geometry("620x430")

	frame = tk.Frame(root, padx=12, pady=12)
	frame.pack(fill="both", expand=True)

	title_label = tk.Label(
		frame,
		text="Select webcam for face tracking",
		font=("DejaVu Sans", 11, "bold"),
	)
	title_label.pack(anchor="w")

	subtitle_label = tk.Label(
		frame,
		text="Choose a camera, resolution, and face detection distance mode.",
		font=("DejaVu Sans", 9),
	)
	subtitle_label.pack(anchor="w", pady=(4, 8))

	settings_frame = tk.LabelFrame(frame, text="Start-up settings", padx=10, pady=10)
	settings_frame.pack(fill="x", pady=(0, 10))

	resolution_row = tk.Frame(settings_frame)
	resolution_row.pack(fill="x")
	tk = None
	try:
		from tkinter import ttk
	except Exception:
		pass

	resolution_label = tk.Label(resolution_row, text="Resolution:")
	resolution_label.pack(side="left")

	resolution_names = [label for label, _size in RESOLUTION_PRESETS]
	resolution_var = tk.StringVar(value=resolution_names[1])
	resolution_menu = tk.OptionMenu(resolution_row, resolution_var, *resolution_names)
	resolution_menu.pack(side="left", padx=(8, 0))

	model_row = tk.Frame(settings_frame)
	model_row.pack(fill="x", pady=(10, 0))

	model_label = tk.Label(model_row, text="Distance mode:")
	model_label.pack(side="left")

	model_button_text = tk.StringVar(value="Short-range")
	model_button = tk.Button(model_row, textvariable=model_button_text, width=16)
	model_button.pack(side="left", padx=(8, 0))

	def sync_resolution() -> None:
		choice = resolution_var.get()
		for label, size in RESOLUTION_PRESETS:
			if label == choice:
				selected_resolution["value"] = size
				return

	def toggle_distance_mode() -> None:
		model_state["value"] = 1 - model_state["value"]
		model_button_text.set("Full-range" if model_state["value"] else "Short-range")

	model_button.configure(command=toggle_distance_mode)
	resolution_var.trace_add("write", lambda *_args: sync_resolution())
	sync_resolution()

	list_frame = tk.Frame(frame)
	list_frame.pack(fill="both", expand=True)

	scrollbar = tk.Scrollbar(list_frame, orient="vertical")
	listbox = tk.Listbox(
		list_frame,
		height=10,
		exportselection=False,
		yscrollcommand=scrollbar.set,
		font=("DejaVu Sans", 10),
	)
	scrollbar.config(command=listbox.yview)
	scrollbar.pack(side="right", fill="y")
	listbox.pack(side="left", fill="both", expand=True)

	for cam in cameras:
		listbox.insert("end", f"Camera {cam.index} ({cam.width}x{cam.height})")

	if cameras:
		listbox.selection_set(0)

	def select_and_close() -> None:
		selection = listbox.curselection()
		if not selection:
			messagebox.showinfo("No selection", "Please select a camera first.")
			return
		selected_index["value"] = selection[0]
		root.destroy()

	def cancel_and_close() -> None:
		selected_index["value"] = None
		root.destroy()

	button_frame = tk.Frame(frame)
	button_frame.pack(fill="x", pady=(10, 0))

	open_button = tk.Button(button_frame, text="Open", width=12, command=select_and_close)
	open_button.pack(side="right", padx=(8, 0))

	cancel_button = tk.Button(button_frame, text="Cancel", width=12, command=cancel_and_close)
	cancel_button.pack(side="right")

	listbox.bind("<Double-1>", lambda _event: select_and_close())
	root.protocol("WM_DELETE_WINDOW", cancel_and_close)
	root.mainloop()

	idx = selected_index["value"]
	if idx is None:
		return None
	if 0 <= idx < len(cameras):
		return CameraSelection(
			camera=cameras[idx],
			resolution=selected_resolution["value"],
			model_selection=model_state["value"],
		)
	return None


def open_camera(camera_index: int) -> cv2.VideoCapture:
	"""Open a webcam with a Linux-friendly backend when possible."""
	if os.name == "posix":
		return cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
	return cv2.VideoCapture(camera_index)


def build_face_detector(
	model_selection: int,
	min_detection_confidence: float,
) -> Any:
	"""Create the MediaPipe face detector used by the tracking loop."""
	try:
		import mediapipe as mp
	except ImportError as exc:
		raise RuntimeError(
			"MediaPipe is not installed. Install opencv-contrib-python==4.10.0.84, mediapipe==0.10.14, and numpy==1.26.4 in your environment first."
		) from exc

	return mp.solutions.face_detection.FaceDetection(
		model_selection=model_selection,
		min_detection_confidence=min_detection_confidence,
	)


def select_largest_detection(
	detections: List[Any],
	frame_width: int,
	frame_height: int,
) -> Optional[Tuple[int, int, int, int, float]]:
	"""Return the largest face detection as a pixel-space bounding box."""
	best: Optional[Tuple[int, int, int, int, float]] = None
	best_area = -1

	for detection in detections:
		box = detection.location_data.relative_bounding_box
		x1 = max(0, int(box.xmin * frame_width))
		y1 = max(0, int(box.ymin * frame_height))
		x2 = min(frame_width, int((box.xmin + box.width) * frame_width))
		y2 = min(frame_height, int((box.ymin + box.height) * frame_height))
		box_width = max(0, x2 - x1)
		box_height = max(0, y2 - y1)
		area = box_width * box_height
		if area <= 0:
			continue

		score = float(detection.score[0]) if detection.score else 0.0
		if area > best_area:
			best_area = area
			best = (x1, y1, box_width, box_height, score)

	return best


def run_face_tracking(
	camera_index: int,
	model_selection: int,
	min_detection_confidence: float,
	target_resolution: Optional[Tuple[int, int]] = None,
) -> None:
	"""Run the real-time face tracking loop."""
	cap = open_camera(camera_index)
	if not cap.isOpened():
		raise RuntimeError(f"Could not open camera {camera_index}")

	if target_resolution is not None:
		width, height = target_resolution
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
		cap.set(cv2.CAP_PROP_FPS, 30.0)

	window_name = f"Face Tracking - Camera {camera_index}"
	window_open = False
	detector = None
	try:
		detector = build_face_detector(
			model_selection=model_selection,
			min_detection_confidence=min_detection_confidence,
		)
		cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
		window_open = True

		prev = time.time()
		while True:
			ok, frame = cap.read()
			if not ok or frame is None:
				print("Frame read failed, stopping.")
				break

			frame = cv2.flip(frame, 1)
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			results = detector.process(rgb)

			h, w = frame.shape[:2]
			cx, cy = w // 2, h // 2
			cv2.drawMarker(
				frame,
				(cx, cy),
				(0, 180, 255),
				markerType=cv2.MARKER_CROSS,
				markerSize=20,
				thickness=2,
			)

			target = None
			if results.detections:
				target = select_largest_detection(results.detections, w, h)

			if target is not None:
				x, y, fw, fh, score = target
				tx, ty = x + fw // 2, y + fh // 2
				cv2.rectangle(frame, (x, y), (x + fw, y + fh), (70, 240, 90), 2)
				cv2.circle(frame, (tx, ty), 4, (70, 240, 90), -1)
				cv2.line(frame, (cx, cy), (tx, ty), (70, 240, 90), 2)

				err_x = (tx - cx) / max(1, cx)
				err_y = (ty - cy) / max(1, cy)
				cv2.putText(
					frame,
					f"MediaPipe face score={score:.2f} | error x={err_x:+.2f}, y={err_y:+.2f}",
					(16, 32),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.65,
					(255, 255, 255),
					2,
					cv2.LINE_AA,
				)
			else:
				cv2.putText(
					frame,
					"No face detected",
					(16, 32),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.7,
					(70, 120, 240),
					2,
					cv2.LINE_AA,
				)

			now = time.time()
			fps = 1.0 / max(1e-6, now - prev)
			prev = now
			cv2.putText(
				frame,
				f"FPS: {fps:.1f}",
				(16, h - 16),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(255, 255, 255),
				2,
				cv2.LINE_AA,
			)

			cv2.imshow(window_name, frame)
			key = cv2.waitKey(1) & 0xFF
			if key in (27, ord("q")):
				break
	finally:
		cap.release()
		if detector is not None:
			detector.close()
		if window_open:
			cv2.destroyWindow(window_name)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Real-time OpenCV + MediaPipe face tracking")
	parser.add_argument(
		"--max-cameras",
		type=int,
		default=10,
		help="How many camera indices to probe.",
	)
	parser.add_argument(
		"--camera-index",
		type=int,
		default=None,
		help="Skip selection and open this camera index directly.",
	)
	parser.add_argument(
		"--model-selection",
		type=int,
		choices=[0, 1],
		default=0,
		help="MediaPipe face detector model: 0 for short-range, 1 for full-range.",
	)
	parser.add_argument(
		"--min-detection-confidence",
		type=float,
		default=0.5,
		help="Minimum confidence required for a face detection.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if args.camera_index is None:
		cameras = discover_cameras(max_cameras=args.max_cameras)
		if not cameras:
			print("No webcams detected. Connect a camera and retry.")
			return

		selected = select_camera_window(cameras)
		if selected is None:
			print("No camera selected. Exiting.")
			return
		cam_index = selected.camera.index
		model_selection = selected.model_selection
		target_resolution = selected.resolution
	else:
		cam_index = args.camera_index
		model_selection = args.model_selection
		target_resolution = None

	run_face_tracking(
		camera_index=cam_index,
		model_selection=model_selection,
		min_detection_confidence=args.min_detection_confidence,
		target_resolution=target_resolution,
	)
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
