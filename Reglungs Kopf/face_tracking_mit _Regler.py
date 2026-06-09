"""Real-time face tracking using OpenCV.

Features:
- Discover connected webcams.
- Startup camera selection window.
- Real-time face tracking visualization.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
import time
from dataclasses import dataclass
from typing import List, Optional

# OpenCV HighGUI uses Qt in many Linux wheels. Set sensible defaults before
# importing cv2 to avoid repeated runtime warnings in common desktop setups.
if os.name == "posix":
	os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
	if "QT_QPA_FONTDIR" not in os.environ:
		for font_dir in (
			"/usr/share/fonts/truetype/dejavu",
			"/usr/share/fonts/truetype",
			"/usr/share/fonts",
		):
			if os.path.isdir(font_dir):
				os.environ["QT_QPA_FONTDIR"] = font_dir
				break

import cv2
import serial

ARDUINO_PORT = "COM6"
BAUD_RATE = 115200

PAN_CENTER = 90
PAN_MIN = 60
PAN_MAX = 120

TILT_CENTER = 90
TILT_MIN = 75
TILT_MAX = 105

Kp_x = 18.0
Kp_y = 12.0

deadzone_x = 0.05
deadzone_y = 0.05

direction_x = 1
direction_y = 1


@dataclass
class CameraInfo:
	index: int
	width: int
	height: int


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


def select_camera_window(cameras: List[CameraInfo]) -> Optional[CameraInfo]:
	"""Open a native OS dialog and let user select a camera."""
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
			return cameras[idx]
		return None

	selected_index = {"value": None}

	root = tk.Tk()
	root.title("Select Webcam")
	root.resizable(False, False)
	root.geometry("520x340")

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
		text="Double-click a camera or select one and click Open.",
		font=("DejaVu Sans", 9),
	)
	subtitle_label.pack(anchor="w", pady=(4, 8))

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
		return cameras[idx]
	return None


def build_face_detector() -> cv2.CascadeClassifier:
	cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
	detector = cv2.CascadeClassifier(cascade_path)
	if detector.empty():
		raise RuntimeError(f"Failed to load Haar cascade at {cascade_path}")
	return detector


def run_face_tracking(camera_index: int) -> None:
	"""Run the real-time face tracking loop."""
	detector = build_face_detector()
	arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.01)
	time.sleep(2)

	
	pan_angle = PAN_CENTER
	tilt_angle = TILT_CENTER

	last_send_time = 0.0
	send_interval = 0.04
	cap = cv2.VideoCapture(camera_index)
	if not cap.isOpened():
		raise RuntimeError(f"Could not open camera {camera_index}")

	window_name = f"Face Tracking - Camera {camera_index}"
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

	prev = time.time()
	while True:
		ok, frame = cap.read()
		if not ok or frame is None:
			print("Frame read failed, stopping.")
			break

		frame = cv2.flip(frame, 1)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		faces = detector.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=6,
			minSize=(60, 60),
		)

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
		if len(faces) > 0:
			# Largest face is usually the main subject.
			target = max(faces, key=lambda b: b[2] * b[3])

		if target is not None:
			x, y, fw, fh = target
			tx, ty = x + fw // 2, y + fh // 2
			cv2.rectangle(frame, (x, y), (x + fw, y + fh), (70, 240, 90), 2)
			cv2.circle(frame, (tx, ty), 4, (70, 240, 90), -1)
			cv2.line(frame, (cx, cy), (tx, ty), (70, 240, 90), 2)

			
			err_x = (tx - cx) / max(1, cx)
			err_y = (ty - cy) / max(1, cy)

			if abs(err_x) > deadzone_x:
				pan_angle = pan_angle + direction_x * Kp_x * err_x

			if abs(err_y) > deadzone_y:
				tilt_angle = tilt_angle + direction_y * Kp_y * err_y

			pan_angle = max(PAN_MIN, min(PAN_MAX, pan_angle))
			tilt_angle = max(TILT_MIN, min(TILT_MAX, tilt_angle))

			now_send = time.time()
			if now_send - last_send_time >= send_interval:
				last_send_time = now_send
				arduino.write(f"{int(pan_angle)},{int(tilt_angle)}\n".encode())

			
			cv2.putText(
				frame,
				f"x={err_x:+.2f}, y={err_y:+.2f}, pan={int(pan_angle)}, tilt={int(tilt_angle)}",
				(16, 32),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(255, 255, 255),
				2,
				cv2.LINE_AA,
			)
		else:
			if pan_angle > PAN_CENTER:
				pan_angle -= 0.5
			elif pan_angle < PAN_CENTER:
				pan_angle += 0.5

			if tilt_angle > TILT_CENTER:
				tilt_angle -= 0.5
			elif tilt_angle < TILT_CENTER:
				tilt_angle += 0.5

			now_send = time.time()
			if now_send - last_send_time >= send_interval:
				last_send_time = now_send
				arduino.write(f"{int(pan_angle)},{int(tilt_angle)}\n".encode())


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

	arduino.write(f"{PAN_CENTER},{TILT_CENTER}\n".encode())
	time.sleep(0.3)
	arduino.close()
	
	cap.release()
	cv2.destroyWindow(window_name)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Real-time OpenCV face tracking")
	parser.add_argument(
		"--mode",
		default="gui",
		choices=["gui"],
		help="Execution mode. GUI is currently supported.",
	)
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
		cam_index = selected.index
	else:
		cam_index = args.camera_index

	run_face_tracking(camera_index=cam_index)
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
