#!/usr/bin/env python3
"""Real-time face tracking that sends position vectors over serial USB.

The backend detects the largest face and streams a packet like:

    POS,<x_error>,<y_error>,<confidence>\n
This is meant for an Arduino Nano that drives servos from the received
normalized position vector.
"""

from __future__ import annotations

import argparse
import os
import re
import site
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

try:
    import serial
except ImportError:
    serial = None

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
    serial_port: Optional[str] = None
    serial_baudrate: int = 115200


RESOLUTION_PRESETS: List[Tuple[str, Tuple[int, int]]] = [
    ("640 x 480", (640, 480)),
    ("1280 x 720", (1280, 720)),
    ("1920 x 1080", (1920, 1080)),
    ("320 x 240", (320, 240)),
]


def detect_serial_ports() -> List[str]:
    ports: List[str] = []
    for pattern in ("ttyUSB*", "ttyACM*", "ttyS*"):
        for path in Path("/dev").glob(pattern):
            ports.append(str(path))
    for path in Path("/dev/pts").glob("*"):
        if path.name.isdigit():
            ports.append(str(path))
    return sorted(set(ports))


def get_serial_port_label(port: str) -> str:
    if "USB" in port:
        return f"{port} (USB)"
    if "ACM" in port:
        return f"{port} (Nano)"
    return port


class SerialController:
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200, timeout: float = 1.0) -> None:
        if serial is None:
            raise RuntimeError("pyserial is not installed. Install with: pip install pyserial")
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[Any] = None
        self.connected = False

    def connect(self) -> bool:
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(0.5)
            self.connected = True
            print(f"[SerialController] Connected to {self.port} at {self.baudrate} baud")
            return True
        except Exception as exc:
            print(f"[SerialController] Failed to connect: {exc}")
            self.connected = False
            return False

    def disconnect(self) -> None:
        if self.ser and getattr(self.ser, "is_open", False):
            try:
                self.ser.close()
            finally:
                self.connected = False
                print("[SerialController] Disconnected")

    def send_position_vector(self, x_error: float, y_error: float, confidence: float) -> bool:
        if not self.connected or not self.ser:
            return False
        try:
            msg = f"POS,{x_error:.4f},{y_error:.4f},{confidence:.2f}\n"
            self.ser.write(msg.encode("utf-8"))
            print(f"[SerialController] tx {msg.strip()}")
            return True
        except Exception as exc:
            print(f"[SerialController] Error sending position vector: {exc}")
            self.connected = False
            return False


def list_linux_video_indices(max_cameras: int) -> List[int]:
    indices: List[int] = []
    for path in Path("/dev").glob("video*"):
        match = re.fullmatch(r"video(\d+)", path.name)
        if match:
            indices.append(int(match.group(1)))
    indices = sorted(set(indices))
    return [idx for idx in indices if idx < max_cameras]


def discover_cameras(max_cameras: int = 10) -> List[CameraInfo]:
    cameras: List[CameraInfo] = []
    candidate_indices = list_linux_video_indices(max_cameras) if os.name == "posix" else []
    if not candidate_indices:
        candidate_indices = list(range(max_cameras))

    for idx in candidate_indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if os.name == "posix" else cv2.VideoCapture(idx)
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
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception:
        print("Tkinter not available. Falling back to terminal selection.")
        for i, cam in enumerate(cameras, start=1):
            print(f"[{i}] Camera {cam.index} ({cam.width}x{cam.height})")
        choice = input("Select camera number (empty to cancel): ").strip()
        if not choice.isdigit():
            return None
        idx = int(choice) - 1
        if not (0 <= idx < len(cameras)):
            return None

        print("Resolution options:")
        for i, (label, _) in enumerate(RESOLUTION_PRESETS, start=1):
            print(f"[{i}] {label}")
        resolution_choice = input("Select resolution number (empty for 1280x720): ").strip()
        resolution = RESOLUTION_PRESETS[1][1]
        if resolution_choice.isdigit() and 1 <= int(resolution_choice) <= len(RESOLUTION_PRESETS):
            resolution = RESOLUTION_PRESETS[int(resolution_choice) - 1][1]

        distance_choice = input("Use full-range face detection? [y/N]: ").strip().lower()
        model_selection = 1 if distance_choice in {"y", "yes", "1", "full"} else 0
        print("Serial ports detected:")
        for i, port in enumerate(detect_serial_ports(), start=1):
            print(f"  [{i}] {port}")
        serial_port = input("Serial port for Arduino (empty for none, or type a tty path): ").strip() or None
        baud_text = input("Serial baudrate [115200]: ").strip()
        baudrate = int(baud_text) if baud_text.isdigit() else 115200
        return CameraSelection(cameras[idx], resolution, model_selection, serial_port, baudrate)

    selected_index = {"value": None}
    selected_resolution = {"value": RESOLUTION_PRESETS[1][1]}
    selected_model = {"value": 0}
    selected_port = {"value": None}
    selected_baudrate = {"value": 115200}

    ports = detect_serial_ports()
    port_labels = ["None (tracking only)"] + [get_serial_port_label(p) for p in ports]
    port_values = [None] + ports

    root = tk.Tk()
    root.title("Select Webcam and Serial Settings")
    root.resizable(False, False)
    root.geometry("560x560")

    frame = tk.Frame(root, padx=12, pady=12)
    frame.pack(fill="both", expand=True)

    tk.Label(frame, text="Face Tracking Setup", font=("DejaVu Sans", 11, "bold")).pack(anchor="w")
    tk.Label(frame, text="Configure camera, resolution, and serial output.").pack(anchor="w", pady=(4, 8))

    camera_frame = tk.LabelFrame(frame, text="Camera", padx=10, pady=10)
    camera_frame.pack(fill="x", pady=(0, 10))

    resolution_names = [label for label, _ in RESOLUTION_PRESETS]
    resolution_var = tk.StringVar(value=resolution_names[1])
    model_var = tk.StringVar(value="Short-range")
    serial_var = tk.StringVar(value=port_labels[0])
    baudrate_var = tk.StringVar(value="115200")

    tk.Label(camera_frame, text="Resolution:").grid(row=0, column=0, sticky="w")
    tk.OptionMenu(camera_frame, resolution_var, *resolution_names).grid(row=0, column=1, sticky="w", padx=8)

    tk.Label(camera_frame, text="Distance mode:").grid(row=1, column=0, sticky="w", pady=(8, 0))

    def toggle_model() -> None:
        selected_model["value"] = 1 - selected_model["value"]
        model_var.set("Full-range" if selected_model["value"] else "Short-range")

    tk.Button(camera_frame, textvariable=model_var, width=16, command=toggle_model).grid(row=1, column=1, sticky="w", padx=8, pady=(8, 0))

    serial_frame = tk.LabelFrame(frame, text="Serial", padx=10, pady=10)
    serial_frame.pack(fill="x", pady=(0, 10))

    tk.Label(serial_frame, text="Port:").grid(row=0, column=0, sticky="w")
    tk.OptionMenu(serial_frame, serial_var, *port_labels).grid(row=0, column=1, sticky="w", padx=8)

    tk.Label(serial_frame, text="Or manual tty:").grid(row=1, column=0, sticky="w", pady=(8, 0))
    manual_port_var = tk.StringVar(value="")
    tk.Entry(serial_frame, textvariable=manual_port_var, width=24).grid(row=1, column=1, sticky="w", padx=8, pady=(8, 0))

    tk.Label(serial_frame, text="Baudrate:").grid(row=2, column=0, sticky="w", pady=(8, 0))
    tk.Entry(serial_frame, textvariable=baudrate_var, width=10).grid(row=2, column=1, sticky="w", padx=8, pady=(8, 0))

    list_label = tk.Label(frame, text="Available Cameras:", font=("DejaVu Sans", 9, "bold"))
    list_label.pack(anchor="w", pady=(8, 4))
    list_frame = tk.Frame(frame)
    list_frame.pack(fill="both", expand=True)

    scrollbar = tk.Scrollbar(list_frame, orient="vertical")
    listbox = tk.Listbox(list_frame, height=8, exportselection=False, yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side="right", fill="y")
    listbox.pack(side="left", fill="both", expand=True)
    for cam in cameras:
        listbox.insert("end", f"Camera {cam.index} ({cam.width}x{cam.height})")
    if cameras:
        listbox.selection_set(0)

    def sync_resolution() -> None:
        for label, size in RESOLUTION_PRESETS:
            if label == resolution_var.get():
                selected_resolution["value"] = size
                break

    def sync_serial() -> None:
        manual = manual_port_var.get().strip()
        if manual:
            selected_port["value"] = manual
            return
        choice = serial_var.get()
        for i, label in enumerate(port_labels):
            if label == choice:
                selected_port["value"] = port_values[i]
                break

    def sync_baudrate() -> None:
        try:
            selected_baudrate["value"] = int(baudrate_var.get().strip())
        except ValueError:
            selected_baudrate["value"] = 115200

    resolution_var.trace_add("write", lambda *_: sync_resolution())
    serial_var.trace_add("write", lambda *_: sync_serial())
    manual_port_var.trace_add("write", lambda *_: sync_serial())
    baudrate_var.trace_add("write", lambda *_: sync_baudrate())
    sync_resolution()
    sync_serial()
    sync_baudrate()

    def open_selected() -> None:
        selection = listbox.curselection()
        if not selection:
            messagebox.showinfo("No selection", "Please select a camera first.")
            return
        selected_index["value"] = selection[0]
        root.destroy()

    def cancel() -> None:
        selected_index["value"] = None
        root.destroy()

    button_frame = tk.Frame(frame)
    button_frame.pack(fill="x", pady=(10, 0))
    tk.Button(button_frame, text="Open", width=12, command=open_selected).pack(side="right", padx=(8, 0))
    tk.Button(button_frame, text="Cancel", width=12, command=cancel).pack(side="right")
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()

    idx = selected_index["value"]
    if idx is None or not (0 <= idx < len(cameras)):
        return None
    return CameraSelection(
        camera=cameras[idx],
        resolution=selected_resolution["value"],
        model_selection=selected_model["value"],
        serial_port=selected_port["value"],
        serial_baudrate=selected_baudrate["value"],
    )


def open_camera(camera_index: int) -> cv2.VideoCapture:
    return cv2.VideoCapture(camera_index, cv2.CAP_V4L2) if os.name == "posix" else cv2.VideoCapture(camera_index)


def build_face_detector(model_selection: int, min_detection_confidence: float) -> Any:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is not installed. Install opencv-contrib-python==4.10.0.84, mediapipe==0.10.14, and numpy==1.26.4 first."
        ) from exc
    return mp.solutions.face_detection.FaceDetection(
        model_selection=model_selection,
        min_detection_confidence=min_detection_confidence,
    )


def select_largest_detection(detections: List[Any], frame_width: int, frame_height: int) -> Optional[Tuple[int, int, int, int, float]]:
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
    serial_port: Optional[str] = None,
    serial_baudrate: int = 115200,
) -> None:
    cap = open_camera(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    if target_resolution is not None:
        width, height = target_resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        cap.set(cv2.CAP_PROP_FPS, 30.0)

    serial_ctrl: Optional[SerialController] = None
    if serial_port:
        serial_ctrl = SerialController(port=serial_port, baudrate=serial_baudrate)
        if not serial_ctrl.connect():
            serial_ctrl = None

    detector = None
    window_name = f"Face Tracking - Camera {camera_index}"
    try:
        detector = build_face_detector(model_selection, min_detection_confidence)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
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
            cv2.drawMarker(frame, (cx, cy), (0, 180, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            target = select_largest_detection(results.detections, w, h) if results.detections else None
            if target is not None:
                x, y, fw, fh, score = target
                tx, ty = x + fw // 2, y + fh // 2
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (70, 240, 90), 2)
                cv2.circle(frame, (tx, ty), 4, (70, 240, 90), -1)
                cv2.line(frame, (cx, cy), (tx, ty), (70, 240, 90), 2)
                err_x = (tx - cx) / max(1, cx)
                err_y = (ty - cy) / max(1, cy)
                packet = f"POS,{err_x:+.4f},{err_y:+.4f},{score:.2f}"
                cv2.putText(
                    frame,
                    packet,
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if serial_ctrl:
                    serial_ctrl.send_position_vector(err_x, err_y, score)
            else:
                cv2.putText(frame, "No face detected", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 120, 240), 2, cv2.LINE_AA)
                if serial_ctrl:
                    serial_ctrl.send_position_vector(0.0, 0.0, 0.0)

            now = time.time()
            fps = 1.0 / max(1e-6, now - prev)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        if detector is not None:
            detector.close()
        cv2.destroyAllWindows()
        if serial_ctrl:
            serial_ctrl.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time OpenCV + MediaPipe face tracking")
    parser.add_argument("--max-cameras", type=int, default=10, help="How many camera indices to probe.")
    parser.add_argument("--camera-index", type=int, default=None, help="Skip selection and open this camera index directly.")
    parser.add_argument("--model-selection", type=int, choices=[0, 1], default=0, help="MediaPipe face detector model.")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5, help="Minimum confidence required for a face detection.")
    parser.add_argument("--serial-port", type=str, default=None, help="Serial port path for Arduino communication.")
    parser.add_argument("--serial-baudrate", type=int, default=115200, help="Baud rate for serial communication.")
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
        serial_port = selected.serial_port
        serial_baudrate = selected.serial_baudrate
    else:
        cam_index = args.camera_index
        model_selection = args.model_selection
        target_resolution = None
        serial_port = args.serial_port
        serial_baudrate = args.serial_baudrate

    run_face_tracking(
        camera_index=cam_index,
        model_selection=model_selection,
        min_detection_confidence=args.min_detection_confidence,
        target_resolution=target_resolution,
        serial_port=serial_port,
        serial_baudrate=serial_baudrate,
    )


if __name__ == "__main__":
    main()
