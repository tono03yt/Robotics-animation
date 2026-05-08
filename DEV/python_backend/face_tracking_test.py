#!/usr/bin/env python3
"""Real-time face tracking with local STT -> LLM -> TTS on the PC.

The backend detects the largest face and reports tracking plus animation
type in the terminal (e.g. waving, speech). Microphone input and audio
output are handled locally on the PC.
"""

from __future__ import annotations

import argparse
import multiprocessing
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
except Exception:
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
import threading
import json
import tempfile
import subprocess
import requests
import shutil

_pyttsx3_engine = None

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
    log_display_kinds: Optional[List[str]] = None
    mic_device: Optional[str] = None
    mic_continuous: bool = False


RESOLUTION_PRESETS: List[Tuple[str, Tuple[int, int]]] = [
    ("640 x 480", (640, 480)),
    ("1280 x 720", (1280, 720)),
    ("1920 x 1080", (1920, 1080)),
    ("320 x 240", (320, 240)),
]

LOG_DISPLAY_PRESETS: List[Tuple[str, Optional[List[str]]]] = [
    ("All", None),
    ("Tracking", ["POS", "STAT"]),
    ("LLM/Audio", ["ANIM", "LLM", "STT", "TTS"]),
    ("Custom", []),
]

ACTIVE_LOG_KINDS: Optional[set[str]] = None


class Ansi:
    RESET = "\033[0m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def use_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def colorize(text: str, color: str) -> str:
    if not use_color():
        return text
    return f"{color}{text}{Ansi.RESET}"


def set_log_display_kinds(kinds: Optional[List[str]]) -> None:
    global ACTIVE_LOG_KINDS
    ACTIVE_LOG_KINDS = None if kinds is None else {kind.upper() for kind in kinds}


def log_line(kind: str, message: str, *, always: bool = False) -> None:
    if always:
        print(colorize(message, Ansi.RED), flush=True)
        return
    if ACTIVE_LOG_KINDS is None or kind.upper() in ACTIVE_LOG_KINDS:
        tag = kind.upper()
        if tag == "POS":
            print(colorize(message, Ansi.BLUE), flush=True)
        elif tag in {"STAT", "TEXT"}:
            print(colorize(message, Ansi.CYAN), flush=True)
        elif tag in {"LLM", "ANIM"}:
            print(colorize(message, Ansi.MAGENTA), flush=True)
        elif tag in {"AUDIO", "TTS"}:
            print(colorize(message, Ansi.GREEN), flush=True)
        elif tag == "STT":
            print(colorize(message, Ansi.YELLOW), flush=True)
        else:
            print(colorize(message, Ansi.DIM), flush=True)


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
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0) -> None:
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
            log_line("STAT", f"[Serial] Connected to {self.port} at {self.baudrate} baud")
            return True
        except Exception as exc:
            log_line("ERROR", f"[Serial] Failed to connect: {exc}", always=True)
            self.connected = False
            return False

    def disconnect(self) -> None:
        if self.ser and getattr(self.ser, "is_open", False):
            try:
                self.ser.close()
            finally:
                self.connected = False
                log_line("STAT", "[Serial] Disconnected")

    def send_position_vector(self, x_error: float, y_error: float, confidence: float) -> None:
        if not self.connected or not self.ser:
            return
        try:
            msg = f"POS,{x_error:.4f},{y_error:.4f},{confidence:.2f}\n"
            self.ser.write(msg.encode("utf-8"))
        except Exception as exc:
            log_line("ERROR", f"[Serial] POS send failed: {exc}", always=True)
            self.connected = False

    def send_animation(self, animation: str, text: Optional[str] = None) -> None:
        if not self.connected or not self.ser:
            return
        try:
            payload = text or ""
            msg = f"ANIM,{animation},{payload}\n"
            self.ser.write(msg.encode("utf-8"))
        except Exception as exc:
            log_line("ERROR", f"[Serial] ANIM send failed: {exc}", always=True)
            self.connected = False


# --- LLM / STT / TTS helpers ---
def _read_api_key(path: str = "api_key_openrouterai") -> Optional[str]:
    try:
        p = Path(__file__).resolve().parent / path
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def call_openrouter(system_prompt: str, user_prompt: str) -> Optional[str]:
    api_key = _read_api_key()
    if not api_key:
        log_line("LLM", "[LLM] No API key found in api_key_openrouterai")
        return None
    # OpenRouter's chat completions endpoint.
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30.0)
        r.raise_for_status()
        data = r.json()
        content = None
        if isinstance(data, dict):
            choices = data.get("choices")
            if choices and isinstance(choices, list) and choices[0].get("message"):
                content = choices[0]["message"].get("content")
            else:
                content = data.get("text")
        return content
    except Exception as exc:
        log_line("LLM", f"[LLM] request failed: {exc}")
        return None


def parse_llm_response_as_json(text: str) -> Optional[dict]:
    # Extract the first JSON object from the LLM reply and validate fields.
    try:
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        if not m:
            return None
        candidate = m.group(1)
        data = json.loads(candidate)
        if not isinstance(data, dict):
            return None
        anim = data.get("animation")
        txt = data.get("text")
        if anim not in {"speech", "waving"}:
            return None
        if not isinstance(txt, str):
            return None
        return {"animation": anim, "text": txt}
    except Exception:
        return None


def transcribe_audio_bytes(audio_bytes: bytes) -> Optional[str]:
    try:
        import whisper
    except Exception:
        return None
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp = f.name
        model = whisper.load_model("small")
        res = model.transcribe(tmp)
        return res.get("text")
    except Exception as exc:
        log_line("STT", f"[STT] transcribe failed: {exc}")
        return None
    finally:
        try:
            if tmp:
                os.unlink(tmp)
        except Exception:
            pass


def tts_synthesize_to_wav(text: str) -> Optional[bytes]:
    espeak = shutil.which("espeak")
    if espeak:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name
            subprocess.run([espeak, "-w", wav_path, text], check=True)
            data = Path(wav_path).read_bytes()
            try:
                os.unlink(wav_path)
            except Exception:
                pass
            return data
        except Exception as exc:
            log_line("TTS", f"[TTS] espeak failed: {exc}")
    try:
        import pyttsx3
        global _pyttsx3_engine
        try:
            engine = _pyttsx3_engine  # type: ignore
        except NameError:
            _pyttsx3_engine = None
            engine = None
        if engine is None:
            _pyttsx3_engine = pyttsx3.init()
            engine = _pyttsx3_engine
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        # pyttsx3 may return without producing a valid file on some Linux setups.
        if not Path(wav_path).exists():
            raise RuntimeError("pyttsx3 did not create output file")
        data = Path(wav_path).read_bytes()
        if len(data) < 44:
            raise RuntimeError("pyttsx3 produced an empty/invalid WAV")
        try:
            os.unlink(wav_path)
        except Exception:
            pass
        return data
    except Exception as exc:
        log_line("TTS", f"[TTS] pyttsx3 failed: {exc}")
    log_line("TTS", "[TTS] No local TTS engine available (install espeak or pyttsx3)")
    return None


def parse_mic_label(label: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not label or label == "None":
        return None, None
    if label.startswith("alsa:"):
        return "arecord", label.split(":", 1)[1]
    if label.startswith("ALSA "):
        return "arecord", label.replace("ALSA ", "", 1)
    return None, None


def record_audio_chunk(backend: str, index: Optional[str], duration: float = 4.0, sr: int = 16000) -> Optional[bytes]:
    if backend == "arecord":
        try:
            exe = shutil.which("arecord")
            if not exe:
                return None
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = f.name

            base_cmd = [
                exe,
                "-q",
                "-f",
                "S16_LE",
                "-r",
                str(sr),
                "-d",
                str(int(duration)),
            ]
            if index:
                base_cmd.extend(["-D", index])

            # Try mono first, then stereo if the device rejects mono.
            for channels in (1, 2):
                cmd = base_cmd + ["-c", str(channels), path]
                result = subprocess.run(cmd, check=False, capture_output=True, text=True)
                if result.returncode == 0:
                    data = Path(path).read_bytes()
                    try:
                        os.unlink(path)
                    except Exception:
                        pass
                    return data
            try:
                os.unlink(path)
            except Exception:
                pass
            log_line("STT", "[STT] arecord failed with mono and stereo")
            return None
        except Exception:
            return None
    return None


def play_audio_bytes(audio_bytes: bytes) -> bool:
    try:
        import sounddevice as sd
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
            f.write(audio_bytes)
        data, sr = sf.read(path, dtype="float32")
        sd.play(data, sr)
        sd.wait()
        try:
            os.unlink(path)
        except Exception:
            pass
        return True
    except Exception:
        pass

    for player in ("aplay", "paplay"):
        exe = shutil.which(player)
        if exe:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    path = f.name
                    f.write(audio_bytes)
                subprocess.run([exe, path], check=False)
                try:
                    os.unlink(path)
                except Exception:
                    pass
                return True
            except Exception:
                pass
    return False


def start_chat_window(serial_port: Optional[str], serial_baudrate: int) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception as exc:
        log_line("ERROR", f"[Chat] Tkinter not available: {exc}", always=True)
        return

    serial_ctrl: Optional[SerialController] = None
    if serial_port:
        try:
            serial_ctrl = SerialController(port=serial_port, baudrate=serial_baudrate)
            if not serial_ctrl.connect():
                serial_ctrl = None
        except Exception as exc:
            log_line("ERROR", f"[Chat] serial init failed: {exc}", always=True)
            serial_ctrl = None

    root = tk.Tk()
    root.title("LLM Chat")
    root.geometry("520x300")
    frame = tk.Frame(root, padx=8, pady=8)
    frame.pack(fill="both", expand=True)
    scroll = tk.Scrollbar(frame, orient="vertical")
    text_widget = tk.Text(frame, height=12, width=60, yscrollcommand=scroll.set)
    scroll.config(command=text_widget.yview)
    scroll.pack(side="right", fill="y")
    text_widget.pack(side="left", fill="both", expand=True)

    input_frame = tk.Frame(root)
    input_frame.pack(fill="x", padx=8, pady=(0, 8))
    input_text = tk.Text(input_frame, height=3, width=60)
    input_text.pack(fill="x", expand=True)

    def send_llm_text_now() -> None:
        txt = input_text.get("1.0", "end").strip()
        if not txt:
            return
        text_widget.insert("end", f"You: {txt}\n\n")
        text_widget.see("end")
        text_widget.update()
        input_text.delete("1.0", "end")
        threading.Thread(target=lambda: handle_incoming_text(txt, serial_ctrl, text_widget), daemon=True).start()

    def _on_enter_key(event: tk.Event) -> str:
        if event.keysym == "Return" and not (event.state & 0x0001):
            send_llm_text_now()
            return "break"
        return ""

    input_text.bind("<Return>", _on_enter_key)

    btn_frame = tk.Frame(root)
    btn_frame.pack(fill="x", padx=8, pady=(0, 8))
    tk.Button(btn_frame, text="Send to LLM", command=send_llm_text_now).pack(side="right", padx=4)
    tk.Button(btn_frame, text="Close", command=root.destroy).pack(side="right")
    try:
        root.mainloop()
    finally:
        if serial_ctrl:
            serial_ctrl.disconnect()


def handle_incoming_text(user_text: str, serial_ctrl: Optional[SerialController] = None, text_widget=None) -> None:
    system_prompt = (
        "You are an assistant that MUST respond with a single, valid JSON object and NOTHING else.\n"
        "The JSON object MUST have exactly two fields: \n"
        "  - \"animation\": a string, either \"speech\" or \"waving\"\n"
        "  - \"text\": a string containing the reply text\n"
        "Do NOT include any extra commentary, markdown, or explanation. Return exactly one JSON object.\n"
        "Example: {\"animation\": \"speech\", \"text\": \"Hello!\"}"
    )
    log_line("LLM", f"[LLM] querying for: {user_text}")
    llm_text = call_openrouter(system_prompt, user_text)
    if not llm_text:
        log_line("LLM", "[LLM] no response")
        if text_widget:
            text_widget.insert("end", "LLM: (no response)\n\n")
            text_widget.see("end")
        return
    parsed = parse_llm_response_as_json(llm_text)
    if not parsed:
        log_line("LLM", f"[LLM] unexpected response format:\n{llm_text}")
        if text_widget:
            text_widget.insert("end", f"LLM: (parsing error)\n\n")
            text_widget.see("end")
        return
    animation = parsed.get("animation", "speech")
    reply_text = parsed.get("text", "")
    log_line("LLM", f"[LLM] animation={animation} reply={reply_text}")
    log_line("ANIM", f"[ANIM] {animation}")
    if text_widget:
        text_widget.insert("end", f"LLM [{animation}]: {reply_text}\n\n")
        text_widget.see("end")
    if serial_ctrl:
        serial_ctrl.send_animation(animation, reply_text)
    wav = tts_synthesize_to_wav(reply_text)
    if wav:
        ok = play_audio_bytes(wav)
        if not ok:
            log_line("TTS", "[TTS] audio playback failed")
    else:
        log_line("TTS", f"[TTS] no audio produced; would reply: {reply_text}")


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
    selected_log_kinds: dict[str, Optional[List[str]]] = {"value": None}

    ports = detect_serial_ports()
    port_labels = ["None (tracking only)"] + [get_serial_port_label(p) for p in ports]
    port_values = [None] + ports

    root = tk.Tk()
    root.title("Select Webcam Settings")
    root.resizable(True, True)
    root.geometry("620x720")
    root.minsize(560, 640)

    # Scrollable main area: a canvas with a vertical scrollbar and an inner frame
    container = tk.Frame(root)
    container.pack(fill="both", expand=True)
    canvas = tk.Canvas(container)
    vscroll = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vscroll.set)
    vscroll.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    frame = tk.Frame(canvas, padx=12, pady=12)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    def _on_frame_configure(event: tk.Event) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    frame.bind("<Configure>", _on_frame_configure)

    def _on_mousewheel(event: tk.Event) -> None:
        delta = event.delta
        if delta == 0 and hasattr(event, "num"):
            if event.num == 4:
                delta = 120
            elif event.num == 5:
                delta = -120
        if delta:
            canvas.yview_scroll(int(-1 * (delta / 120)), "units")

    root.bind_all("<MouseWheel>", _on_mousewheel)
    root.bind_all("<Button-4>", _on_mousewheel)
    root.bind_all("<Button-5>", _on_mousewheel)

    tk.Label(frame, text="Face Tracking Setup", font=("DejaVu Sans", 11, "bold")).pack(anchor="w")
    tk.Label(frame, text="Configure camera and audio settings.").pack(anchor="w", pady=(4, 8))

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

    # --- Microphone selection / text input ---
    def _collect_input_devices() -> List[Tuple[str, str, Optional[int]]]:
        devices: List[Tuple[str, str, Optional[int]]] = []
        if shutil.which("arecord"):
            devices.append(("ALSA hw:1,0", "arecord", None))
            devices.append(("ALSA plughw:1,0", "arecord", None))
        return devices

    mic_device_entries = _collect_input_devices()
    mic_support = bool(mic_device_entries)

    mic_frame = tk.LabelFrame(frame, text="Microphone / LLM Test", padx=10, pady=10)
    mic_frame.pack(fill="x", pady=(0, 10))

    tk.Label(mic_frame, text="Input device:").grid(row=0, column=0, sticky="w")
    mic_options = ["None"] + [entry[0] for entry in mic_device_entries]
    mic_default = mic_options[1] if len(mic_options) > 1 else "None"
    mic_var = tk.StringVar(value=mic_default)
    mic_menu = tk.OptionMenu(mic_frame, mic_var, *mic_options)
    mic_menu.grid(row=0, column=1, sticky="w", padx=8)

    mic_hint = tk.Label(mic_frame, text="" if mic_support else "No input devices detected.", fg="#9b2c2c")
    mic_hint.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6,0))

    tk.Label(mic_frame, text="ALSA device string:").grid(row=2, column=0, sticky="w", pady=(6,0))
    mic_alsa_var = tk.StringVar(value="hw:1,0")
    tk.Entry(mic_frame, textvariable=mic_alsa_var, width=18).grid(row=2, column=1, sticky="w", padx=8, pady=(6,0))

    mic_cont_var = tk.BooleanVar(value=True)
    tk.Checkbutton(mic_frame, text="Record continuously (background)", variable=mic_cont_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8,0))

    def _sync_chat_visibility() -> None:
        pass

    def _refresh_mic_devices() -> None:
        nonlocal mic_device_entries
        mic_device_entries = _collect_input_devices()
        updated = ["None"] + [entry[0] for entry in mic_device_entries]
        menu = mic_menu["menu"]
        menu.delete(0, "end")
        for option in updated:
            menu.add_command(label=option, command=lambda value=option: mic_var.set(value))
        mic_var.set(updated[1] if len(updated) > 1 else "None")
        mic_hint.config(text="" if mic_device_entries else "No input devices detected.")
        _sync_chat_visibility()

    def _dump_alsa_devices() -> None:
        try:
            output = subprocess.check_output(["arecord", "-l"], text=True, stderr=subprocess.STDOUT)
            for line in output.splitlines():
                log_line("STAT", f"[ALSA] {line}")
        except Exception as exc:
            messagebox.showerror("ALSA error", str(exc))

    tk.Button(mic_frame, text="Rescan Devices", command=_refresh_mic_devices).grid(row=5, column=0, sticky="w", padx=8, pady=(8,0))
    tk.Button(mic_frame, text="List ALSA", command=_dump_alsa_devices).grid(row=5, column=1, sticky="w", padx=8, pady=(8,0))

    # Chat window is launched after Start if mic is None.

    

    log_frame = tk.LabelFrame(frame, text="Log Display", padx=10, pady=10)
    log_frame.pack(fill="x", pady=(0, 10))

    log_mode_names = [label for label, _ in LOG_DISPLAY_PRESETS]
    log_mode_var = tk.StringVar(value=log_mode_names[0])
    log_custom_var = tk.StringVar(value="POS,STAT")

    tk.Label(log_frame, text="Show in terminal:").grid(row=0, column=0, sticky="w")
    tk.OptionMenu(log_frame, log_mode_var, *log_mode_names).grid(row=0, column=1, sticky="w", padx=8)

    tk.Label(log_frame, text="Custom kinds:").grid(row=1, column=0, sticky="w", pady=(8, 0))
    tk.Entry(log_frame, textvariable=log_custom_var, width=28).grid(row=1, column=1, sticky="w", padx=8, pady=(8, 0))

    def sync_log_display() -> None:
        mode = log_mode_var.get()
        if mode == "All":
            selected_log_kinds["value"] = None
        elif mode == "Tracking":
            selected_log_kinds["value"] = ["POS", "STAT"]
        elif mode == "LLM/Audio":
            selected_log_kinds["value"] = ["TEXT", "ANIM", "AUDIO", "LLM", "STT", "TTS"]
        else:
            raw = log_custom_var.get().strip()
            if not raw:
                selected_log_kinds["value"] = None
            else:
                selected_log_kinds["value"] = [part.strip().upper() for part in raw.split(",") if part.strip()]

    log_mode_var.trace_add("write", lambda *_: sync_log_display())
    log_custom_var.trace_add("write", lambda *_: sync_log_display())
    sync_log_display()

    list_label = tk.Label(frame, text="Available Cameras:", font=("DejaVu Sans", 9, "bold"))
    list_label.pack(anchor="w", pady=(8, 4))

    list_frame = tk.Frame(frame)
    list_frame.pack(fill="both", expand=True)

    scrollbar = tk.Scrollbar(list_frame, orient="vertical")
    listbox = tk.Listbox(list_frame, height=6, exportselection=False, yscrollcommand=scrollbar.set)
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
        sync_log_display()
        root.destroy()

    def cancel() -> None:
        selected_index["value"] = None
        root.destroy()

    button_frame = tk.Frame(frame)
    button_frame.pack(side="bottom", fill="x", pady=(10, 0))
    tk.Button(button_frame, text="Start Tracking", width=12, command=open_selected).pack(side="right", padx=(8, 0))
    tk.Button(button_frame, text="Cancel", width=12, command=cancel).pack(side="right")

    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()

    idx = selected_index["value"]
    if idx is None or not (0 <= idx < len(cameras)):
        return None
    manual_alsa = mic_alsa_var.get().strip()
    if mic_var.get() == "None":
        mic_device_value = None
    elif manual_alsa:
        mic_device_value = f"alsa:{manual_alsa}"
    else:
        mic_device_value = mic_var.get()

    return CameraSelection(
        camera=cameras[idx],
        resolution=selected_resolution["value"],
        model_selection=selected_model["value"],
        serial_port=selected_port["value"],
        serial_baudrate=selected_baudrate["value"],
        log_display_kinds=selected_log_kinds["value"],
        mic_device=mic_device_value,
        mic_continuous=mic_cont_var.get(),
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
    log_display_kinds: Optional[List[str]] = None,
    serial_port: Optional[str] = None,
    serial_baudrate: int = 115200,
    mic_device: Optional[str] = None,
    mic_continuous: bool = False,
) -> None:
    set_log_display_kinds(log_display_kinds)
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
        try:
            serial_ctrl = SerialController(port=serial_port, baudrate=serial_baudrate)
            if not serial_ctrl.connect():
                serial_ctrl = None
        except Exception as exc:
            log_line("ERROR", f"[Serial] init failed: {exc}", always=True)
            serial_ctrl = None

    stop_event = threading.Event()
    mic_thread: Optional[threading.Thread] = None
    if mic_continuous and mic_device:
        backend, index = parse_mic_label(mic_device)
        if backend:
            def _mic_loop() -> None:
                log_line("STT", "[STT] continuous mic recording started")
                while not stop_event.is_set():
                    audio_bytes = record_audio_chunk(backend, index)
                    if not audio_bytes:
                        log_line("STT", "[STT] recording failed")
                        time.sleep(1.0)
                        continue
                    text = transcribe_audio_bytes(audio_bytes)
                    if text:
                        log_line("STT", f"[STT] {text}")
                        handle_incoming_text(text, serial_ctrl)
                    time.sleep(0.2)

            mic_thread = threading.Thread(target=_mic_loop, daemon=True)
            mic_thread.start()

    detector = None
    window_name = f"Face Tracking - Camera {camera_index}"
    try:
        detector = build_face_detector(model_selection, min_detection_confidence)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        prev = time.time()
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                log_line("ERROR", "Frame read failed, stopping.", always=True)
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
                log_line("POS", f"[Tracking] POS {err_x:+.4f},{err_y:+.4f} conf={score:.2f}")
                if serial_ctrl:
                    serial_ctrl.send_position_vector(err_x, err_y, score)
            else:
                cv2.putText(frame, "No face detected", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 120, 240), 2, cv2.LINE_AA)
                log_line("STAT", "[Tracking] No face detected")
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
        stop_event.set()
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
    parser.add_argument("--help-backend", action="store_true", help="Show backend features and exit")
    return parser.parse_args()


def print_backend_help() -> None:
    print("\n" + "=" * 60)
    print("FACE TRACKING BACKEND HELP")
    print("=" * 60)
    print("Core Features:")
    print("  1. Real-time face detection using MediaPipe")
    print("  2. Continuous microphone STT -> LLM -> TTS on this PC")
    print("  3. Send POS + ANIM over serial (no audio over serial)")
    print()
    print("LLM Features:")
    print("  - Transcribe microphone audio using Whisper")
    print("  - Send to OpenRouter LLM (requires API key in api_key_openrouterai)")
    print("  - Synthesize response to audio using espeak/pyttsx3")
    print("  - Play audio locally on this PC")
    print()
    print("Example usage:")
    print("  # Interactive camera selection:")
    print("  python3 face_tracking_test.py")
    print()
    print("  # CLI args (bypass GUI):")
    print("  python3 face_tracking_test.py \\")
    print("    --camera-index 0 \\")
    print("    --serial-port /dev/ttyUSB0 \\")
    print("    --serial-baudrate 115200")
    print("=" * 60 + "\n")


def main() -> None:
    args = parse_args()

    if args.help_backend:
        print_backend_help()
        return

    print_backend_help()

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
        log_display_kinds = selected.log_display_kinds
        mic_device = selected.mic_device
        mic_continuous = selected.mic_continuous
        serial_port = selected.serial_port
        serial_baudrate = selected.serial_baudrate
    else:
        cam_index = args.camera_index
        model_selection = args.model_selection
        target_resolution = None
        log_display_kinds = None
        mic_device = None
        mic_continuous = False
        serial_port = args.serial_port
        serial_baudrate = args.serial_baudrate

    if mic_device is None:
        proc = multiprocessing.Process(target=start_chat_window, args=(serial_port, serial_baudrate))
        proc.daemon = False
        proc.start()

    run_face_tracking(
        camera_index=cam_index,
        model_selection=model_selection,
        min_detection_confidence=args.min_detection_confidence,
        target_resolution=target_resolution,
        log_display_kinds=log_display_kinds,
        serial_port=serial_port,
        serial_baudrate=serial_baudrate,
        mic_device=mic_device,
        mic_continuous=mic_continuous,
    )


if __name__ == "__main__":
    main()
