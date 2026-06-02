#!/usr/bin/env python3
"""Real-time face tracking + LLM chat + optional wakeword pipeline + German TTS."""
from __future__ import annotations

import argparse
import multiprocessing
import os
import re
import site
import sys
import time
import threading
import json
import queue
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

try:
    import serial
except Exception:
    serial = None  # type: ignore

if os.name == "posix":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    if "QT_QPA_FONTDIR" not in os.environ:
        for fontdir in ["/usr/share/fonts/truetype/dejavu",
                        "/usr/share/fonts/truetype",
                        "/usr/share/fonts"]:
            if os.path.isdir(fontdir):
                os.environ["QT_QPA_FONTDIR"] = fontdir
                break


def _remove_user_site_from_path() -> None:
    try:
        usersite = Path(site.getusersitepackages()).resolve()
    except Exception:
        return
    filtered: List[str] = []
    for entry in sys.path:
        try:
            resolved = Path(entry).resolve()
        except Exception:
            filtered.append(entry)
            continue
        if resolved == usersite or str(resolved).startswith(str(usersite)):
            continue
        filtered.append(entry)
    sys.path = filtered


def _bootstrap_project_venv() -> None:
    if ".venv" in Path(sys.executable).parts:
        return
    venv_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)


_remove_user_site_from_path()
_bootstrap_project_venv()

import cv2
import requests
import numpy as np
import sounddevice as sd
import torch
import onnxruntime as ort
import openwakeword
from openwakeword.model import Model
from faster_whisper import WhisperModel

# ── TTS (Piper) ───────────────────────────────────────────────────────────────
try:
    from piper.voice import PiperVoice
    _PIPER_AVAILABLE = True
except ImportError:
    _PIPER_AVAILABLE = False
    PiperVoice = None  # type: ignore

TTS_MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_models")
TTS_MODEL_FILE = os.path.join(TTS_MODEL_DIR, "de_DE-thorsten-high.onnx")
TTS_CFG_FILE   = os.path.join(TTS_MODEL_DIR, "de_DE-thorsten-high.onnx.json")
TTS_HF_BASE    = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/high"

_tts_voice = None  # loaded lazily once


def _ensure_tts_model() -> None:
    os.makedirs(TTS_MODEL_DIR, exist_ok=True)
    if not os.path.isfile(TTS_MODEL_FILE):
        print("[tts] Downloading Thorsten-Voice HIGH (~65 MB)...")
        urllib.request.urlretrieve(
            f"{TTS_HF_BASE}/de_DE-thorsten-high.onnx", TTS_MODEL_FILE
        )
    if not os.path.isfile(TTS_CFG_FILE):
        print("[tts] Downloading model config...")
        urllib.request.urlretrieve(
            f"{TTS_HF_BASE}/de_DE-thorsten-high.onnx.json", TTS_CFG_FILE
        )


def _get_tts_voice():
    global _tts_voice
    if _tts_voice is not None:
        return _tts_voice
    if not _PIPER_AVAILABLE:
        return None
    _ensure_tts_model()
    _tts_voice = PiperVoice.load(TTS_MODEL_FILE, config_path=TTS_CFG_FILE)
    return _tts_voice


def _chunk_to_pcm(chunk) -> bytes:
    if isinstance(chunk, (bytes, bytearray)):
        return bytes(chunk)
    if hasattr(chunk, "audio_int16_bytes"):
        return bytes(chunk.audio_int16_bytes)
    if hasattr(chunk, "audio_int16_array"):
        return chunk.audio_int16_array.tobytes()
    if hasattr(chunk, "audio_float_array"):
        return (chunk.audio_float_array * 32767).astype(np.int16).tobytes()
    if hasattr(chunk, "audio"):
        d = chunk.audio
        if isinstance(d, (bytes, bytearray)):
            return bytes(d)
        return np.asarray(d, dtype=np.int16).tobytes()
    raise TypeError(f"Cannot extract PCM from {type(chunk)}")


def speak(text: str) -> None:
    """Speak text aloud using Piper TTS (German Thorsten-Voice)."""
    if not _PIPER_AVAILABLE:
        print("[tts] piper-tts not installed — skipping TTS.")
        return
    text = text.strip()
    if not text:
        return
    voice = _get_tts_voice()
    if voice is None:
        return
    try:
        from piper.voice import SynthesisConfig
        synth_cfg = SynthesisConfig()
    except ImportError:
        synth_cfg = None
    pcm_parts = []
    gen = voice.synthesize(text, synth_cfg) if synth_cfg is not None else voice.synthesize(text)
    for chunk in gen:
        raw = _chunk_to_pcm(chunk)
        if raw:
            pcm_parts.append(np.frombuffer(raw, dtype=np.int16))
    if not pcm_parts:
        return
    audio = np.concatenate(pcm_parts)
    sd.play(audio, samplerate=voice.config.sample_rate)
    sd.wait()


# ── Audio pipeline constants ──────────────────────────────────────────────────
SAMPLE_RATE      = 16000
OWW_CHUNK        = 1280
VAD_CHUNK        = 512
VAD_THRESHOLD    = 0.5
WW_THRESHOLD     = 0.5
SILENCE_SECS     = 1.5
WW_COOLDOWN_SECS = 1.2
MAX_RECORD_SECS  = 15.0

STATE_WAITING  = "WAITING_FOR_WAKEWORD"
STATE_COOLDOWN = "WAKEWORD_COOLDOWN"
STATE_SAMPLING = "SAMPLING"
STATE_BUSY     = "STT_BUSY"


def ensure_oww_models() -> list:
    try:
        openwakeword.utils.download_models()
    except Exception as e:
        print(f"[oww] Built-in download failed: {e}, trying manual fallback...")
    paths = [p for p in openwakeword.get_pretrained_model_paths() if "alexa" in p.lower()]
    if paths:
        return paths
    oww_pkg_dir = os.path.dirname(openwakeword.__file__)
    models_dir  = os.path.join(oww_pkg_dir, "resources", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_url  = "https://github.com/dscripka/openWakeWord/releases/download/v0.1.1/alexa_v0.1.onnx"
    model_path = os.path.join(models_dir, "alexa_v0.1.onnx")
    if not os.path.isfile(model_path):
        print("[oww] Downloading alexa model...")
        urllib.request.urlretrieve(model_url, model_path)
    paths = [p for p in openwakeword.get_pretrained_model_paths() if "alexa" in p.lower()]
    return paths if paths else [model_path]


# ── Dataclasses ───────────────────────────────────────────────────────────────
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
    serial_port: Optional[str]           = None
    serial_baudrate: int                  = 115200
    log_display_kinds: Optional[List[str]] = None
    use_wakeword: bool                    = False
    tts_enabled: bool                     = True


RESOLUTION_PRESETS: List[Tuple[str, Tuple[int, int]]] = [
    ("640 x 480",   (640,  480)),
    ("1280 x 720",  (1280, 720)),
    ("1920 x 1080", (1920, 1080)),
    ("320 x 240",   (320,  240)),
]

LOG_DISPLAY_PRESETS: List[Tuple[str, Optional[List[str]]]] = [
    ("All",      None),
    ("Tracking", ["POS", "STAT"]),
    ("LLM",      ["ANIM", "LLM"]),
    ("Custom",   []),
]

ACTIVE_LOG_KINDS: Optional[set] = None


class Ansi:
    RESET   = "\033[0m"
    DIM     = "\033[2m"
    RED     = "\033[1;31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"


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
        elif tag in ("STAT", "TEXT"):
            print(colorize(message, Ansi.CYAN), flush=True)
        elif tag in ("LLM", "ANIM"):
            print(colorize(message, Ansi.MAGENTA), flush=True)
        elif tag == "AUDIO":
            print(colorize(message, Ansi.GREEN), flush=True)
        else:
            print(colorize(message, Ansi.DIM), flush=True)


def detect_serial_ports() -> List[str]:
    ports: List[str] = []
    for pattern in ["ttyUSB*", "ttyACM*", "ttyS*"]:
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
            raise RuntimeError("pyserial is not installed.")
        self.port      = port
        self.baudrate  = baudrate
        self.timeout   = timeout
        self.ser: Any  = None
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
        if self.ser and getattr(self.ser, "isOpen", lambda: False)():
            try:
                self.ser.close()
            finally:
                self.connected = False
                log_line("STAT", "[Serial] Disconnected")

    def send_position_vector(self, x_error: float, y_error: float, confidence: float) -> None:
        if not self.connected or not self.ser:
            return
        try:
            msg = f"POS,{x_error:.4f},{y_error:.4f},{confidence:.2f}"
            self.ser.write(msg.encode("utf-8"))
        except Exception as exc:
            log_line("ERROR", f"[Serial] POS send failed: {exc}", always=True)
            self.connected = False

    def send_animation(self, animation: str, text: Optional[str] = None) -> None:
        if not self.connected or not self.ser:
            return
        try:
            payload = text or ""
            msg = f"ANIM,{animation},{payload}"
            self.ser.write(msg.encode("utf-8"))
        except Exception as exc:
            log_line("ERROR", f"[Serial] ANIM send failed: {exc}", always=True)
            self.connected = False


# ── LLM helpers ───────────────────────────────────────────────────────────────
def _read_api_key(path: str = "api_key_openrouterai") -> Optional[str]:
    try:
        p = Path(__file__).resolve().parent / path
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def call_openrouter(system_prompt: str, user_prompt: str) -> Optional[str]:
    api_key = _read_api_key()
    if not api_key:
        log_line("LLM", "[LLM] No API key found in apikey_openrouter.ai")
        return None
    url     = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.3,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30.0)
        r.raise_for_status()
        data    = r.json()
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
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if not m:
            return None
        data = json.loads(m.group(0))
        if not isinstance(data, dict):
            return None
        anim = data.get("animation")
        txt  = data.get("text")
        if anim not in ("speech", "waving"):
            return None
        if not isinstance(txt, str):
            return None
        return {"animation": anim, "text": txt}
    except Exception:
        return None


# ── LLM system prompt ─────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    'You are an assistant that MUST respond with a single, valid JSON object and NOTHING else.\n'
    'The JSON must have exactly two fields:\n'
    '  - "animation": either "speech" or "waving"\n'
    '  - "text": your reply as a string\n'
    'Return ONLY the JSON object. No markdown, no explanation.'
)


def handle_incoming_text(
    user_text: str,
    serial_ctrl: Optional[SerialController] = None,
    text_widget=None,
    tts_enabled: bool = False,
) -> None:
    log_line("LLM", f"[LLM] querying for: {user_text}")
    llm_text = call_openrouter(_SYSTEM_PROMPT, user_text)
    if not llm_text:
        log_line("LLM", "[LLM] no response")
        if text_widget:
            text_widget.after(0, lambda: text_widget.insert("end", "[LLM] no response\n"))
        return

    parsed = parse_llm_response_as_json(llm_text)
    if not parsed:
        log_line("LLM", f"[LLM] unexpected format: {llm_text}")
        if text_widget:
            msg = f"[LLM raw] {llm_text}\n"
            text_widget.after(0, lambda m=msg: text_widget.insert("end", m))
        return

    animation  = parsed.get("animation", "speech")
    reply_text = parsed.get("text", "")
    log_line("LLM",  f"[LLM] animation={animation} reply={reply_text}")
    log_line("ANIM", f"[ANIM] {animation}")

    if text_widget:
        msg = f"[LLM | {animation}] {reply_text}\n"
        text_widget.after(0, lambda m=msg: (text_widget.insert("end", m), text_widget.see("end")))

    if serial_ctrl:
        serial_ctrl.send_animation(animation, reply_text)

    if tts_enabled:
        try:
            speak(reply_text)
        except Exception as e:
            log_line("ERROR", f"[TTS] failed: {e}", always=True)


# ── Lightweight GUI adapter for AudioPipeline when no AppGUI is present ───────
class _PipelineGUIAdapter:
    def __init__(self, on_transcription=None):
        self._on_transcription = on_transcription

    def safe_update(self, fn, *args):
        fn(*args)

    def log(self, msg: str):
        log_line("AUDIO", f"[Pipeline] {msg}")

    def set_status(self, text: str):
        pass

    def show_result(self, text: str):
        if self._on_transcription:
            self._on_transcription(text)

    def update_mic_level(self, level: float):
        pass

    def update_speech_indicator(self, active: bool):
        pass

    def update_wakeword_indicator(self, active: bool):
        pass

    def update_ww_score(self, score: float):
        pass


# ── AudioPipeline ─────────────────────────────────────────────────────────────
class AudioPipeline(threading.Thread):
    def __init__(self, gui, use_wakeword: bool):
        super().__init__(daemon=True)
        self.gui          = gui
        self.use_wakeword = use_wakeword
        self.running      = True
        self.state        = STATE_WAITING if use_wakeword else STATE_SAMPLING
        self._vad_lock    = threading.Lock()

        providers = ["ROCMExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        selected  = next((p for p in providers if p in available), "CPUExecutionProvider")
        self.gui.log(f"[ONNX] Provider: {selected}")

        self.gui.log("Loading WakeWord model (alexa)...")
        alexa_paths         = ensure_oww_models()
        self.oww_model      = Model(wakeword_model_paths=alexa_paths)
        self.oww_model_path = alexa_paths

        self.gui.log("Loading Silero VAD...")
        self.vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,
        )

        self.gui.log("Loading Faster-Whisper large-v3-turbo...")
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type   = "float16" if self.device == "cuda" else "int8"
        self.beam_size = 5 if self.device == "cuda" else 1
        self.stt_model = WhisperModel(
            "large-v3-turbo", device=self.device, compute_type=compute_type
        )
        self.gui.log("Pipeline ready. Say \'Alexa\' then your command!")

    def _vad_on_chunk(self, audio_int16: np.ndarray):
        max_prob = 0.0
        with self._vad_lock:
            for start in range(0, len(audio_int16) - VAD_CHUNK + 1, VAD_CHUNK):
                sub  = audio_int16[start:start + VAD_CHUNK].astype(np.float32) / 32768.0
                prob = self.vad_model(torch.from_numpy(sub), SAMPLE_RATE).item()
                if prob > max_prob:
                    max_prob = prob
        return max_prob > VAD_THRESHOLD, max_prob

    def run(self):
        silence_threshold = int(SILENCE_SECS * SAMPLE_RATE / OWW_CHUNK)
        max_rec_chunks    = int(MAX_RECORD_SECS * SAMPLE_RATE / OWW_CHUNK)
        audio_buffer: list = []
        silence_counter    = 0
        cooldown_until     = 0.0

        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=OWW_CHUNK
        ) as stream:
            while self.running:
                chunk, overflow = stream.read(OWW_CHUNK)
                if overflow:
                    continue
                audio_data = chunk.flatten().astype(np.int16)
                self.gui.safe_update(self.gui.update_mic_level, float(np.abs(audio_data).mean()))
                is_speech, _ = self._vad_on_chunk(audio_data)
                self.gui.safe_update(self.gui.update_speech_indicator, is_speech)
                now = time.monotonic()

                if self.state == STATE_WAITING:
                    prediction = self.oww_model.predict(audio_data)
                    score      = max(prediction.values()) if prediction else 0.0
                    self.gui.safe_update(self.gui.update_ww_score, score)
                    if score > WW_THRESHOLD and now > cooldown_until:
                        self.gui.log(f"[WW] Wake word detected! score={score:.3f}")
                        self.gui.safe_update(self.gui.update_wakeword_indicator, True)
                        cooldown_until  = now + WW_COOLDOWN_SECS
                        self.state      = STATE_COOLDOWN
                        audio_buffer    = []
                        silence_counter = 0

                elif self.state == STATE_COOLDOWN:
                    self.oww_model.predict(audio_data)
                    self.gui.safe_update(self.gui.update_ww_score, 0.0)
                    if now > cooldown_until:
                        self.oww_model = Model(wakeword_model_paths=self.oww_model_path)
                        self.gui.log("Listening for your command...")
                        self.gui.safe_update(self.gui.update_wakeword_indicator, False)
                        self.gui.safe_update(self.gui.set_status, "Recording...")
                        self.state      = STATE_SAMPLING
                        audio_buffer    = []
                        silence_counter = 0

                elif self.state == STATE_SAMPLING:
                    audio_buffer.append(audio_data)
                    silence_counter = 0 if is_speech else silence_counter + 1
                    hit_silence = silence_counter >= silence_threshold
                    hit_max     = len(audio_buffer) >= max_rec_chunks
                    if hit_silence or hit_max:
                        trim          = silence_counter if hit_silence else 0
                        useful_frames = len(audio_buffer) - trim
                        if useful_frames >= 2:
                            audio_trimmed = (
                                np.concatenate(audio_buffer[:useful_frames])
                                .astype(np.float32) / 32768.0
                            )
                            self.state = STATE_BUSY
                            threading.Thread(
                                target=self._transcribe_and_reset,
                                args=(audio_trimmed.copy(),),
                                daemon=True,
                            ).start()
                        else:
                            self.gui.log("Nothing useful recorded, back to waiting.")
                            self.gui.safe_update(self.gui.set_status, "")
                            self.state      = STATE_WAITING if self.use_wakeword else STATE_SAMPLING
                            audio_buffer    = []
                            silence_counter = 0

                elif self.state == STATE_BUSY:
                    pass  # drain mic while STT is running

    def _transcribe_and_reset(self, audio_data: np.ndarray):
        try:
            segments, _ = self.stt_model.transcribe(
                audio_data,
                language="de",
                beam_size=self.beam_size,
                condition_on_previous_text=False,
            )
            text = " ".join(seg.text for seg in segments).strip()
            if text:
                self.gui.log(f"[STT] {text}")
                self.gui.safe_update(self.gui.show_result, text)
            else:
                self.gui.log("[STT] nichts erkannt")
        except Exception as e:
            self.gui.log(f"[STT] error: {e}")
        finally:
            self.gui.safe_update(self.gui.set_status, "")
            self.oww_model = Model(wakeword_model_paths=self.oww_model_path)
            self.state = STATE_WAITING if self.use_wakeword else STATE_SAMPLING
            if self.use_wakeword:
                self.gui.log("Ready — say \'Alexa\' again.")

    def stop(self):
        self.running = False


# ── Wakeword mode GUI window ──────────────────────────────────────────────────
def _start_wakeword_gui(
    serial_port: Optional[str], serial_baudrate: int, tts_enabled: bool
) -> None:
    import tkinter as tk
    from tkinter import ttk

    serial_ctrl: Optional[SerialController] = None
    if serial_port:
        try:
            serial_ctrl = SerialController(port=serial_port, baudrate=serial_baudrate)
            if not serial_ctrl.connect():
                serial_ctrl = None
        except Exception as e:
            log_line("ERROR", f"[WW GUI] Serial error: {e}", always=True)

    root = tk.Tk()
    root.title("Voice Pipeline")
    root.resizable(False, False)

    tk.Label(root, text="Voice Pipeline", font=("Helvetica", 16, "bold")).pack(pady=10)

    use_wakeword_var = tk.BooleanVar(value=True)
    tk.Checkbutton(root, text="Enable Wake Word (Alexa)", variable=use_wakeword_var).pack()


    start_btn = tk.Button(
        root, text="Start Pipeline", bg="#2e7d32", fg="white",
        font=("Helvetica", 11, "bold"), padx=12, pady=6,
    )
    start_btn.pack(pady=10)

    status_lbl = tk.Label(root, text="", font=("Helvetica", 11), fg="#1565c0")
    status_lbl.pack()

    result_frame = tk.LabelFrame(root, text="Last Transcription", padx=8, pady=4)
    result_frame.pack(fill="x", padx=10, pady=4)
    result_var = tk.StringVar(value="")
    tk.Label(
        result_frame, textvariable=result_var,
        font=("Helvetica", 12, "bold"), wraplength=360, fg="#000", justify="left",
    ).pack(anchor="w")

    logs = tk.Text(root, height=10, width=60, state="disabled", bg="#1e1e1e", fg="#d4d4d4")
    logs.pack(padx=10, pady=(4, 6))

    meter_frame = tk.Frame(root)
    meter_frame.pack(pady=6)
    tk.Label(meter_frame, text="Mic Level").grid(row=0, column=0, sticky="w", padx=6)
    mic_bar = ttk.Progressbar(meter_frame, length=220, mode="determinate")
    mic_bar.grid(row=0, column=1, padx=6)
    tk.Label(meter_frame, text="Speech").grid(row=1, column=0, sticky="w", padx=6, pady=4)
    speech_ind = tk.Label(meter_frame, text=" OFF ", bg="grey", fg="white", width=6)
    speech_ind.grid(row=1, column=1, sticky="w")
    tk.Label(meter_frame, text="Wake Word").grid(row=2, column=0, sticky="w", padx=6, pady=4)
    ww_ind = tk.Label(meter_frame, text=" NO ", bg="grey", fg="white", width=6)
    ww_ind.grid(row=2, column=1, sticky="w")
    tk.Label(meter_frame, text="WW Score").grid(row=3, column=0, sticky="w", padx=6, pady=4)
    ww_score_bar = ttk.Progressbar(meter_frame, length=220, mode="determinate")
    ww_score_bar.grid(row=3, column=1, padx=6)
    ww_score_lbl = tk.Label(meter_frame, text="0.000", width=6)
    ww_score_lbl.grid(row=3, column=2)

    pipeline_holder: List[Optional[AudioPipeline]] = [None]

    class _Adapter:
        def safe_update(self, fn, *args):
            root.after(0, fn, *args)

        def log(self, msg: str):
            def _write():
                logs.config(state="normal")
                logs.insert(tk.END, msg + "\n")
                logs.see(tk.END)
                logs.config(state="disabled")
            root.after(0, _write)
            print(f"[LOG] {msg}")

        def set_status(self, text: str):
            status_lbl.config(text=text)

        def show_result(self, text: str):
            result_var.set(text)
            threading.Thread(
                target=handle_incoming_text,
                args=(text, serial_ctrl, None, tts_enabled),
                daemon=True,
            ).start()

        def update_mic_level(self, level: float):
            mic_bar["value"] = min(100.0, level / 50.0 * 100)

        def update_speech_indicator(self, active: bool):
            speech_ind.config(
                text=" ON " if active else " OFF ",
                bg="#2e7d32" if active else "grey",
            )

        def update_wakeword_indicator(self, active: bool):
            ww_ind.config(
                text=" YES " if active else " NO ",
                bg="#e65100" if active else "grey",
            )

        def update_ww_score(self, score: float):
            ww_score_bar["value"] = min(100.0, score * 100)
            ww_score_lbl.config(text=f"{score:.3f}")

    adapter = _Adapter()

    def start_pipeline():
        if pipeline_holder[0] is not None:
            return
        start_btn.config(state="disabled", text="Loading")

        def _start():
            try:
                pipeline_holder[0] = AudioPipeline(adapter, use_wakeword_var.get())
                pipeline_holder[0].start()
                root.after(0, lambda: start_btn.config(text="Running", bg="#1565c0"))
            except Exception as e:
                adapter.log(f"ERROR: {e}")
                root.after(
                    0,
                    lambda: start_btn.config(
                        state="normal", text="Start Pipeline", bg="#2e7d32"
                    ),
                )

        threading.Thread(target=_start, daemon=True).start()

    start_btn.config(command=start_pipeline)

    def on_close():
        if pipeline_holder[0]:
            pipeline_holder[0].stop()
            pipeline_holder[0].join(timeout=2)
        if serial_ctrl:
            serial_ctrl.disconnect()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


# ── Chat window (text input + optional mic toggle) ────────────────────────────
def start_chat_window(
    serial_port: Optional[str], serial_baudrate: int, tts_enabled: bool = True
) -> None:
    try:
        import tkinter as tk
    except Exception as exc:
        log_line("ERROR", f"[Chat] Tkinter not available: {exc}", always=True)
        return

    serial_ctrl: Optional[SerialController] = None
    if serial_port:
        try:
            serial_ctrl = SerialController(port=serial_port, baudrate=serial_baudrate)
            if not serial_ctrl.connect():
                serial_ctrl = None
        except Exception as e:
            log_line("ERROR", f"[Chat] Serial error: {e}", always=True)

    root = tk.Tk()
    root.title("LLM Chat & Audio Controls")

    frame = tk.Frame(root, padx=8, pady=8)
    frame.pack(fill="both", expand=True)
    scroll = tk.Scrollbar(frame, orient="vertical")
    scroll.pack(side="right", fill="y")
    text_widget = tk.Text(frame, height=14, width=64, yscrollcommand=scroll.set)
    text_widget.pack(side="left", fill="both", expand=True)
    scroll.config(command=text_widget.yview)

    ctrl_frame = tk.Frame(root, padx=8, pady=4)
    ctrl_frame.pack(fill="x")
    tts_var = tk.BooleanVar(value=tts_enabled)
    mic_var = tk.BooleanVar(value=False)

    pipeline_holder: List[Optional[AudioPipeline]] = [None]

    def on_mic_toggle():
        if mic_var.get():
            input_text.config(state="disabled")

            def on_transcription(text: str):
                text_widget.insert("end", f"[Mic] {text}\n")
                text_widget.see("end")
                threading.Thread(
                    target=handle_incoming_text,
                    args=(text, serial_ctrl, text_widget, tts_var.get()),
                    daemon=True,
                ).start()

            adapter = _PipelineGUIAdapter(on_transcription=on_transcription)

            def _load():
                try:
                    pipeline_holder[0] = AudioPipeline(adapter, use_wakeword=True)
                    pipeline_holder[0].start()
                except Exception as e:
                    log_line("ERROR", f"[Chat Mic] {e}", always=True)

            threading.Thread(target=_load, daemon=True).start()
        else:
            input_text.config(state="normal")
            if pipeline_holder[0] is not None:
                pipeline_holder[0].stop()
                pipeline_holder[0] = None

    tk.Checkbutton(
        ctrl_frame, text="Enable Mic (Wakeword)", variable=mic_var, command=on_mic_toggle
    ).pack(side="left")
    tk.Checkbutton(ctrl_frame, text="Enable TTS Output", variable=tts_var).pack(side="left")

    in_frame = tk.Frame(root, padx=8, pady=4)
    in_frame.pack(fill="x")
    input_text = tk.Text(in_frame, height=3, width=64)
    input_text.pack(side="left", fill="x", expand=True)

    def send_now() -> None:
        user_text = input_text.get("1.0", "end").strip()
        if not user_text:
            return
        input_text.delete("1.0", "end")
        text_widget.insert("end", f"[You] {user_text}\n")
        text_widget.see("end")
        threading.Thread(
            target=handle_incoming_text,
            args=(user_text, serial_ctrl, text_widget, tts_var.get()),
            daemon=True,
        ).start()

    def _on_enter(event) -> str:
        if not (event.state & 0x0001):
            send_now()
            return "break"
        return ""

    input_text.bind("<Return>", _on_enter)

    btn_frame = tk.Frame(root, padx=8, pady=8)
    btn_frame.pack(fill="x")
    tk.Button(btn_frame, text="Send to LLM", command=send_now).pack(side="right", padx=4)
    tk.Button(btn_frame, text="Close", command=root.destroy).pack(side="right")

    input_text.focus_set()
    root.mainloop()

    if pipeline_holder[0]:
        pipeline_holder[0].stop()
    if serial_ctrl:
        serial_ctrl.disconnect()


# ── Camera discovery & selection window ──────────────────────────────────────
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
    candidate_indices = (
        list_linux_video_indices(max_cameras) if os.name == "posix" else []
    )
    if not candidate_indices:
        candidate_indices = list(range(max_cameras))
    for idx in candidate_indices:
        cap = (
            cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if os.name == "posix"
            else cv2.VideoCapture(idx)
        )
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
        for i, cam in enumerate(cameras, start=1):
            print(f"[{i}] Camera {cam.index} ({cam.width}x{cam.height})")
        choice = input("Select camera number (empty to cancel): ").strip()
        if not choice.isdigit():
            return None
        idx = int(choice) - 1
        if not 0 <= idx < len(cameras):
            return None
        resolution  = RESOLUTION_PRESETS[1][1]
        serial_port = input("Serial port (empty for none): ").strip() or None
        baud_text   = input("Baudrate [115200]: ").strip()
        baudrate    = int(baud_text) if baud_text.isdigit() else 115200
        return CameraSelection(
            camera=cameras[idx], resolution=resolution,
            model_selection=0, serial_port=serial_port, serial_baudrate=baudrate,
        )

    selected_index        = {"value": None}
    selected_resolution   = {"value": RESOLUTION_PRESETS[1][1]}
    selected_model        = {"value": 0}
    selected_port         = {"value": None}
    selected_baudrate     = {"value": 115200}
    selected_log_kinds    = {"value": None}
    selected_use_wakeword = {"value": False}
    selected_tts_enabled  = {"value": True}

    ports       = detect_serial_ports()
    port_labels = ["None (tracking only)"] + [get_serial_port_label(p) for p in ports]
    port_values = [None] + ports

    root = tk.Tk()
    root.title("Select Webcam Settings")
    root.resizable(True, True)
    root.geometry("620x820")
    root.minsize(560, 700)

    container = tk.Frame(root)
    container.pack(fill="both", expand=True)
    canvas  = tk.Canvas(container)
    vscroll = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vscroll.set)
    vscroll.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    frame = tk.Frame(canvas, padx=12, pady=12)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    def _on_frame_configure(event) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_mousewheel(event) -> None:
        delta = event.delta
        if delta == 0 and hasattr(event, "num"):
            delta = 120 if event.num == 4 else (-120 if event.num == 5 else 0)
        if delta:
            canvas.yview_scroll(int(-1 * delta / 120), "units")

    frame.bind("<Configure>", _on_frame_configure)
    root.bind_all("<MouseWheel>", _on_mousewheel)
    root.bind_all("<Button-4>",   _on_mousewheel)
    root.bind_all("<Button-5>",   _on_mousewheel)

    tk.Label(frame, text="Face Tracking Setup", font=("DejaVu Sans", 11, "bold")).pack(anchor="w")
    tk.Label(frame, text="Configure camera and serial settings.").pack(anchor="w", pady=(4, 8))

    # Camera section
    cam_frame = tk.LabelFrame(frame, text="Camera", padx=10, pady=10)
    cam_frame.pack(fill="x", pady=(0, 10))
    resolution_names = [label for label, _ in RESOLUTION_PRESETS]
    resolution_var   = tk.StringVar(value=resolution_names[1])
    model_var        = tk.StringVar(value="Short-range")
    tk.Label(cam_frame, text="Resolution:").grid(row=0, column=0, sticky="w")
    tk.OptionMenu(cam_frame, resolution_var, *resolution_names).grid(row=0, column=1, sticky="w", padx=8)
    tk.Label(cam_frame, text="Distance mode:").grid(row=1, column=0, sticky="w", pady=(8, 0))

    def toggle_model() -> None:
        selected_model["value"] = 1 - selected_model["value"]
        model_var.set("Full-range" if selected_model["value"] else "Short-range")

    tk.Button(cam_frame, textvariable=model_var, width=16, command=toggle_model).grid(
        row=1, column=1, sticky="w", padx=8, pady=(8, 0)
    )

    # Serial section
    serial_frame = tk.LabelFrame(frame, text="Serial", padx=10, pady=10)
    serial_frame.pack(fill="x", pady=(0, 10))
    serial_var      = tk.StringVar(value=port_labels[0])
    baudrate_var    = tk.StringVar(value="115200")
    manual_port_var = tk.StringVar(value="")
    tk.Label(serial_frame, text="Port:").grid(row=0, column=0, sticky="w")
    tk.OptionMenu(serial_frame, serial_var, *port_labels).grid(row=0, column=1, sticky="w", padx=8)
    tk.Label(serial_frame, text="Or manual tty:").grid(row=1, column=0, sticky="w", pady=(8, 0))
    tk.Entry(serial_frame, textvariable=manual_port_var, width=24).grid(
        row=1, column=1, sticky="w", padx=8, pady=(8, 0)
    )
    tk.Label(serial_frame, text="Baudrate:").grid(row=2, column=0, sticky="w", pady=(8, 0))
    tk.Entry(serial_frame, textvariable=baudrate_var, width=10).grid(
        row=2, column=1, sticky="w", padx=8, pady=(8, 0)
    )

    # Log display section
    log_frame = tk.LabelFrame(frame, text="Log Display", padx=10, pady=10)
    log_frame.pack(fill="x", pady=(0, 10))
    log_mode_names = [label for label, _ in LOG_DISPLAY_PRESETS]
    log_mode_var   = tk.StringVar(value=log_mode_names[0])
    log_custom_var = tk.StringVar(value="POS,STAT")
    tk.Label(log_frame, text="Show in terminal:").grid(row=0, column=0, sticky="w")
    tk.OptionMenu(log_frame, log_mode_var, *log_mode_names).grid(row=0, column=1, sticky="w", padx=8)
    tk.Label(log_frame, text="Custom kinds:").grid(row=1, column=0, sticky="w", pady=(8, 0))
    tk.Entry(log_frame, textvariable=log_custom_var, width=28).grid(
        row=1, column=1, sticky="w", padx=8, pady=(8, 0)
    )

    # Audio & Chat section
    audio_frame = tk.LabelFrame(frame, text="Audio & Chat", padx=10, pady=10)
    audio_frame.pack(fill="x", pady=8)
    use_wakeword_var = tk.BooleanVar(value=False)
    tts_enabled_var  = tk.BooleanVar(value=True)
    tk.Checkbutton(
        audio_frame,
        text="Use Wakeword & Mic  (opens Voice Pipeline window instead of Chat)",
        variable=use_wakeword_var,
    ).pack(anchor="w")
    tk.Checkbutton(
        audio_frame,
        text="Enable TTS  (read LLM answers aloud via Piper)",
        variable=tts_enabled_var,
    ).pack(anchor="w")

    # Camera list
    tk.Label(frame, text="Available Cameras:", font=("DejaVu Sans", 9, "bold")).pack(
        anchor="w", pady=(8, 4)
    )
    list_frame = tk.Frame(frame)
    list_frame.pack(fill="both", expand=True)
    scrollbar = tk.Scrollbar(list_frame, orient="vertical")
    listbox   = tk.Listbox(
        list_frame, height=6, exportselection=False, yscrollcommand=scrollbar.set
    )
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side="right", fill="y")
    listbox.pack(side="left", fill="both", expand=True)
    for cam in cameras:
        listbox.insert("end", f"Camera {cam.index}  ({cam.width}x{cam.height})")
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

    def sync_log_display() -> None:
        mode = log_mode_var.get()
        if mode == "All":
            selected_log_kinds["value"] = None
        elif mode == "Tracking":
            selected_log_kinds["value"] = ["POS", "STAT"]
        elif mode == "LLM":
            selected_log_kinds["value"] = ["ANIM", "LLM"]
        else:
            raw = log_custom_var.get().strip()
            selected_log_kinds["value"] = (
                None
                if not raw
                else [p.strip().upper() for p in raw.split(",") if p.strip()]
            )

    resolution_var.trace_add("write",  lambda *_: sync_resolution())
    serial_var.trace_add("write",      lambda *_: sync_serial())
    manual_port_var.trace_add("write", lambda *_: sync_serial())
    baudrate_var.trace_add("write",    lambda *_: sync_baudrate())
    log_mode_var.trace_add("write",    lambda *_: sync_log_display())
    log_custom_var.trace_add("write",  lambda *_: sync_log_display())
    sync_resolution()
    sync_serial()
    sync_baudrate()
    sync_log_display()

    def open_selected() -> None:
        selection = listbox.curselection()
        if not selection:
            messagebox.showinfo("No selection", "Please select a camera first.")
            return
        selected_index["value"]        = selection[0]
        selected_use_wakeword["value"] = use_wakeword_var.get()
        selected_tts_enabled["value"]  = tts_enabled_var.get()
        sync_log_display()
        root.destroy()

    def cancel() -> None:
        selected_index["value"] = None
        root.destroy()

    btn_frame = tk.Frame(frame)
    btn_frame.pack(side="bottom", fill="x", pady=(10, 0))
    tk.Button(btn_frame, text="Start Tracking", width=12, command=open_selected).pack(
        side="left", padx=(0, 8)
    )
    tk.Button(btn_frame, text="Cancel", width=12, command=cancel).pack(side="left")

    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()

    idx = selected_index["value"]
    if idx is None or not 0 <= idx < len(cameras):
        return None
    return CameraSelection(
        camera            = cameras[idx],
        resolution        = selected_resolution["value"],
        model_selection   = selected_model["value"],
        serial_port       = selected_port["value"],
        serial_baudrate   = selected_baudrate["value"],
        log_display_kinds = selected_log_kinds["value"],
        use_wakeword      = selected_use_wakeword["value"],
        tts_enabled       = selected_tts_enabled["value"],
    )


# ── OpenCV face detection & tracking loop ────────────────────────────────────
def open_camera(camera_index: int) -> cv2.VideoCapture:
    return (
        cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if os.name == "posix"
        else cv2.VideoCapture(camera_index)
    )


def build_face_detector(model_selection: int, min_detection_confidence: float) -> Any:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe is not installed.") from exc
    return mp.solutions.face_detection.FaceDetection(
        model_selection=model_selection,
        min_detection_confidence=min_detection_confidence,
    )


def select_largest_detection(detections, frame_width: int, frame_height: int):
    best      = None
    best_area = -1
    for detection in detections:
        box  = detection.location_data.relative_bounding_box
        x1   = max(0, int(box.xmin * frame_width))
        y1   = max(0, int(box.ymin * frame_height))
        x2   = min(frame_width,  int((box.xmin + box.width)  * frame_width))
        y2   = min(frame_height, int((box.ymin + box.height) * frame_height))
        bw   = max(0, x2 - x1)
        bh   = max(0, y2 - y1)
        area = bw * bh
        if area <= 0:
            continue
        score = float(detection.score[0]) if detection.score else 0.0
        if area > best_area:
            best_area = area
            best      = (x1, y1, bw, bh, score)
    return best


def run_face_tracking(
    camera_index: int,
    model_selection: int,
    min_detection_confidence: float,
    target_resolution: Optional[Tuple[int, int]] = None,
    log_display_kinds: Optional[List[str]] = None,
    serial_port: Optional[str] = None,
    serial_baudrate: int = 115200,
) -> None:
    set_log_display_kinds(log_display_kinds)
    cap = open_camera(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")
    if target_resolution is not None:
        w, h = target_resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
        cap.set(cv2.CAP_PROP_FPS,          30.0)

    serial_ctrl: Optional[SerialController] = None
    if serial_port:
        try:
            serial_ctrl = SerialController(port=serial_port, baudrate=serial_baudrate)
            if not serial_ctrl.connect():
                serial_ctrl = None
        except Exception as exc:
            log_line("ERROR", f"[Serial] init failed: {exc}", always=True)

    detector    = None
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
            frame   = cv2.flip(frame, 1)
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            h, w    = frame.shape[:2]
            cx, cy  = w // 2, h // 2
            cv2.drawMarker(
                frame, (cx, cy), (0, 180, 255),
                markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2,
            )

            target = (
                select_largest_detection(results.detections, w, h)
                if results.detections
                else None
            )
            if target is not None:
                x, y, fw, fh, score = target
                tx, ty = x + fw // 2, y + fh // 2
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (70, 240, 90), 2)
                cv2.circle(frame, (tx, ty), 4, (70, 240, 90), -1)
                cv2.line(frame, (cx, cy), (tx, ty), (70, 240, 90), 2)
                err_x  = (tx - cx) / max(1, cx)
                err_y  = (ty - cy) / max(1, cy)
                packet = f"POS,{err_x:.4f},{err_y:.4f},{score:.2f}"
                cv2.putText(
                    frame, packet, (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA,
                )
                log_line("POS", f"[Tracking] POS {err_x:.4f},{err_y:.4f} conf={score:.2f}")
                if serial_ctrl:
                    serial_ctrl.send_position_vector(err_x, err_y, score)
            else:
                cv2.putText(
                    frame, "No face detected", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 120, 240), 2, cv2.LINE_AA,
                )
                log_line("STAT", "[Tracking] No face detected")
                if serial_ctrl:
                    serial_ctrl.send_position_vector(0.0, 0.0, 0.0)

            now  = time.time()
            fps  = 1.0 / max(1e-6, now - prev)
            prev = now
            cv2.putText(
                frame, f"FPS {fps:.1f}", (16, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )
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


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time OpenCV MediaPipe face tracking")
    parser.add_argument("--max-cameras",              type=int,   default=10)
    parser.add_argument("--camera-index",             type=int,   default=None)
    parser.add_argument("--model-selection",          type=int,   choices=[0, 1], default=0)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--serial-port",              type=str,   default=None)
    parser.add_argument("--serial-baudrate",          type=int,   default=115200)
    parser.add_argument("--help-backend",             action="store_true")
    return parser.parse_args()


def print_backend_help() -> None:
    print("=" * 60)
    print("FACE TRACKING BACKEND")
    print("=" * 60)
    print("  1. Real-time face detection via MediaPipe")
    print("  2. LLM chat via OpenRouter (apikey_openrouter.ai)")
    print("  3. Serial output POS/ANIM to Arduino")
    print("  4. Optional wakeword pipeline (Alexa -> Whisper STT)")
    print("  5. German TTS via Piper (Thorsten-Voice High)")
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────
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
        cam_index         = selected.camera.index
        model_selection   = selected.model_selection
        target_resolution = selected.resolution
        log_display_kinds = selected.log_display_kinds
        serial_port       = selected.serial_port
        serial_baudrate   = selected.serial_baudrate
        use_wakeword      = selected.use_wakeword
        tts_enabled       = selected.tts_enabled
    else:
        cam_index         = args.camera_index
        model_selection   = args.model_selection
        target_resolution = None
        log_display_kinds = None
        serial_port       = args.serial_port
        serial_baudrate   = args.serial_baudrate
        use_wakeword      = False
        tts_enabled       = True

    def launch_gui():
        if use_wakeword:
            print("[main] Starting Voice Pipeline (Wakeword mode)...")
            _start_wakeword_gui(serial_port, serial_baudrate, tts_enabled)
        else:
            print("[main] Starting Chat Window...")
            start_chat_window(serial_port, serial_baudrate, tts_enabled)

    proc = multiprocessing.Process(target=launch_gui)
    proc.daemon = False
    proc.start()

    run_face_tracking(
        camera_index             = cam_index,
        model_selection          = model_selection,
        min_detection_confidence = args.min_detection_confidence,
        target_resolution        = target_resolution,
        log_display_kinds        = log_display_kinds,
        serial_port              = serial_port,
        serial_baudrate          = serial_baudrate,
    )


if __name__ == "__main__":
    main()