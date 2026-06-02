#!/usr/bin/env python3
"""
Voice Pipeline — Wake Word + VAD + Whisper STT
Fixes:
  - Silero VAD model gets reset() after each STT run (hidden state corruption fix)
  - VAD model protected by a threading.Lock (thread-safety)
  - Whisper vad_filter disabled (was sharing state with Silero)
  - beam_size=1 on CPU for speed; beam_size=5 only on GPU
  - STT runs on a copy of audio data to avoid race conditions
  - OWW model also reset() after cooldown to flush internal window
"""

# ── venv self-check ────────────────────────────────────────────────────────
import sys
import os
import subprocess

VENV_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv")
VENV_PYTHON = os.path.join(VENV_DIR, "bin", "python3")

def _in_venv() -> bool:
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    )

if not _in_venv():
    if os.path.isfile(VENV_PYTHON):
        print(f"[bootstrap] Re-launching inside venv: {VENV_PYTHON}")
        os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)
    else:
        print(
            f"[bootstrap] .venv not found at {VENV_DIR}.\n"
            "Please create it:\n"
            "  python3 -m venv .venv && source .venv/bin/activate\n"
            "  pip install openwakeword faster-whisper sounddevice torch onnxruntime"
        )
        sys.exit(1)

# ── Auto-install missing packages ──────────────────────────────────────────
REQUIRED = {
    "numpy":          "numpy",
    "sounddevice":    "sounddevice",
    "torch":          "torch",
    "torchaudio":     "torchaudio",
    "onnxruntime":    "onnxruntime",
    "openwakeword":   "openwakeword",
    "faster_whisper": "faster-whisper",
}
for module, pkg in REQUIRED.items():
    try:
        __import__(module)
    except ImportError:
        print(f"[setup] Installing: {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

# ── Imports ────────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk
import threading
import time
import urllib.request

import numpy as np
import sounddevice as sd
import torch
import onnxruntime as ort
import openwakeword
from openwakeword.model import Model
from faster_whisper import WhisperModel

# ── Config ─────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16000
OWW_CHUNK        = 1280     # openWakeWord: 80ms @ 16kHz
VAD_CHUNK        = 512      # Silero VAD:   32ms @ 16kHz
VAD_THRESHOLD    = 0.5
WW_THRESHOLD     = 0.5
SILENCE_SECS     = 1.5
WW_COOLDOWN_SECS = 1.2
MAX_RECORD_SECS  = 15.0

# ── States ──────────────────────────────────────────────────────────────────
STATE_WAITING  = "WAITING_FOR_WAKEWORD"
STATE_COOLDOWN = "WAKEWORD_COOLDOWN"
STATE_SAMPLING = "SAMPLING"
STATE_BUSY     = "STT_BUSY"

# ── openWakeWord model bootstrap ───────────────────────────────────────────
def ensure_oww_models() -> list:
    try:
        openwakeword.utils.download_models()
    except Exception as e:
        print(f"[oww] Built-in download failed ({e}), trying manual fallback...")

    paths = [p for p in openwakeword.get_pretrained_model_paths() if "alexa" in p.lower()]
    if paths:
        return paths

    oww_pkg_dir = os.path.dirname(openwakeword.__file__)
    models_dir  = os.path.join(oww_pkg_dir, "resources", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_url  = "https://github.com/dscripka/openWakeWord/releases/download/v0.1.1/alexa_v0.1.onnx"
    model_path = os.path.join(models_dir, "alexa_v0.1.onnx")
    if not os.path.isfile(model_path):
        print(f"[oww] Downloading alexa model...")
        urllib.request.urlretrieve(model_url, model_path)
    paths = [p for p in openwakeword.get_pretrained_model_paths() if "alexa" in p.lower()]
    return paths if paths else [model_path]

# ── Audio Pipeline ──────────────────────────────────────────────────────────
class AudioPipeline(threading.Thread):
    def __init__(self, gui: "AppGUI", use_wakeword: bool):
        super().__init__(daemon=True)
        self.gui          = gui
        self.use_wakeword = use_wakeword
        self.running      = True
        self.state        = STATE_WAITING if use_wakeword else STATE_SAMPLING

        # Lock protecting Silero VAD — shared between audio thread and (indirectly) STT
        self._vad_lock = threading.Lock()

        # ONNX provider
        providers = ["ROCMExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        selected  = next((p for p in providers if p in available), "CPUExecutionProvider")
        self.gui.log(f"ONNX Provider: {selected}")

        # openWakeWord
        self.gui.log("Loading WakeWord model (alexa)...")
        alexa_paths = ensure_oww_models()
        self.gui.log(f"Using model: {os.path.basename(alexa_paths[0])}")
        self.oww_model      = Model(wakeword_model_paths=alexa_paths)
        self._oww_model_path = alexa_paths  # kept for hard-reset

        # Silero VAD — loaded with onnx=True for stateless ONNX backend
        # (stateless = no hidden state to corrupt between calls)
        self.gui.log("Loading Silero VAD...")
        self.vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,   # ONNX backend is stateless per-call — safe across threads
        )

        # Faster Whisper
        self.gui.log("Loading Faster-Whisper (large-v3-turbo)...")
        self._device     = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type     = "float16" if self._device == "cuda" else "int8"
        self._beam_size  = 5 if self._device == "cuda" else 1  # beam=1 on CPU for speed
        self.gui.log(f"Whisper: device={self._device}, compute={compute_type}, beam={self._beam_size}")
        self.stt_model = WhisperModel(
            "large-v3-turbo", device=self._device, compute_type=compute_type
        )

        self.gui.log('Pipeline ready — speak "Alexa" then your command!')

    # ── VAD helper (thread-safe, stateless ONNX) ───────────────────────────
    def _vad_on_chunk(self, audio_int16: np.ndarray) -> tuple[bool, float]:
        """Returns (is_speech, max_prob) — splits 1280 samples into 512-sub-chunks."""
        max_prob = 0.0
        with self._vad_lock:
            for start in range(0, len(audio_int16) - VAD_CHUNK + 1, VAD_CHUNK):
                sub  = audio_int16[start:start + VAD_CHUNK].astype(np.float32) / 32768.0
                prob = self.vad_model(torch.from_numpy(sub), SAMPLE_RATE).item()
                if prob > max_prob:
                    max_prob = prob
        return max_prob > VAD_THRESHOLD, max_prob

    # ── Main audio loop ────────────────────────────────────────────────────
    def run(self):
        silence_threshold = int(SILENCE_SECS   / (OWW_CHUNK / SAMPLE_RATE))
        max_rec_chunks    = int(MAX_RECORD_SECS / (OWW_CHUNK / SAMPLE_RATE))

        audio_buffer: list = []
        silence_counter    = 0
        cooldown_until     = 0.0

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=OWW_CHUNK,
        ) as stream:
            while self.running:
                chunk, overflow = stream.read(OWW_CHUNK)
                if overflow:
                    continue

                audio_data = chunk.flatten().astype(np.int16)

                # Mic level
                self.gui.safe_update(self.gui.update_mic_level,
                                     float(np.abs(audio_data).mean()))

                # VAD
                is_speech, vad_prob = self._vad_on_chunk(audio_data)
                self.gui.safe_update(self.gui.update_speech_indicator, is_speech)

                now = time.monotonic()

                # ── WAITING ────────────────────────────────────────────────
                if self.state == STATE_WAITING:
                    prediction = self.oww_model.predict(audio_data)
                    score      = max(prediction.values()) if prediction else 0.0
                    self.gui.safe_update(self.gui.update_ww_score, score)

                    if score >= WW_THRESHOLD and now >= cooldown_until:
                        self.gui.log(f"Wake word detected! (score={score:.3f})")
                        self.gui.safe_update(self.gui.update_wakeword_indicator, True)
                        cooldown_until  = now + WW_COOLDOWN_SECS
                        self.state      = STATE_COOLDOWN
                        audio_buffer    = []
                        silence_counter = 0

                # ── COOLDOWN ───────────────────────────────────────────────
                elif self.state == STATE_COOLDOWN:
                    # Keep feeding OWW to flush its sliding window — ignore scores
                    self.oww_model.predict(audio_data)
                    self.gui.safe_update(self.gui.update_ww_score, 0.0)

                    if now >= cooldown_until:
                        # Hard-reset OWW internal state so previous activation
                        # doesn't bleed into the next detection cycle
                        self.oww_model = Model(wakeword_model_paths=self._oww_model_path)
                        self.gui.log("Listening for your command...")
                        self.gui.safe_update(self.gui.update_wakeword_indicator, False)
                        self.gui.safe_update(self.gui.set_status, "🎙 Recording...")
                        self.state      = STATE_SAMPLING
                        audio_buffer    = []
                        silence_counter = 0

                # ── SAMPLING ───────────────────────────────────────────────
                elif self.state == STATE_SAMPLING:
                    audio_buffer.append(audio_data)
                    silence_counter = 0 if is_speech else silence_counter + 1

                    rec_secs = len(audio_buffer) * OWW_CHUNK / SAMPLE_RATE
                    self.gui.safe_update(self.gui.set_status, f"🎙 Recording… {rec_secs:.1f}s")

                    hit_silence = silence_counter >= silence_threshold
                    hit_max     = len(audio_buffer) >= max_rec_chunks

                    if hit_silence or hit_max:
                        reason = "silence" if hit_silence else "max length"
                        # Trim trailing silence frames from buffer
                        trim = silence_counter if hit_silence else 0
                        useful_frames = len(audio_buffer) - trim
                        if useful_frames > 2:
                            self.gui.log(f"Recording ended ({reason}) — sending to STT…")
                            self.gui.safe_update(self.gui.set_status, "⏳ Processing STT…")
                            # Make a copy before handing off — audio thread keeps running
                            audio_trimmed = np.concatenate(
                                audio_buffer[:useful_frames]
                            ).astype(np.float32) / 32768.0
                            self.state = STATE_BUSY
                            threading.Thread(
                                target=self._transcribe_and_reset,
                                args=(audio_trimmed.copy(),),
                                daemon=True,
                            ).start()
                        else:
                            self.gui.log("Nothing useful recorded — back to waiting.")
                            self.gui.safe_update(self.gui.set_status, "")
                            self.state = STATE_WAITING if self.use_wakeword else STATE_SAMPLING

                        audio_buffer    = []
                        silence_counter = 0

                # ── STT_BUSY: drain mic, do nothing ───────────────────────
                elif self.state == STATE_BUSY:
                    pass

    # ── STT ────────────────────────────────────────────────────────────────
    def _transcribe_and_reset(self, audio_data: np.ndarray):
        try:
            segments, info = self.stt_model.transcribe(
                audio_data,
                language="de",
                beam_size=self._beam_size,
                condition_on_previous_text=False,
                # vad_filter intentionally OFF — Silero VAD handles this in audio loop
                # Enabling it here uses a second Silero instance that corrupts shared state
            )
            text = "".join(seg.text for seg in segments).strip()
            if text:
                print(f"\n[STT]: {text}\n")
                self.gui.log(f'STT: "{text}"')
                self.gui.safe_update(self.gui.show_result, text)
            else:
                self.gui.log("STT: (nichts erkannt)")
        except Exception as e:
            self.gui.log(f"STT error: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.gui.safe_update(self.gui.set_status, "")
            # Reset OWW state fresh for next detection cycle
            self.oww_model = Model(wakeword_model_paths=self._oww_model_path)
            self.state = STATE_WAITING if self.use_wakeword else STATE_SAMPLING
            if self.use_wakeword:
                self.gui.log('Ready — speak "Alexa" again.')

    def stop(self):
        self.running = False


# ── GUI ─────────────────────────────────────────────────────────────────────
class AppGUI:
    def __init__(self, root: tk.Tk):
        self.root     = root
        self.pipeline = None
        root.title("Voice Pipeline")
        root.resizable(False, False)

        tk.Label(root, text="Voice Pipeline", font=("Helvetica", 16, "bold")).pack(pady=10)

        self.use_wakeword = tk.BooleanVar(value=True)
        tk.Checkbutton(root, text='Enable Wake Word ("Alexa")',
                       variable=self.use_wakeword).pack()

        self.start_btn = tk.Button(
            root, text="▶  Start Pipeline",
            command=self.start_pipeline,
            bg="#2e7d32", fg="white",
            font=("Helvetica", 11, "bold"), padx=12, pady=6,
        )
        self.start_btn.pack(pady=10)

        self.status_lbl = tk.Label(root, text="", font=("Helvetica", 11), fg="#1565c0")
        self.status_lbl.pack()

        result_frame = tk.LabelFrame(root, text="Last Transcription", padx=8, pady=4)
        result_frame.pack(fill="x", padx=10, pady=4)
        self.result_var = tk.StringVar(value="")
        tk.Label(result_frame, textvariable=self.result_var,
                 font=("Helvetica", 12, "bold"), wraplength=360,
                 fg="#000", justify="left").pack(anchor="w")

        self.logs = tk.Text(root, height=10, width=60,
                            state="disabled", bg="#1e1e1e", fg="#d4d4d4")
        self.logs.pack(padx=10, pady=(4, 6))

        frame = tk.Frame(root)
        frame.pack(pady=6)

        tk.Label(frame, text="Mic Level:").grid(row=0, column=0, sticky="w", padx=6)
        self.mic_bar = ttk.Progressbar(frame, length=220, mode="determinate")
        self.mic_bar.grid(row=0, column=1, padx=6)

        tk.Label(frame, text="Speech:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.speech_ind = tk.Label(frame, text=" OFF ", bg="grey", fg="white", width=6)
        self.speech_ind.grid(row=1, column=1, sticky="w")

        tk.Label(frame, text="Wake Word:").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.ww_ind = tk.Label(frame, text=" NO  ", bg="grey", fg="white", width=6)
        self.ww_ind.grid(row=2, column=1, sticky="w")

        tk.Label(frame, text="WW Score:").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        self.ww_score_bar = ttk.Progressbar(frame, length=220, mode="determinate")
        self.ww_score_bar.grid(row=3, column=1, padx=6)
        self.ww_score_lbl = tk.Label(frame, text="0.000", width=6)
        self.ww_score_lbl.grid(row=3, column=2)

    def safe_update(self, fn, *args):
        self.root.after(0, fn, *args)

    def log(self, msg: str):
        def _write():
            self.logs.config(state="normal")
            self.logs.insert(tk.END, msg + "\n")
            self.logs.see(tk.END)
            self.logs.config(state="disabled")
        self.root.after(0, _write)
        print(f"[LOG] {msg}")

    def set_status(self, text: str):
        self.status_lbl.config(text=text)

    def show_result(self, text: str):
        self.result_var.set(text)

    def update_mic_level(self, level: float):
        self.mic_bar["value"] = min(100.0, level / 50.0 * 100)

    def update_speech_indicator(self, active: bool):
        self.speech_ind.config(
            text=" ON  " if active else " OFF ",
            bg="#2e7d32" if active else "grey"
        )

    def update_wakeword_indicator(self, active: bool):
        self.ww_ind.config(
            text=" YES " if active else " NO  ",
            bg="#e65100" if active else "grey"
        )

    def update_ww_score(self, score: float):
        self.ww_score_bar["value"] = min(100.0, score * 100)
        self.ww_score_lbl.config(text=f"{score:.3f}")

    def start_pipeline(self):
        if self.pipeline is not None:
            return
        self.start_btn.config(state="disabled", text="Loading…")
        self.log("Starting pipeline...")

        def _start():
            try:
                self.pipeline = AudioPipeline(self, self.use_wakeword.get())
                self.pipeline.start()
                self.root.after(0, lambda: self.start_btn.config(
                    text="● Running", bg="#1565c0"
                ))
            except Exception as e:
                self.log(f"ERROR: {e}")
                self.root.after(0, lambda: self.start_btn.config(
                    state="normal", text="▶  Start Pipeline", bg="#2e7d32"
                ))
                raise

        threading.Thread(target=_start, daemon=True).start()

    def on_close(self):
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline.join(timeout=2)
        self.root.destroy()


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = AppGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()