#!/usr/bin/env python3
"""
German TTS Pipeline — Piper TTS + Thorsten-Voice (High Quality)
piper-tts 1.4.x: AudioChunk.audio_int16_bytes
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
        print("Run: python3 -m venv .venv && source .venv/bin/activate\n"
              "     pip install piper-tts sounddevice numpy")
        sys.exit(1)

# ── Auto-install ───────────────────────────────────────────────────────────
for module, pkg in {"sounddevice": "sounddevice", "numpy": "numpy"}.items():
    try:
        __import__(module)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

try:
    from piper.voice import PiperVoice
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "piper-tts", "--quiet"])
    from piper.voice import PiperVoice

# ── Imports ────────────────────────────────────────────────────────────────
import importlib.metadata
import time
import urllib.request

import numpy as np
import sounddevice as sd

# ── Extract PCM bytes from any piper AudioChunk version ───────────────────
def _chunk_to_pcm(chunk) -> bytes:
    """
    piper 1.4.x AudioChunk attributes (confirmed from error output):
      audio_int16_bytes  ← raw int16 PCM bytes  ✅  use this
      audio_int16_array  ← numpy int16 array
      audio_float_array  ← numpy float32 array
    """
    if isinstance(chunk, (bytes, bytearray)):
        return bytes(chunk)                              # legacy < 1.3
    if hasattr(chunk, "audio_int16_bytes"):
        return bytes(chunk.audio_int16_bytes)            # piper 1.4.x ✅
    if hasattr(chunk, "audio_int16_array"):
        return chunk.audio_int16_array.tobytes()
    if hasattr(chunk, "audio_float_array"):
        return (chunk.audio_float_array * 32767).astype(np.int16).tobytes()
    if hasattr(chunk, "audio"):
        d = chunk.audio
        if isinstance(d, (bytes, bytearray)):
            return bytes(d)
        return np.asarray(d, dtype=np.int16).tobytes()
    raise TypeError(f"Cannot extract PCM from {type(chunk)}: {dir(chunk)}")

# ── Model paths ────────────────────────────────────────────────────────────
MODEL_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_models")
MODEL_FILE  = os.path.join(MODEL_DIR, "de_DE-thorsten-high.onnx")
CONFIG_FILE = os.path.join(MODEL_DIR, "de_DE-thorsten-high.onnx.json")
HF_BASE     = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/high"

def ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    def _bar(b, bs, total):
        done = min(b * bs, total)
        pct  = int(done / total * 40) if total > 0 else 0
        print(f"\r      [{'█'*pct}{'░'*(40-pct)}] {done/1024/1024:.1f}/{total/1024/1024:.1f} MB",
              end="", flush=True)

    if not os.path.isfile(MODEL_FILE):
        print("[tts] Downloading Thorsten-Voice HIGH (~65 MB)...")
        urllib.request.urlretrieve(f"{HF_BASE}/de_DE-thorsten-high.onnx", MODEL_FILE, _bar)
        print()
    if not os.path.isfile(CONFIG_FILE):
        print("[tts] Downloading model config...")
        urllib.request.urlretrieve(f"{HF_BASE}/de_DE-thorsten-high.onnx.json", CONFIG_FILE)
    print("[tts] Model ready.")

# ── Speak ──────────────────────────────────────────────────────────────────
def speak(voice: PiperVoice, text: str):
    text = text.strip()
    if not text:
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
        print("[tts] Warning: no audio generated!")
        return

    audio = np.concatenate(pcm_parts)
    sd.play(audio, samplerate=voice.config.sample_rate)
    sd.wait()

# ── Main REPL ──────────────────────────────────────────────────────────────
def main():
    print(f"[tts] piper-tts {importlib.metadata.version('piper-tts')}")
    ensure_model()
    print("[tts] Loading voice model...")
    voice = PiperVoice.load(MODEL_FILE, config_path=CONFIG_FILE)
    print(f"[tts] Sample rate: {voice.config.sample_rate} Hz — ready!")
    print()
    print("=" * 55)
    print("  Deutsches TTS — Thorsten-Voice (High Quality)")
    print("  Gib Text ein und drücke Enter zum Vorlesen.")
    print("  'exit' oder Ctrl+C zum Beenden.")
    print("=" * 55)
    print()

    speak(voice, "Hallo! Ich bin bereit. Bitte gib deinen Text ein.")

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            speak(voice, "Auf Wiedersehen!")
            print("\n[tts] Beendet.")
            break

        if text.lower() in ("exit", "quit", "beenden", "ende"):
            speak(voice, "Auf Wiedersehen!")
            print("[tts] Beendet.")
            break

        if text:
            speak(voice, text)

if __name__ == "__main__":
    main()