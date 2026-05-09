#!/usr/bin/env python3
"""
stt_tts.py — Record mic → faster-whisper (German) STT → Kokoro-82M TTS → play back.
Press Ctrl+C to stop recording.
"""

import sys
import time
import warnings
import tempfile

warnings.filterwarnings("ignore")

try:
    import numpy as np
    import sounddevice as sd
    import soundfile as sf
    from faster_whisper import WhisperModel
except ModuleNotFoundError as e:
    sys.exit(f"\n❌  Missing: {e}\n    pip install faster-whisper sounddevice soundfile numpy\n")

try:
    from kokoro import KPipeline
except ModuleNotFoundError:
    sys.exit("\n❌  Kokoro not found.\n    pip install kokoro && sudo apt install espeak-ng\n")

# ── Config ──────────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16_000
CHANNELS       = 1

# faster-whisper: use a German-fine-tuned model for best accuracy,
# or fall back to "large-v3" which also handles German very well.
# Options:
#   "base"       → fast, decent German
#   "large-v3"   → best multilingual accuracy (slower, ~1.5 GB)
#   "TheChola/whisper-large-v3-turbo-german-faster-whisper" → German-optimised turbo
WHISPER_MODEL  = "large-v3-turbo"
COMPUTE_TYPE   = "int8"       # int8 = fast CPU; float16 needs GPU/ROCm
DEVICE         = "cpu"        # change to "cuda" if ROCm is configured

KOKORO_VOICE   = "af_sarah"   # af_bella | am_adam | am_michael | bf_emma | bm_george
KOKORO_LANG    = "a"          # 'a' = American English, 'b' = British English

# ── Terminal colours ────────────────────────────────────────────────────────────
BOLD   = "\033[1m"; GREEN  = "\033[32m"; CYAN   = "\033[36m"
YELLOW = "\033[33m"; DIM   = "\033[2m";  RESET  = "\033[0m"

# ── Record mic ──────────────────────────────────────────────────────────────────
def record(wav_path: str) -> None:
    print(f"\n{BOLD}🎙  Aufnahme läuft…{RESET} {DIM}Strg+C zum Beenden.{RESET}\n")
    chunks = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                            dtype="int16", callback=callback):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    audio = np.concatenate(chunks, axis=0)
    sf.write(wav_path, audio, SAMPLE_RATE)
    print(f"\n{DIM}✅ Aufgenommen: {len(audio) / SAMPLE_RATE:.1f}s{RESET}")

# ── faster-whisper STT (German) ─────────────────────────────────────────────────
def transcribe(wav_path: str) -> str:
    print(f"{DIM}⏳ Lade faster-whisper ({WHISPER_MODEL}) auf {DEVICE}…{RESET}")
    model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)

    print(f"{DIM}🔍 Transkribiere (Deutsch)…{RESET}")
    segments, info = model.transcribe(
        wav_path,
        language="de",                        # force German
        beam_size=5,
        best_of=5,
        temperature=0.0,                       # deterministic output
        condition_on_previous_text=False,
        vad_filter=True,                       # skip silence chunks
        vad_parameters=dict(
            min_silence_duration_ms=500
        ),
    )

    text = " ".join(seg.text.strip() for seg in segments)
    print(f"{DIM}🌐 Erkannte Sprache: {info.language} "
          f"(Konfidenz: {info.language_probability:.0%}){RESET}")
    return text.strip()

# ── Kokoro TTS ──────────────────────────────────────────────────────────────────
def speak(text: str) -> None:
    print(f"{DIM}🔊 Kokoro synthetisiert ({KOKORO_VOICE})…{RESET}")
    pipeline     = KPipeline(lang_code=KOKORO_LANG)
    audio_chunks = []

    for _, _, audio in pipeline(text, voice=KOKORO_VOICE, speed=1.0):
        audio_chunks.append(audio)

    if not audio_chunks:
        print(f"{DIM}⚠  Kein Audio generiert.{RESET}")
        return

    sd.play(np.concatenate(audio_chunks), samplerate=24_000)
    sd.wait()

# ── Print transcript word-by-word ───────────────────────────────────────────────
def print_transcript(text: str) -> None:
    print(f"\n{BOLD}{GREEN}📝 Transkript:{RESET}\n")
    for word in text.split():
        sys.stdout.write(f"{CYAN}{word}{RESET} ")
        sys.stdout.flush()
        time.sleep(0.04)
    print("\n")

# ── Main ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    record(wav_path)
    text = transcribe(wav_path)
    print_transcript(text)

    print(f"{BOLD}{YELLOW}🗣  Spreche…{RESET}")
    speak(text)
