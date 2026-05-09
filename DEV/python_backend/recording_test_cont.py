#!/usr/bin/env python3
"""
recording_test.py
Live microphone sampling -> faster-whisper STT -> Kokoro TTS playback.
Press Ctrl+C to stop.
"""

from __future__ import annotations

import queue
import signal
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

try:
    import numpy as np
    import sounddevice as sd
    import soundfile as sf
    from faster_whisper import WhisperModel
except ModuleNotFoundError as exc:
    sys.exit(f"\nMissing dependency: {exc}\nInstall: pip install faster-whisper sounddevice soundfile numpy\n")

try:
    from kokoro import KPipeline
except ModuleNotFoundError:
    sys.exit("\nKokoro not found. Install: pip install kokoro\n")


# Audio and model config
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 1600  # 100 ms
WINDOW_SEC = 2.0
HOP_SEC = 1.0
MIN_RMS = 0.008

WHISPER_MODEL = "large-v3-turbo"
COMPUTE_TYPE = "int8"
DEVICE = "cpu"

KOKORO_VOICE = "af_sarah"
KOKORO_LANG = "a"
KOKORO_RATE = 24000


class LiveSpeechLoop:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
        self.playback_lock = threading.Lock()
        self.whisper = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
        self.kokoro = KPipeline(lang_code=KOKORO_LANG)

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"[Audio] input status: {status}")
        if self.stop_event.is_set():
            return
        chunk = indata[:, 0].astype(np.float32) if indata.ndim == 2 else indata.astype(np.float32)
        try:
            self.audio_queue.put_nowait(chunk.copy())
        except queue.Full:
            # Drop oldest chunk behavior by skipping new data when overloaded.
            pass

    def _transcribe_chunk(self, chunk: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)
        try:
            sf.write(str(wav_path), chunk, SAMPLE_RATE)
            segments, _ = self.whisper.transcribe(
                str(wav_path),
                language="de",
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 400},
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _speak(self, text: str) -> None:
        with self.playback_lock:
            audio_chunks = []
            for _, _, audio in self.kokoro(text, voice=KOKORO_VOICE, speed=1.0):
                audio_chunks.append(audio)
            if not audio_chunks:
                return
            audio = np.concatenate(audio_chunks)
            sd.play(audio, samplerate=KOKORO_RATE)
            sd.wait()

    def _processor_loop(self) -> None:
        window_samples = int(WINDOW_SEC * SAMPLE_RATE)
        hop_samples = int(HOP_SEC * SAMPLE_RATE)
        buffer = np.zeros(0, dtype=np.float32)

        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            buffer = np.concatenate([buffer, chunk])
            while buffer.size >= window_samples:
                window = buffer[:window_samples]
                buffer = buffer[hop_samples:] if hop_samples < buffer.size else np.zeros(0, dtype=np.float32)

                rms = float(np.sqrt(np.mean(window * window))) if window.size else 0.0
                if rms < MIN_RMS:
                    continue

                text = self._transcribe_chunk(window)
                if not text:
                    continue

                print(f"[STT] {text}")
                self._speak(text)

    def run(self) -> None:
        print("[Init] Loading models done.")
        print("[Run] Live sampling started. Press Ctrl+C to stop.")

        worker = threading.Thread(target=self._processor_loop, daemon=True)
        worker.start()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=BLOCKSIZE,
            callback=self._callback,
        ):
            while not self.stop_event.is_set():
                time.sleep(0.1)

    def stop(self) -> None:
        self.stop_event.set()


def main() -> None:
    loop = LiveSpeechLoop()

    def _handle_sigint(signum, frame) -> None:
        loop.stop()

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        loop.run()
    except KeyboardInterrupt:
        loop.stop()
    finally:
        print("[Exit] Stopped live sampling.")


if __name__ == "__main__":
    main()
