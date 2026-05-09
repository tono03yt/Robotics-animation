#!/usr/bin/env python3
"""
recording_test_cont.py — Continuous Lossless STT Pipeline.
Uses Queues and Voice Activity Detection (VAD) to segment full sentences.
GUI updates continuously while background threads handle processing without dropping frames.
"""

from __future__ import annotations

import collections
import queue
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import numpy as np
    import sounddevice as sd
    import soundfile as sf
    from faster_whisper import WhisperModel
except ModuleNotFoundError as exc:
    sys.exit(f"\nMissing dependency: {exc}\nInstall: pip install faster-whisper sounddevice soundfile numpy\n")

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    sys.exit("\nTkinter not found. Install: sudo apt-get install python3-tk\n")


# Audio and model config
SAMPLE_RATE = 16000
CHANNELS = 2
BLOCKSIZE = 1600  # 100 ms chunks
MIN_RMS = 0.010   # Volume threshold to trigger speech
PRE_ROLL_CHUNKS = 5   # Keep 0.5s of audio before speech starts to catch consonants
POST_ROLL_CHUNKS = 10 # Wait 1.0s of silence before finalizing the sentence

WHISPER_MODEL = "large-v3-turbo"
COMPUTE_TYPE = "int8"
DEVICE = "cpu"
LANGUAGE = "de"  # Set your target language here


class LosslessSpeechApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Lossless STT Pipeline")
        self.root.geometry("400x250")
        self.root.resizable(False, False)

        # Thread-safe GUI states
        self.is_running = True
        self.gain_value = tk.DoubleVar(value=1.0)
        self.rms_value = tk.DoubleVar(value=0.0)
        self.status_text = tk.StringVar(value="Initializing models...")
        self.internal_status = "Initializing models..." 

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Queues for lossless pipeline
        self.raw_audio_queue = queue.Queue()      # Holds 100ms chunks directly from mic
        self.utterance_queue = queue.Queue()      # Holds completed full sentences (numpy arrays)
        
        self.last_rms = 0.0

        # Start Background Threads
        threading.Thread(target=self._init_system, daemon=True).start()
        threading.Thread(target=self._vad_worker, daemon=True).start()
        threading.Thread(target=self._stt_worker, daemon=True).start()

        # Start standard Tkinter UI update loop
        self.root.after(100, self._update_gui_loop)

    def _build_ui(self):
        """Build the GUI layout."""
        title = tk.Label(self.root, text="🎤 Continuous STT", font=("Arial", 14, "bold"))
        title.pack(pady=10)

        # Volume Display
        vol_frame = tk.LabelFrame(self.root, text="Input Level (RMS)", padx=10, pady=10)
        vol_frame.pack(fill="x", padx=10, pady=5)
        
        self.vol_bar = ttk.Progressbar(vol_frame, variable=self.rms_value, maximum=0.1, mode="determinate")
        self.vol_bar.pack(fill="x", pady=5)
        
        self.vol_label = tk.Label(vol_frame, text="0.000", font=("Courier", 11, "bold"))
        self.vol_label.pack()

        # Gain Control
        gain_frame = tk.LabelFrame(self.root, text="Gain Multiplier", padx=10, pady=10)
        gain_frame.pack(fill="x", padx=10, pady=5)
        
        self.gain_slider = ttk.Scale(
            gain_frame, from_=0.1, to=5.0, variable=self.gain_value, orient="horizontal"
        )
        self.gain_slider.pack(fill="x", pady=5)

        # Status
        status_frame = tk.LabelFrame(self.root, text="Pipeline Status", padx=10, pady=10)
        status_frame.pack(fill="x", padx=10, pady=5)
        
        self.status_label = tk.Label(status_frame, textvariable=self.status_text, font=("Arial", 10), wraplength=350)
        self.status_label.pack()

    def _init_system(self):
        """Background initialization of models and audio stream."""
        print("[Init] Loading Whisper model...", end=" ", flush=True)
        try:
            self.whisper = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
            print("✓")
        except Exception as e:
            print(f"✗\n[Error] Failed to load Whisper: {e}")
            self.internal_status = "Error loading Whisper!"
            return

        self.internal_status = "Listening for speech..."
        print("\n[Run] Pipeline started. Speak into the microphone. Output will appear here.\n")

        # Start Audio Stream
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=BLOCKSIZE,
                callback=self._audio_callback,
            )
            self.stream.start()
        except Exception as e:
            print(f"[Stream Error] {e}")
            self.internal_status = f"Mic Error: {e}"

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Instantaneous audio callback. Drops data immediately into the raw queue."""
        if not self.is_running:
            return

        try:
            # Extract first channel, apply gain
            chunk = indata[:, 0].astype(np.float32) if indata.ndim == 2 else indata.astype(np.float32)
            chunk = chunk * self.gain_value.get()
            
            # Fast RMS calculation
            rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
            self.last_rms = rms

            # Immediately queue data, blocking nothing
            self.raw_queue_put(chunk, rms)
        except Exception as e:
            print(f"\n[Audio Callback Error] {e}")

    def raw_queue_put(self, chunk, rms):
        # Prevent infinite memory if processing crashes
        if self.raw_audio_queue.qsize() < 1000:  
            self.raw_audio_queue.put((chunk, rms))

    def _vad_worker(self):
        """
        Voice Activity Detection Thread.
        Reads 100ms chunks, detects speech boundaries, and packages full sentences.
        """
        pre_roll_buffer = collections.deque(maxlen=PRE_ROLL_CHUNKS)
        is_speaking = False
        silence_counter = 0
        current_utterance = []

        while self.is_running:
            try:
                # Get next chunk
                chunk, rms = self.raw_audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not is_speaking:
                if rms >= MIN_RMS:
                    # Speech started!
                    is_speaking = True
                    self.internal_status = "🟢 Recording Sentence..."
                    
                    # Attach the pre-roll (last 0.5 seconds) so we don't chop the first consonant
                    current_utterance = list(pre_roll_buffer)
                    current_utterance.append(chunk)
                    silence_counter = 0
                else:
                    # Maintain rolling window of silence
                    pre_roll_buffer.append(chunk)
            else:
                # Currently recording a sentence
                current_utterance.append(chunk)
                
                if rms < MIN_RMS:
                    silence_counter += 1
                    if silence_counter >= POST_ROLL_CHUNKS:
                        # 1.0 second of silence detected. Sentence complete!
                        is_speaking = False
                        self.internal_status = "⚙️ Queuing Sentence..."
                        
                        full_audio = np.concatenate(current_utterance)
                        self.utterance_queue.put(full_audio)
                        
                        current_utterance = []
                        pre_roll_buffer.clear()
                else:
                    # Still speaking, reset silence counter
                    silence_counter = 0

    def _stt_worker(self):
        """
        Transcription Thread.
        Pops completed sentences off the queue and transcribes them safely in the background.
        """
        while self.is_running:
            try:
                # Wait for a fully recorded sentence
                audio_data = self.utterance_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # If multiple sentences stack up in the queue, let the user know
            q_size = self.utterance_queue.qsize()
            backlog_text = f" (+{q_size} in queue)" if q_size > 0 else ""
            self.internal_status = f"📝 Transcribing{backlog_text}..."

            text = self._transcribe(audio_data)

            if text:
                print(f"[STT] {text}")
                self.internal_status = "Listening for speech..."
            else:
                self.internal_status = "Listening for speech..."

    def _transcribe(self, audio_array: np.ndarray) -> str:
        """Helper to invoke faster-whisper over an audio array."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)
            
        try:
            sf.write(str(wav_path), audio_array, SAMPLE_RATE)
            segments, _ = self.whisper.transcribe(
                str(wav_path),
                language=LANGUAGE,
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=False,
            )
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as e:
            print(f"[STT Error] {e}")
            return ""
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _update_gui_loop(self):
        """Main thread loop to sync state into the GUI safely."""
        if not self.is_running:
            return

        # Synchronize UI values with class state
        rms = self.last_rms
        self.rms_value.set(min(rms, 0.1))
        
        # Color bar green when speaking, blue when silent
        if rms >= MIN_RMS:
            self.vol_label.config(text=f"{rms:.4f}", fg="green")
        else:
            self.vol_label.config(text=f"{rms:.4f}", fg="black")

        self.status_text.set(self.internal_status)

        # Reschedule update
        self.root.after(50, self._update_gui_loop)

    def _on_close(self):
        """Cleanup logic when application window is closed."""
        self.is_running = False
        self.internal_status = "Shutting down..."
        
        try:
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass

        self.root.destroy()
        print("\n[Exit] Application closed.")
        sys.exit(0)


def main():
    root = tk.Tk()
    app = LosslessSpeechApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()