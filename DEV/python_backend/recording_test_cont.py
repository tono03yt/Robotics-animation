#!/usr/bin/env python3
"""
recording_test_cont.py — Continuous Lossless STT Pipeline with Wakeword + VAD.
"""

from __future__ import annotations

import collections
import os
import queue
import sys
import signal
import threading
import time
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("ORT_DISABLE_CUDA", "1")

warnings.filterwarnings("ignore")

try:
    import numpy as np
    import sounddevice as sd
    from faster_whisper import WhisperModel
except ModuleNotFoundError as exc:
    sys.exit(f"\nMissing dependency: {exc}\nInstall: pip install faster-whisper sounddevice numpy\n")

try:
    import webrtcvad
except ModuleNotFoundError as exc:
    sys.exit(f"\nMissing dependency: {exc}\nInstall: pip install webrtcvad\n")

try:
    import openwakeword
    from openwakeword.model import Model as WakewordModel
except ModuleNotFoundError as exc:
    sys.exit(f"\nMissing dependency: {exc}\nInstall: pip install openwakeword\n")

try:
    from kokoro import KPipeline
except (ModuleNotFoundError, ImportError):
    sys.exit("\nKokoro not found or KPipeline could not be imported. Install/reinstall: pip install --upgrade kokoro\n")

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    sys.exit("\nTkinter not found. Install: sudo apt-get install python3-tk\n")


# Audio and model config
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 1600  # 100 ms chunks
MIN_RMS = 0.010   # Volume threshold to trigger speech
PRE_ROLL_CHUNKS = 5   # Keep 0.5s of audio before speech starts
POST_ROLL_CHUNKS = 10 # Wait 1.0s of silence before finalizing sentence

WAKEWORD_MODEL_NAME = "hey_jarvis"
WAKEWORD_MODEL_PATH = ""  # Optional: full path to a custom openwakeword model
WAKEWORD_DISPLAY = "hey jarvis"
WAKEWORD_THRESHOLD = 0.5
WAKEWORD_DEBOUNCE_FRAMES = 3
WAKEWORD_FRAME_MS = 80
WAKEWORD_FRAME_SAMPLES = int(SAMPLE_RATE * WAKEWORD_FRAME_MS / 1000)

VAD_AGGRESSIVENESS = 2
VAD_FRAME_MS = 30
VAD_FRAME_SAMPLES = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
VAD_SPEECH_FRAMES_REQUIRED = 2

RING_BUFFER_SECONDS = 15

WHISPER_MODEL_TRANSCRIPTION = "large-v3-turbo"
COMPUTE_TYPE = "int8"
DEVICE = "cpu"
LANGUAGE = "de"  
KOKORO_LANG = "de"
KOKORO_RATE = 24000

class VolumeMonitorGUI:
    def __init__(self, root_window=None, on_close=None):
        self.root = root_window or tk.Tk()
        self.root.title("Recording Monitor - Live STT/TTS")
        self.root.geometry("400x440")
        self.root.resizable(False, False)

        self.rms_value = tk.DoubleVar(value=0.0)
        self.status_text = tk.StringVar(value="Initializing models...")
        self.internal_status = "Initializing models..." 
        self.on_close = on_close

        self.last_rms = 0.0

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _build_ui(self):
        """Build the GUI layout."""
        # Title
        title_frame = tk.Frame(self.root)
        title_frame.pack(pady=5)
        self.wakeword_indicator = tk.Label(title_frame, text="●", font=("Arial", 20), fg="grey")
        self.wakeword_indicator.pack(side="left", padx=5)
        title = tk.Label(title_frame, text="🎤 Recording Monitor", font=("Arial", 14, "bold"))
        title.pack(side="left")
        
        # Volume Display
        vol_frame = tk.LabelFrame(self.root, text="Input Level (RMS)", padx=10, pady=5)
        vol_frame.pack(fill="x", padx=10, pady=5)
        
        self.vol_bar = ttk.Progressbar(vol_frame, variable=self.rms_value, maximum=0.1, mode="determinate")
        self.vol_bar.pack(fill="x", pady=2)
        
        self.vol_label = tk.Label(vol_frame, text="0.000", font=("Courier", 11, "bold"))
        self.vol_label.pack()
        
        # Status
        status_frame = tk.LabelFrame(self.root, text="Status", padx=10, pady=5)
        status_frame.pack(fill="x", padx=10, pady=5)
        
        self.status_label = tk.Label(status_frame, textvariable=self.status_text, font=("Arial", 10), wraplength=350)
        self.status_label.pack()
        
        # Toggle Options
        toggle_frame = tk.LabelFrame(self.root, text="Mode", padx=10, pady=5)
        toggle_frame.pack(fill="x", padx=10, pady=5)
        
        self.wakeword_mode_var = tk.BooleanVar(value=True)
        self.wakeword_mode_check = tk.Checkbutton(
            toggle_frame,
            text=f"Wakeword Detection ('{WAKEWORD_DISPLAY}')",
            variable=self.wakeword_mode_var
        )
        self.wakeword_mode_check.pack(anchor="w", pady=2)
        
        self.tts_output_var = tk.BooleanVar(value=True)
        self.tts_output_check = tk.Checkbutton(
            toggle_frame,
            text="Enable TTS Output",
            variable=self.tts_output_var,
        )
        self.tts_output_check.pack(anchor="w", pady=2)
        
        # Control Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", padx=10, pady=5)
        
        self.stop_btn = tk.Button(btn_frame, text="Stop Recording", bg="#ff6b6b", fg="white", command=self._on_close)
        self.stop_btn.pack(fill="x", pady=2)
    
    def update_volume(self, rms: float):
        """Update volume level (RMS) in the GUI."""
        self.rms_value.set(min(rms, 0.1))
        
        # Color bar green when speaking, black when silent
        if rms >= MIN_RMS:
            self.vol_label.config(text=f"{rms:.4f}", fg="green")
        else:
            self.vol_label.config(text=f"{rms:.4f}", fg="black")

    def update_status(self, text: str):
        """Update status message."""
        self.status_text.set(text)
        
    def set_wakeword_state(self, state: str):
        """Change color of wakeword indicator. States: 'waiting', 'listening', 'processing'."""
        colors = {"waiting": "grey", "listening": "green", "processing": "orange"}
        self.wakeword_indicator.config(fg=colors.get(state, "grey"))
    
    def is_wakeword_mode_enabled(self) -> bool:
        """Check if wakeword mode is enabled."""
        return self.wakeword_mode_var.get()

    def is_tts_output_enabled(self) -> bool:
        """Check if TTS output is enabled."""
        return self.tts_output_var.get()

    def process_events(self):
        """Process pending GUI events (MUST be called from main thread only)."""
        try:
            self.root.update()
            return True
        except tk.TclError:
            return False

    def _on_close(self):
        """Cleanup logic when application window is closed."""
        if callable(self.on_close):
            self.on_close()
        self.root.destroy()
        print("\n[Exit] Application closed.")


class LiveSpeechLoop:
    def __init__(self, gui: VolumeMonitorGUI = None) -> None:
        self.gui = gui
        self.stop_event = threading.Event()
        self.audio_lock = threading.Lock()
        self.audio_buffer = RingBuffer(max_samples=RING_BUFFER_SECONDS * SAMPLE_RATE)
        self.command_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
        self.speaking_event = threading.Event()
        self.playback_lock = threading.Lock()
        self.last_rms = 0.0
        self.is_listening_for_command = False
        self.command_audio_chunks: list[np.ndarray] = []
        self.command_samples_needed = 0
        self.wakeword_hits = 0
        self.speech_frame_hits = 0
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.wakeword_model_name = WAKEWORD_MODEL_NAME
        
        print("[Init] Loading Wakeword model...", end=" ", flush=True)
        try:
            download_models = getattr(openwakeword.utils, "download_models", None)
            if callable(download_models):
                download_models()

            if WAKEWORD_MODEL_PATH:
                self.oww_model = WakewordModel(wakeword_model_paths=[WAKEWORD_MODEL_PATH])
                self.wakeword_model_name = list(self.oww_model.models.keys())[0]
            else:
                if WAKEWORD_MODEL_NAME in openwakeword.models:
                    model_path = openwakeword.models[WAKEWORD_MODEL_NAME]["model_path"]
                    self.oww_model = WakewordModel(wakeword_model_paths=[model_path])
                    self.wakeword_model_name = WAKEWORD_MODEL_NAME
                else:
                    self.oww_model = WakewordModel()
                    self.wakeword_model_name = list(self.oww_model.models.keys())[0]

            print("✓")
        except Exception as e:
            print(f"✗\n[Error] Failed to load Wakeword model: {e}")
            raise

        print("[Init] Loading Transcription model...", end=" ", flush=True)
        try:
            self.whisper_transcription = WhisperModel(WHISPER_MODEL_TRANSCRIPTION, device=DEVICE, compute_type=COMPUTE_TYPE)
            print("✓")
        except Exception as e:
            print(f"✗\n[Error] Failed to load Transcription model: {e}")
            raise
        
        print("[Init] Loading Kokoro TTS...", end=" ", flush=True)
        try:
            self.kokoro = KPipeline(lang_code="a") # Force English for TTS
            print("✓")
        except Exception as e:
            print(f"✗\n[Error] Failed to load Kokoro: {e}")
            raise

    def start(self):
        """Starts the audio stream and processing loop."""
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCKSIZE,
            channels=CHANNELS,
            callback=self._callback,
            dtype="float32",
        )
        self.stream.start()
        print("\n[Info] Audio stream started.")
        threading.Thread(target=self._kws_loop, daemon=True).start()
        threading.Thread(target=self._transcription_loop, daemon=True).start()

    def stop(self) -> None:
        """Stop audio processing and release the input stream."""
        self.stop_event.set()
        if hasattr(self, "stream"):
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Audio stream callback for processing incoming audio data."""
        try:
            if self.speaking_event.is_set():
                self.last_rms = 0.0
                return

            # Extract first channel
            chunk = indata[:, 0].astype(np.float32) if indata.ndim == 2 else indata.astype(np.float32)

            with self.audio_lock:
                self.audio_buffer.push(chunk)

            # --- HARD CLIPPER ---
            chunk = np.clip(chunk, -0.99, 0.99)

            # Calculate and display RMS
            self.last_rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
        except Exception as e:
            print(f"[Audio Callback Error] {e}")

    def _transcribe(self, chunk: np.ndarray, model: WhisperModel, language="en") -> str:
        """Transcribe audio chunk using the specified Whisper model."""
        if chunk.size == 0:
            return ""
        try:
            segments, _ = model.transcribe(
                chunk,
                language=language,
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
            return " ".join(seg.text.strip() for seg in segments).strip().lower()
        except Exception as e:
            print(f"[STT Error] {e}")
            return ""

    def _speak(self, text: str) -> None:
        """Convert text to speech using Kokoro TTS."""
        if self.gui and not self.gui.is_tts_output_enabled():
            return
        try:
            with self.playback_lock:
                self.speaking_event.set()
                try:
                    if hasattr(self.kokoro, "tts"):
                        result = self.kokoro.tts(text)
                    elif callable(self.kokoro):
                        result = self.kokoro(text)
                    else:
                        raise AttributeError("KPipeline does not expose tts()")

                    if isinstance(result, tuple) and len(result) == 2:
                        audio, sr = result
                    else:
                        audio, sr = result, KOKORO_RATE

                    sd.play(audio, sr)
                    sd.wait()
                finally:
                    self.speaking_event.clear()
        except Exception as e:
            self.speaking_event.clear()
            print(f"[TTS Error] {e}")

    def _kws_loop(self) -> None:
        """Wakeword detection loop with VAD gating and debouncing."""
        command_listen_duration = 4.0  # seconds
        command_samples = int(command_listen_duration * SAMPLE_RATE)
        continuous_buffer_samples = int(3.0 * SAMPLE_RATE)
        leftover = np.zeros(0, dtype=np.float32)

        while not self.stop_event.is_set():
            try:
                time.sleep(0.01)
                if self.speaking_event.is_set():
                    if self.gui:
                        self.gui.internal_status = "Speaking (TTS)..."
                    continue

                wakeword_mode = self.gui.is_wakeword_mode_enabled() if self.gui else True

                with self.audio_lock:
                    chunk = self.audio_buffer.pop(WAKEWORD_FRAME_SAMPLES)

                if chunk is None:
                    continue

                if not wakeword_mode:
                    if self.gui:
                        self.gui.internal_status = "Continuous Mode - transcribing..."
                    if leftover.size:
                        chunk = np.concatenate((leftover, chunk))
                        leftover = np.zeros(0, dtype=np.float32)

                    if chunk.size < continuous_buffer_samples:
                        leftover = chunk
                        continue

                    chunk_to_process = chunk[:continuous_buffer_samples]
                    leftover = chunk[continuous_buffer_samples:]
                    try:
                        self.command_queue.put_nowait(chunk_to_process)
                    except queue.Full:
                        pass
                    continue

                # Wakeword mode
                if self.gui and not self.is_listening_for_command:
                    self.gui.internal_status = f"Waiting for '{WAKEWORD_DISPLAY}'..."

                audio_int16 = np.clip(chunk, -1.0, 1.0)
                audio_int16 = (audio_int16 * 32767.0).astype(np.int16)

                if not self._is_speech(audio_int16):
                    self.speech_frame_hits = 0
                    if not self.is_listening_for_command:
                        self.wakeword_hits = 0
                    continue

                self.speech_frame_hits += 1
                if self.speech_frame_hits < VAD_SPEECH_FRAMES_REQUIRED:
                    continue

                if self.is_listening_for_command:
                    self.command_audio_chunks.append(chunk)
                    self.command_samples_needed -= chunk.size
                    if self.command_samples_needed <= 0:
                        command_audio = np.concatenate(self.command_audio_chunks).astype(np.float32, copy=False)
                        self.command_audio_chunks = []
                        self.is_listening_for_command = False
                        if self.gui:
                            self.gui.internal_status = "Transcribing command..."
                        try:
                            self.command_queue.put_nowait(command_audio[:command_samples])
                        except queue.Full:
                            pass
                    continue

                prediction = self.oww_model.predict(audio_int16)
                score = float(prediction.get(self.wakeword_model_name, 0.0))

                if score >= WAKEWORD_THRESHOLD:
                    self.wakeword_hits += 1
                else:
                    self.wakeword_hits = 0

                if self.wakeword_hits >= WAKEWORD_DEBOUNCE_FRAMES:
                    print(f"[WakeWord] Detected '{WAKEWORD_DISPLAY}'")
                    self.is_listening_for_command = True
                    self.command_samples_needed = command_samples
                    self.command_audio_chunks = []
                    self.wakeword_hits = 0
                    if self.gui:
                        self.gui.internal_status = "Listening for command..."

            except Exception as e:
                print(f"[KWS Error] {e}")
                if self.gui:
                    self.gui.internal_status = f"Error: {str(e)[:50]}"
                self.is_listening_for_command = False
                self.command_audio_chunks = []
                self.command_samples_needed = 0

    def _transcription_loop(self) -> None:
        """Transcription loop decoupled from KWS and audio capture."""
        while not self.stop_event.is_set():
            try:
                chunk = self.command_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            text = self._transcribe(chunk, self.whisper_transcription, language="de")
            if not text:
                continue

            print(f"[STT] {text}")
            if self.gui:
                self.gui.internal_status = f"Replying: {text[:50]}..."
            self._speak(text)

    def _is_speech(self, audio_int16: np.ndarray) -> bool:
        """Return True if VAD detects speech in the provided audio frame."""
        if audio_int16.size < VAD_FRAME_SAMPLES:
            return False

        frame_bytes = audio_int16.tobytes()
        frame_length_bytes = VAD_FRAME_SAMPLES * 2
        for i in range(0, len(frame_bytes) - frame_length_bytes + 1, frame_length_bytes):
            if self.vad.is_speech(frame_bytes[i:i + frame_length_bytes], SAMPLE_RATE):
                return True
        return False


class RingBuffer:
    def __init__(self, max_samples: int) -> None:
        self.max_samples = max_samples
        self.buffers: collections.deque[np.ndarray] = collections.deque()
        self.size = 0

    def push(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return

        self.buffers.append(samples)
        self.size += samples.size

        while self.size > self.max_samples and self.buffers:
            oldest = self.buffers.popleft()
            self.size -= oldest.size

    def pop(self, n_samples: int) -> np.ndarray | None:
        if self.size < n_samples:
            return None

        chunks: list[np.ndarray] = []
        remaining = n_samples
        while remaining > 0 and self.buffers:
            head = self.buffers[0]
            if head.size <= remaining:
                chunks.append(self.buffers.popleft())
                self.size -= head.size
                remaining -= head.size
            else:
                chunks.append(head[:remaining])
                self.buffers[0] = head[remaining:]
                self.size -= remaining
                remaining = 0

        if not chunks:
            return None

        return np.concatenate(chunks).astype(np.float32, copy=False)


def main():
    root = tk.Tk()
    speech_loop = LiveSpeechLoop(None)

    def _shutdown():
        speech_loop.stop()

    def _handle_sigint(_signal, _frame):
        _shutdown()
        try:
            root.quit()
        except tk.TclError:
            pass

    signal.signal(signal.SIGINT, _handle_sigint)
    app = VolumeMonitorGUI(root, on_close=_shutdown)
    speech_loop.gui = app

    def _gui_updater():
        """This function runs in the main thread to safely update the GUI."""
        if not app.process_events(): # Checks if window was closed
            speech_loop.stop_event.set()
            return

        # Update status text
        app.update_status(app.internal_status)

        # Update wakeword indicator
        if "Waiting" in app.internal_status:
            app.set_wakeword_state("waiting")
        elif "Listening" in app.internal_status:
            app.set_wakeword_state("listening")
        elif "Transcribing" in app.internal_status or "Replying" in app.internal_status:
            app.set_wakeword_state("processing")

        # Update volume
        app.update_volume(speech_loop.last_rms)

        root.after(100, _gui_updater) # Schedule next update

    # Start the audio stream and processor loop
    speech_loop.start()
    
    # Start the GUI updater loop
    root.after(100, _gui_updater)
    root.mainloop()
    _shutdown()


if __name__ == "__main__":
    main()