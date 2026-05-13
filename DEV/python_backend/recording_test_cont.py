#!/usr/bin/env python3
"""
recording_test_cont.py — Continuous Lossless STT Pipeline with Smart Auto Gain.
Uses Queues, VAD, and 3-Stage AGC (Attack, Speech-Leveling, Silence-Decay).
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
CHANNELS = 2
BLOCKSIZE = 1600  # 100 ms chunks
MIN_RMS = 0.010   # Volume threshold to trigger speech
PRE_ROLL_CHUNKS = 5   # Keep 0.5s of audio before speech starts
POST_ROLL_CHUNKS = 10 # Wait 1.0s of silence before finalizing sentence

WHISPER_MODEL_WAKEWORD = "tiny.en"
WHISPER_MODEL_TRANSCRIPTION = "large-v3-turbo"
COMPUTE_TYPE = "int8"
DEVICE = "cpu"
LANGUAGE = "de"  
KOKORO_LANG = "de"
KOKORO_RATE = 24000

GAIN = 1.0  # Initial gain multiplier
WAKEWORD = "computer"


class VolumeMonitorGUI:
    def __init__(self, root_window=None):
        self.root = root_window or tk.Tk()
        self.root.title("Recording Monitor - Live STT/TTS")
        self.root.geometry("400x440")
        self.root.resizable(False, False)
        
        self.gain_value = tk.DoubleVar(value=GAIN)
        self.rms_value = tk.DoubleVar(value=0.0)
        self.status_text = tk.StringVar(value="Initializing models...")
        self.internal_status = "Initializing models..." 

        # Internal state for the audio thread
        self.internal_gain = 1.0
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
        
        # Gain Control
        gain_frame = tk.LabelFrame(self.root, text="Gain Multiplier", padx=10, pady=5)
        gain_frame.pack(fill="x", padx=10, pady=5)
        
        self.gain_slider = ttk.Scale(
            gain_frame, 
            from_=0.1, 
            to=5.0, 
            variable=self.gain_value, 
            orient="horizontal"
        )
        self.gain_slider.pack(fill="x", pady=2)
        
        self.gain_label = tk.Label(gain_frame, text="1.0x", font=("Courier", 11, "bold"))
        self.gain_label.pack()
        
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
            text="Wakeword Detection ('computer')", 
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
    
    def get_gain(self) -> float:
        """Retrieve the current gain multiplier value."""
        return self.gain_value.get()
    
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
        self.root.destroy()
        print("\n[Exit] Application closed.")
        sys.exit(0)


class LiveSpeechLoop:
    def __init__(self, gui: VolumeMonitorGUI = None) -> None:
        self.gui = gui
        self.stop_event = threading.Event()
        self.audio_lock = threading.Lock()
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.total_samples = 0
        self.last_processed_total_samples = 0
        self.speaking_event = threading.Event()
        self.playback_lock = threading.Lock()
        self.last_rms = 0.0
        self.is_listening_for_command = False
        self.command_listen_start_time = 0
        
        print("[Init] Loading Wakeword model...", end=" ", flush=True)
        try:
            self.whisper_wakeword = WhisperModel(WHISPER_MODEL_WAKEWORD, device=DEVICE, compute_type=COMPUTE_TYPE)
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
        threading.Thread(target=self._processor_loop, daemon=True).start()

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Audio stream callback for processing incoming audio data."""
        try:
            # Apply gain
            gain = self.gui.internal_gain if self.gui else 1.0
            indata *= gain

            # Extract first channel
            chunk = indata[:, 0].astype(np.float32) if indata.ndim == 2 else indata.astype(np.float32)

            with self.audio_lock:
                # Append new audio data to buffer
                self.audio_buffer = np.concatenate((self.audio_buffer, chunk))

            # --- HARD CLIPPER ---
            chunk = np.clip(chunk, -0.99, 0.99)

            # Calculate and display RMS
            self.last_rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
        except Exception as e:
            print(f"[Audio Callback Error] {e}")

    def _transcribe(self, chunk: np.ndarray, model: WhisperModel, language="en") -> str:
        """Transcribe audio chunk using the specified Whisper model."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)
        try:
            sf.write(str(wav_path), chunk, SAMPLE_RATE)
            segments, _ = model.transcribe(
                str(wav_path),
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
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass

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
                        if isinstance(result, tuple) and len(result) == 2:
                            audio, sr = result
                        else:
                            audio, sr = result, KOKORO_RATE
                        sd.play(audio, sr)
                        sd.wait()
                    else:
                        raise AttributeError("KPipeline does not expose tts()")
                finally:
                    self.speaking_event.clear()
        except Exception as e:
            self.speaking_event.clear()
            print(f"[TTS Error] {e}")

    def _processor_loop(self) -> None:
        """Main processing loop for wakeword detection and command transcription."""
        wakeword_window_samples = int(1.5 * SAMPLE_RATE)
        command_listen_duration = 4.0  # seconds
        continuous_buffer_samples = int(3.0 * SAMPLE_RATE) # Process 3-second chunks in continuous mode

        while not self.stop_event.is_set():
            try:
                time.sleep(0.1) # Loop faster for responsiveness
                
                if self.speaking_event.is_set():
                    self.gui.internal_status = "Speaking (TTS)..."
                    continue

                with self.audio_lock:
                    buffer = self.audio_buffer.copy()

                wakeword_mode = self.gui.is_wakeword_mode_enabled() if self.gui else True

                if wakeword_mode:
                    # --- Wakeword Detection Mode ---
                    if not self.is_listening_for_command:
                        if self.gui:
                            self.gui.internal_status = f"Waiting for '{WAKEWORD}'..."

                        if buffer.size < wakeword_window_samples:
                            continue
                        
                        window = buffer[-wakeword_window_samples:]
                        text = self._transcribe(window, self.whisper_wakeword)

                        if WAKEWORD in text:
                            print(f"[WakeWord] Detected '{WAKEWORD}'")
                            self.is_listening_for_command = True
                            self.command_listen_start_time = time.time()
                            if self.gui:
                                self.gui.internal_status = "Listening for command..."
                            # Clear buffer to only get command
                            with self.audio_lock:
                                self.audio_buffer = np.zeros(0, dtype=np.float32)

                    else: # We are listening for a command
                        elapsed = time.time() - self.command_listen_start_time
                        if self.gui:
                            self.gui.internal_status = f"Listening... {command_listen_duration - elapsed:.1f}s left"

                        if elapsed > command_listen_duration:
                            if self.gui:
                                self.gui.internal_status = "Transcribing command..."
                            
                            command_audio = buffer.copy()
                            text = self._transcribe(command_audio, self.whisper_transcription, language="de")
                            
                            self.is_listening_for_command = False # Reset state
                            with self.audio_lock:
                                self.audio_buffer = np.zeros(0, dtype=np.float32)

                            if not text:
                                print("[STT] No command recognized.")
                                continue

                            print(f"[STT] {text}")
                            
                            if self.gui:
                                self.gui.internal_status = f"Replying: {text[:50]}..."
                            
                            self._speak(text)
                
                else:
                    # --- Continuous Sampling Mode ---
                    if self.gui:
                        self.gui.internal_status = "Continuous Mode - transcribing..."
                    
                    if buffer.size < continuous_buffer_samples:
                        continue
                    
                    # Process the oldest chunk of audio
                    chunk_to_process = buffer[:continuous_buffer_samples]
                    
                    # Trim the buffer
                    with self.audio_lock:
                        self.audio_buffer = self.audio_buffer[continuous_buffer_samples:]

                    text = self._transcribe(chunk_to_process, self.whisper_transcription, language="de")

                    if not text:
                        continue

                    print(f"[STT] {text}")
                    
                    if self.gui:
                        self.gui.internal_status = f"Replying: {text[:50]}..."
                    
                    self._speak(text)

            except Exception as e:
                print(f"[Processor Error] {e}")
                if self.gui:
                    self.gui.internal_status = f"Error: {str(e)[:50]}"
                    self.is_listening_for_command = False # Reset on error


def main():
    root = tk.Tk()
    app = VolumeMonitorGUI(root)
    speech_loop = LiveSpeechLoop(app)

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

        # Update volume and gain
        app.update_volume(speech_loop.last_rms)
        app.internal_gain = app.get_gain() # Safely get gain from slider
        app.gain_label.config(text=f"{app.internal_gain:.1f}x")

        root.after(100, _gui_updater) # Schedule next update

    # Start the audio stream and processor loop
    speech_loop.start()
    
    # Start the GUI updater loop
    root.after(100, _gui_updater)
    root.mainloop()


if __name__ == "__main__":
    main()