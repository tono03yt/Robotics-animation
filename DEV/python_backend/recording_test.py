import tkinter as tk
from tkinter import ttk
import threading
import queue
import time
import sys
import subprocess
import numpy as np
import sounddevice as sd
import torch
try:
    import torchaudio
except ImportError:
    print("Installing torchaudio...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchaudio"])
    import torchaudio
import onnxruntime as ort

# OpenWakeWord
try:
    import openwakeword
    from openwakeword.model import Model
except ImportError:
    print("pip install openwakeword")
    sys.exit(1)

# Faster Whisper
try:
    from faster_whisper import WhisperModel
except ImportError:
    print("pip install faster-whisper")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # 32ms chunks (Supported by Silero VAD at 16kHz)
VAD_THRESHOLD = 0.5

class AudioPipeline(threading.Thread):
    def __init__(self, gui, use_wakeword):
        super().__init__()
        self.gui = gui
        self.use_wakeword = use_wakeword
        self.audio_queue = queue.Queue()
        self.running = True
        self.state = "WAITING_FOR_WAKEWORD" if use_wakeword else "SAMPLING"
        
        # Load OpenWakeWord Models
        self.gui.log("Loading WakeWord Model (alexa)...")
        # Find hardware accelerators for AMD (ROCm provides CUDA-like interface in PyTorch/ORT usually)
        providers = ['ROCMExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        available_providers = ort.get_available_providers()
        selected_provider = [p for p in providers if p in available_providers][0]
        self.gui.log(f"ONNX Provider used: {selected_provider}")
        
        # Select the Alexa pretrained model
        import openwakeword
        alexa_model = [p for p in openwakeword.get_pretrained_model_paths() if "alexa" in p.lower()]
        if not alexa_model:
            self.gui.log("Alexa model not found in openwakeword!")
        self.oww_model = Model(wakeword_model_paths=alexa_model)

        # Load Silero VAD
        self.gui.log("Loading Silero VAD...")
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                               model='silero_vad',
                                               force_reload=False, onnx=True)
        self.get_speech_timestamps = utils[0]

        # Load Faster Whisper
        self.gui.log("Loading Faster-Whisper (German optimized)...")
        # On AMD ROCm, FasterWhisper uses cuda device string but ROCm backend under the hood PyTorch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gui.log(f"Whisper Device: {device}")
        self.stt_model = WhisperModel("large-v3-turbo", device=device, compute_type="float16" if device=="cuda" else "int8")
        
        self.gui.log("Pipeline Ready!")
        
    def run(self):
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=CHUNK_SIZE) as stream:
            audio_buffer = []
            silence_counter = 0
            
            while self.running:
                chunk, overflow = stream.read(CHUNK_SIZE)
                if overflow:
                    continue
                
                audio_data = np.frombuffer(chunk, dtype=np.int16)
                
                # Volume Indicator Update
                vol = np.abs(audio_data).mean()
                self.gui.update_mic_level(vol)
                
                # Check VAD
                # Convert to float32 for Silero
                audio_float = audio_data.astype(np.float32) / 32768.0
                vad_prob = self.vad_model(torch.from_numpy(audio_float), SAMPLE_RATE).item()
                is_speech = vad_prob > VAD_THRESHOLD
                self.gui.update_speech_indicator(is_speech)
                
                if self.state == "WAITING_FOR_WAKEWORD":
                    prediction = self.oww_model.predict(audio_data)
                    max_score = prediction.get('alexa', 0.0)
                    if max_score > 0.5:
                        self.gui.log("WAKEWORD DETECTED! Starting recording...")
                        self.gui.update_wakeword_indicator(True)
                        self.state = "SAMPLING"
                        audio_buffer = []
                        time.sleep(0.5)
                        self.gui.update_wakeword_indicator(False)
                        
                elif self.state == "SAMPLING":
                    audio_buffer.append(audio_data)
                    
                    if is_speech:
                        silence_counter = 0
                    else:
                        silence_counter += 1
                        
                    # If silence for ~1.5 seconds, process sentence
                    if silence_counter > int(1.5 / (CHUNK_SIZE / SAMPLE_RATE)):
                        if len(audio_buffer) > (silence_counter + 5): # Ensure we actually recorded something
                            self.gui.log("Processing Speech...")
                            audio_concat = np.concatenate(audio_buffer).astype(np.float32) / 32768.0
                            threading.Thread(target=self.transcribe, args=(audio_concat,)).start()
                            
                        audio_buffer = []
                        silence_counter = 0
                        if self.use_wakeword:
                            self.state = "WAITING_FOR_WAKEWORD"

    def transcribe(self, audio_data):
        segments, info = self.stt_model.transcribe(audio_data, language="de", beam_size=1, condition_on_previous_text=False)
        sys.stdout.write("\n[STT]: ")
        for segment in segments:
            # Print instantly as segment arrives
            sys.stdout.write(segment.text)
            sys.stdout.flush()
        sys.stdout.write("\n")
        self.gui.log("STT Done.")
        
    def stop(self):
        self.running = False


class AppGUI:
    def __init__(self, root):
        self.root = root
        root.title("Voice Pipeline Start")
        
        tk.Label(root, text="Voice Pipeline Settings", font=("Helvetica", 16, "bold")).pack(pady=10)
        
        self.use_wakeword = tk.BooleanVar(value=True)
        tk.Checkbutton(root, text="Enable WakeWord (Alexa)", variable=self.use_wakeword).pack(pady=5)
        
        tk.Button(root, text="Start Pipeline", command=self.start_pipeline, bg="green", fg="white").pack(pady=10)
        
        self.logs = tk.Text(root, height=10, width=50)
        self.logs.pack(pady=10)
        
        frame = tk.Frame(root)
        frame.pack(pady=5)
        
        tk.Label(frame, text="Mic Level:").grid(row=0, column=0)
        self.mic_bar = ttk.Progressbar(frame, length=200, mode='determinate')
        self.mic_bar.grid(row=0, column=1)
        
        tk.Label(frame, text="Speech:").grid(row=1, column=0)
        self.speech_ind = tk.Label(frame, text="OFF", bg="grey", fg="white")
        self.speech_ind.grid(row=1, column=1, sticky="w", pady=5)
        
        tk.Label(frame, text="Wakeword:").grid(row=2, column=0)
        self.ww_ind = tk.Label(frame, text="NO", bg="grey", fg="white")
        self.ww_ind.grid(row=2, column=1, sticky="w", pady=5)
        
        self.pipeline = None

    def log(self, msg):
        self.logs.insert(tk.END, msg + "\n")
        self.logs.see(tk.END)
        print(f"[GUI Log] {msg}")

    def update_mic_level(self, level):
        # normalize to 0-100 roughly
        val = min(100, level / 50) 
        self.mic_bar['value'] = val
        
    def update_speech_indicator(self, active):
        if active:
            self.speech_ind.config(text="ON", bg="green")
        else:
            self.speech_ind.config(text="OFF", bg="grey")

    def update_wakeword_indicator(self, active):
        if active:
            self.ww_ind.config(text="YES", bg="orange")
        else:
            self.ww_ind.config(text="NO", bg="grey")

    def start_pipeline(self):
        if self.pipeline is None:
            self.pipeline = AudioPipeline(self, self.use_wakeword.get())
            self.pipeline.start()

    def on_close(self):
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline.join()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AppGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
