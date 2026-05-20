#!/usr/bin/env python3
"""
wakeword_detection.py — Minimal openWakeWord microphone test.
"""

from __future__ import annotations

import os
import queue
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("ORT_DISABLE_CUDA", "1")

try:
    import numpy as np
    import sounddevice as sd
except ModuleNotFoundError as exc:
    sys.exit(f"\nMissing dependency: {exc}\nInstall: pip install sounddevice numpy\n")

try:
    import openwakeword
    from openwakeword.model import Model as WakewordModel
except ModuleNotFoundError as exc:
    sys.exit(f"\nMissing dependency: {exc}\nInstall: pip install openwakeword\n")

TARGET_SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 160
FRAME_SAMPLES = int(TARGET_SAMPLE_RATE * FRAME_MS / 1000)
DEVICE_INDEX = None  # Set to an int to select a specific input device

WAKEWORD_MODEL_NAME = "alexa"
WAKEWORD_MODEL_PATH = ".models/alexa_v0.1.onnx"
WAKEWORD_THRESHOLD = 0.5
WAKEWORD_DEBOUNCE_FRAMES = 1
DEBUG_SCORES = True

# openwakeword model options (best effort depending on platform)
ENABLE_SPEEX_NS = False
VAD_THRESHOLD = 0.0


class AudioStream:
    def __init__(self) -> None:
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self.last_rms = 0.0
        self.last_status_time = 0.0

    def callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            now = time.time()
            if now - self.last_status_time >= 1.0:
                print(f"[Audio] {status}")
                self.last_status_time = now
        try:
            chunk = indata[:, 0].astype(np.float32, copy=False)
            if chunk.size:
                self.last_rms = float(np.sqrt(np.mean(chunk * chunk)))
            try:
                self.queue.put_nowait(chunk)
            except queue.Full:
                try:
                    _ = self.queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.queue.put_nowait(chunk)
                except queue.Full:
                    pass
        except queue.Full:
            pass


def load_wakeword_model() -> tuple[WakewordModel, str]:
    model_kwargs = {
        "enable_speex_noise_suppression": ENABLE_SPEEX_NS,
        "vad_threshold": VAD_THRESHOLD,
    }

    model_path = WAKEWORD_MODEL_PATH
    if not model_path and WAKEWORD_MODEL_NAME in openwakeword.models:
        model_path = openwakeword.models[WAKEWORD_MODEL_NAME]["model_path"]

    if model_path:
        if not os.path.exists(model_path):
            download_models = getattr(openwakeword.utils, "download_models", None)
            if callable(download_models):
                download_models()
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Wakeword model not found at {model_path}. Reinstall openwakeword."
            )
        model = WakewordModel(wakeword_model_paths=[model_path], **model_kwargs)
        model_name = list(model.models.keys())[0]
        return model, model_name

    model = WakewordModel(**model_kwargs)
    model_name = list(model.models.keys())[0]
    return model, model_name


def main() -> None:
    print("[Init] Loading wakeword model...", end=" ", flush=True)
    model, model_name = load_wakeword_model()
    print("✓")

    audio = AudioStream()
    hits = 0

    device_list = []
    try:
        print("[Info] Available input devices:")
        for idx, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0:
                device_list.append((idx, dev["name"]))
                print(f"  [{idx}] {dev['name']}")
    except Exception as exc:
        print(f"[Warn] Could not list devices: {exc}")

    selected_device = DEVICE_INDEX
    selected_sample_rate = TARGET_SAMPLE_RATE
    if selected_device is None and device_list:
        while True:
            choice = input("Select input device index (blank = default): ").strip()
            if choice == "":
                selected_device = None
                break
            if choice.isdigit():
                selected_device = int(choice)
                break
            print("Invalid selection. Enter a number or press Enter for default.")

    if selected_device is not None:
        try:
            device_info = sd.query_devices(selected_device)
            default_rate = device_info.get("default_samplerate")
            if default_rate:
                selected_sample_rate = int(default_rate)
        except Exception as exc:
            print(f"[Warn] Could not read device sample rate: {exc}")

    def _resample_audio(chunk: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
        if in_rate == out_rate or chunk.size == 0:
            return chunk
        new_length = int(round(chunk.size * out_rate / in_rate))
        if new_length <= 0:
            return np.zeros(0, dtype=np.float32)
        x_old = np.arange(chunk.size)
        x_new = np.linspace(0, chunk.size - 1, num=new_length)
        return np.interp(x_new, x_old, chunk).astype(np.float32, copy=False)

    def _run_stream(sample_rate: int) -> None:
        last_print = 0.0
        pending = np.zeros(0, dtype=np.float32)
        warned_key_mismatch = False
        with sd.InputStream(
            samplerate=sample_rate,
            channels=CHANNELS,
            blocksize=0,
            device=selected_device,
            callback=audio.callback,
            latency="high",
            dtype="float32",
        ):
            print(f"[Info] Listening for wakeword at {sample_rate} Hz...")
            while True:
                chunk = audio.queue.get()
                chunk = _resample_audio(chunk, sample_rate, TARGET_SAMPLE_RATE)
                if chunk.size == 0:
                    continue

                if pending.size:
                    pending = np.concatenate((pending, chunk)).astype(np.float32, copy=False)
                else:
                    pending = chunk

                while pending.size >= FRAME_SAMPLES:
                    frame = pending[:FRAME_SAMPLES]
                    pending = pending[FRAME_SAMPLES:]

                    audio_int16 = np.clip(frame, -1.0, 1.0)
                    audio_int16 = (audio_int16 * 32767.0).astype(np.int16)

                    prediction = model.predict(audio_int16)
                    if model_name not in prediction and prediction and not warned_key_mismatch:
                        print(f"[Warn] Wakeword key not found. Keys: {list(prediction.keys())}")
                        warned_key_mismatch = True
                    score = float(prediction.get(model_name, 0.0))

                    if score >= WAKEWORD_THRESHOLD:
                        hits += 1
                    else:
                        hits = 0

                    now = time.time()
                    if hits >= WAKEWORD_DEBOUNCE_FRAMES:
                        print(f"[WakeWord] Detected '{WAKEWORD_MODEL_NAME}' (score={score:.3f}, rms={audio.last_rms:.4f})", flush=True)
                        hits = 0
                    elif DEBUG_SCORES and now - last_print >= 0.2:
                        print(f"[Score] {score:.3f} | rms={audio.last_rms:.4f}")
                        last_print = now

    try:
        _run_stream(selected_sample_rate)
    except sd.PortAudioError as exc:
        print(f"[Warn] Failed to open stream at {selected_sample_rate} Hz: {exc}")
        print("[Info] Retrying with 48000 Hz...")
        _run_stream(48000)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Exit] Stopped.")
