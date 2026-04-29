#!/usr/bin/env python3
"""Interactive serial protocol test client for the face-tracking / Arduino link.

This tool is meant to test the same newline-delimited protocol used by
face_tracking_test.py without starting the full vision pipeline.

Typical use:
- connect to Arduino Nano over USB serial
- send SERVO commands manually
- send PLAY / REC commands for audio protocol testing
- print any incoming AUDIO / STAT / debug messages from Arduino
"""

from __future__ import annotations

import argparse
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


def detect_serial_ports() -> List[str]:
    """Return a list of serial port device paths available on this machine."""
    ports: List[str] = []

    if list_ports is not None:
        for port in list_ports.comports():
            ports.append(port.device)

    if ports:
        return sorted(set(ports))

    # Fallback scan for Linux/macOS-style tty devices when pyserial tools are unavailable.
    for pattern in ("ttyUSB*", "ttyACM*", "ttyS*", "cu.*"):
        for path in Path("/dev").glob(pattern):
            ports.append(str(path))

    return sorted(set(ports))


class SerialProtocolClient:
    """Interactive helper for sending and receiving serial protocol messages."""

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.2) -> None:
        if serial is None:
            raise RuntimeError("pyserial is not installed. Install with: pip install pyserial")

        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[object] = None
        self.running = False
        self.reader_thread: Optional[threading.Thread] = None
        self.incoming: queue.Queue[str] = queue.Queue()

    def connect(self) -> bool:
        """Open the serial port."""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(0.5)  # Give Arduino Nano time to reset
            self.running = True
            self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.reader_thread.start()
            print(f"[connected] {self.port} @ {self.baudrate}")
            return True
        except Exception as exc:
            print(f"[error] Could not open {self.port}: {exc}")
            self.ser = None
            self.running = False
            return False

    def disconnect(self) -> None:
        """Close the serial port and stop background reader."""
        self.running = False
        if self.reader_thread is not None:
            self.reader_thread.join(timeout=1.0)
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None
        print("[disconnected]")

    def _reader_loop(self) -> None:
        """Print incoming lines from Arduino as they arrive."""
        buffer = b""
        while self.running and self.ser is not None:
            try:
                waiting = getattr(self.ser, "in_waiting", 0)
                if waiting:
                    chunk = self.ser.read(waiting)
                    if chunk:
                        buffer += chunk
                        while b"\n" in buffer:
                            raw, buffer = buffer.split(b"\n", 1)
                            line = raw.decode("utf-8", errors="replace").strip()
                            if line:
                                self.incoming.put(line)
                                print(f"[rx] {line}")
                else:
                    time.sleep(0.02)
            except Exception as exc:
                print(f"[reader-error] {exc}")
                time.sleep(0.1)

    def send_line(self, line: str) -> bool:
        """Send one newline-terminated message."""
        if self.ser is None:
            return False
        try:
            payload = (line.rstrip("\r\n") + "\n").encode("utf-8")
            self.ser.write(payload)
            print(f"[tx] {line.rstrip()}")
            return True
        except Exception as exc:
            print(f"[error] Send failed: {exc}")
            return False

    def send_servo(self, x_error: float, confidence: float = 1.0) -> bool:
        return self.send_line(f"SERVO,{x_error:.4f},{confidence:.2f}")

    def send_play(self, duration_ms: int, sample_rate: int, hex_audio_data: str) -> bool:
        return self.send_line(f"PLAY,{duration_ms},{sample_rate},{hex_audio_data}")

    def send_rec(self, state: str) -> bool:
        return self.send_line(f"REC,{state}")

    def send_raw(self, line: str) -> bool:
        return self.send_line(line)

    def demo_sequence(self) -> None:
        """Send a small set of protocol messages for quick testing."""
        self.send_servo(0.0, 1.0)
        time.sleep(0.2)
        self.send_servo(-0.25, 0.92)
        time.sleep(0.2)
        self.send_servo(0.35, 0.88)
        time.sleep(0.2)
        self.send_rec("START")
        time.sleep(0.2)
        self.send_rec("STOP")


def print_help() -> None:
    print(
        """
Commands:
  help
      Show this help.

  servo <x_error> [confidence]
      Send a SERVO command, e.g.:
      servo -0.15 0.93

  play <duration_ms> <sample_rate> <hex_audio_data>
      Send a PLAY command, e.g.:
      play 500 16000 ffc0ffc0ffc0

  rec start|stop
      Send REC,START or REC,STOP

  raw <message>
      Send any newline-delimited message as-is.

  demo
      Send a short test sequence of SERVO/REC commands.

  ports
      List available serial ports.

  quit / exit
      Close the connection and exit.
""".strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive serial test client for face-tracking protocol")
    parser.add_argument("--port", type=str, default=None, help="Serial port path (e.g. /dev/ttyUSB0)")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports and exit")
    args = parser.parse_args()

    if args.list_ports:
        ports = detect_serial_ports()
        if not ports:
            print("No serial ports found.")
        else:
            print("Available serial ports:")
            for port in ports:
                print(f"  - {port}")
        return

    port = args.port
    if port is None:
        ports = detect_serial_ports()
        if not ports:
            print("No serial ports found. Use --port if you already know the device path.")
            return

        print("Available serial ports:")
        for i, candidate in enumerate(ports, start=1):
            print(f"  [{i}] {candidate}")

        choice = input("Select port number (empty to cancel): ").strip()
        if not choice:
            return
        if not choice.isdigit() or not (1 <= int(choice) <= len(ports)):
            print("Invalid selection.")
            return
        port = ports[int(choice) - 1]

    client = SerialProtocolClient(port=port, baudrate=args.baudrate)
    if not client.connect():
        return

    print_help()

    try:
        while True:
            try:
                line = input("io-test> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not line:
                continue

            command, *rest = line.split(maxsplit=1)
            command = command.lower()
            payload = rest[0] if rest else ""

            if command in {"quit", "exit", "q"}:
                break
            if command == "help":
                print_help()
                continue
            if command == "ports":
                ports = detect_serial_ports()
                if not ports:
                    print("No serial ports found.")
                else:
                    for candidate in ports:
                        print(f"  - {candidate}")
                continue
            if command == "demo":
                client.demo_sequence()
                continue
            if command == "servo":
                parts = payload.split()
                if not parts:
                    print("Usage: servo <x_error> [confidence]")
                    continue
                x_error = float(parts[0])
                confidence = float(parts[1]) if len(parts) > 1 else 1.0
                client.send_servo(x_error, confidence)
                continue
            if command == "play":
                parts = payload.split(maxsplit=2)
                if len(parts) < 3:
                    print("Usage: play <duration_ms> <sample_rate> <hex_audio_data>")
                    continue
                duration_ms = int(parts[0])
                sample_rate = int(parts[1])
                hex_audio_data = parts[2]
                if not re.fullmatch(r"[0-9A-Fa-f]*", hex_audio_data):
                    print("hex_audio_data must contain only hex characters")
                    continue
                client.send_play(duration_ms, sample_rate, hex_audio_data)
                continue
            if command == "rec":
                state = payload.strip().upper()
                if state not in {"START", "STOP"}:
                    print("Usage: rec start|stop")
                    continue
                client.send_rec(state)
                continue
            if command == "raw":
                if not payload:
                    print("Usage: raw <message>")
                    continue
                client.send_raw(payload)
                continue

            print("Unknown command. Type 'help'.")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
