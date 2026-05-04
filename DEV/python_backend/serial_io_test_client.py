#!/usr/bin/env python3
"""Serial packet monitor for the face-tracking backend.

Modes:
- real serial port: attach to Arduino or a backend connected to a tty
- bridge mode: create an empty pseudo-TTY for the backend to connect to,
  while this script snoops the master side and prints packets
"""

from __future__ import annotations

import argparse
import sys
import os
import pty
import re
import errno
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


POS_RE = re.compile(r"^POS,(-?[\d.]+),(-?[\d.]+),([\d.]+)$")
STAT_RE = re.compile(r"^STAT,(\d+),(\d+)$")
DEBUG_RE = re.compile(r"^\[")


@dataclass
class Packet:
    valid: bool
    kind: str
    fields: Dict[str, object]
    raw: str


def detect_serial_ports() -> List[str]:
    ports: List[str] = []
    if list_ports is not None:
        ports.extend(port.device for port in list_ports.comports())
    if ports:
        return sorted(set(ports))
    for pattern in ("ttyUSB*", "ttyACM*", "ttyS*"):
        ports.extend(str(path) for path in Path("/dev").glob(pattern))
    return sorted(set(ports))


def detect_virtual_ttys() -> List[str]:
    ports: List[str] = []
    for path in Path("/dev/pts").glob("*"):
        if path.name.isdigit():
            ports.append(str(path))
    return sorted(set(ports))


def parse_packet(line: str) -> Packet:
    line = line.strip()
    if not line:
        return Packet(False, "EMPTY", {"error": "empty packet"}, line)
    if DEBUG_RE.match(line):
        return Packet(True, "DEBUG", {"text": line}, line)

    match = POS_RE.match(line)
    if match:
        return Packet(
            True,
            "POS",
            {"x_error": float(match.group(1)), "y_error": float(match.group(2)), "confidence": float(match.group(3))},
            line,
        )

    match = STAT_RE.match(line)
    if match:
        return Packet(True, "STAT", {"servo_us": int(match.group(1)), "millis": int(match.group(2))}, line)

    if "," in line:
        prefix = line.split(",", 1)[0] or "UNKNOWN"
        return Packet(False, prefix, {"error": "unrecognized packet"}, line)
    return Packet(False, "MALFORMED", {"error": "malformed packet"}, line)


class SerialMonitor:
    def __init__(self, port: Optional[str], baudrate: int = 115200) -> None:
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.master_fd: Optional[int] = None
        self.bridge_slave_path: Optional[str] = None
        self.running = False
        self.rx_total = 0
        self.rx_valid = 0
        self.rx_invalid = 0
        self.by_kind: Dict[str, int] = {}

    def create_bridge(self) -> str:
        master_fd, slave_fd = pty.openpty()
        self.master_fd = master_fd
        self.bridge_slave_path = os.ttyname(slave_fd)
        os.close(slave_fd)
        os.set_blocking(master_fd, False)
        return self.bridge_slave_path

    def connect(self) -> bool:
        if self.port is None:
            raise RuntimeError("No serial port specified")
        if serial is None:
            raise RuntimeError("pyserial is not installed. Install with: pip install pyserial")
        self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
        return True

    def start(self) -> None:
        self.running = True
        if self.master_fd is not None:
            self._bridge_loop()
        else:
            self._serial_loop()

    def _record(self, packet: Packet) -> None:
        self.rx_total += 1
        self.by_kind[packet.kind] = self.by_kind.get(packet.kind, 0) + 1
        if packet.valid:
            self.rx_valid += 1
        else:
            self.rx_invalid += 1

    def _print_packet(self, packet: Packet) -> None:
        status = "✓" if packet.valid else "✗"
        if packet.kind == "POS":
            print(
                f"[rx {status}] POS     | x={packet.fields['x_error']:+.4f} "
                f"y={packet.fields['y_error']:+.4f} conf={packet.fields['confidence']:.2f}"
            )
        elif packet.kind == "STAT":
            print(f"[rx {status}] STAT    | servo_us={packet.fields['servo_us']} millis={packet.fields['millis']}")
        elif packet.kind == "DEBUG":
            print(f"[rx {status}] DEBUG   | {packet.fields['text']}")
        else:
            print(f"[rx {status}] {packet.kind:<7} | {packet.raw}")

    def _bridge_loop(self) -> None:
        assert self.master_fd is not None
        buffer = b""
        print(f"[bridge] backend should connect to: {self.bridge_slave_path}")
        print("[bridge] monitoring master side; press Ctrl+C to stop")
        try:
            while self.running:
                try:
                    chunk = os.read(self.master_fd, 4096)
                    if not chunk:
                        time.sleep(0.02)
                        continue
                    buffer += chunk
                    while b"\n" in buffer:
                        raw, buffer = buffer.split(b"\n", 1)
                        line = raw.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        packet = parse_packet(line)
                        self._record(packet)
                        self._print_packet(packet)
                except BlockingIOError:
                    time.sleep(0.02)
                except OSError as exc:
                    if exc.errno in {errno.EIO, errno.ENXIO}:
                        # The slave side is not yet opened by the backend.
                        time.sleep(0.05)
                        continue
                    raise
        finally:
            self.print_summary()

    def _serial_loop(self) -> None:
        assert self.serial is not None
        buffer = b""
        print(f"[connected] {self.port} @ {self.baudrate}")
        try:
            while self.running:
                waiting = getattr(self.serial, "in_waiting", 0)
                if waiting:
                    chunk = self.serial.read(waiting)
                    if chunk:
                        buffer += chunk
                        while b"\n" in buffer:
                            raw, buffer = buffer.split(b"\n", 1)
                            line = raw.decode("utf-8", errors="replace").strip()
                            if not line:
                                continue
                            packet = parse_packet(line)
                            self._record(packet)
                            self._print_packet(packet)
                else:
                    time.sleep(0.02)
        finally:
            self.print_summary()

    def print_summary(self) -> None:
        print("\n--- monitor summary ---")
        print(f"RX total: {self.rx_total}")
        print(f"Valid:    {self.rx_valid}")
        print(f"Invalid:  {self.rx_invalid}")
        for kind, count in sorted(self.by_kind.items()):
            print(f"  {kind}: {count}")


@dataclass
class StartupConfig:
    mode: str  # "bridge" or "serial"
    port: Optional[str]
    baudrate: int


def _ask_choice(prompt: str, valid: set[str]) -> str:
    while True:
        value = input(prompt).strip().lower()
        if value in valid:
            return value
        print(f"Invalid choice. Options: {', '.join(sorted(valid))}")


def choose_startup_configuration() -> Optional[StartupConfig]:
    print("\nSerial monitor startup")
    print("======================")
    print("1) Real serial device")
    print("2) Internal bridge (no Arduino needed)")
    print("3) List detected interfaces")
    print("q) Quit")

    choice = _ask_choice("Select mode [1/2/3/q]: ", {"1", "2", "3", "q"})
    if choice == "q":
        return None

    if choice == "3":
        real_ports = detect_serial_ports()
        pts_ports = detect_virtual_ttys()
        if real_ports:
            print("Detected real serial ports:")
            for port in real_ports:
                print(f"  - {port}")
        else:
            print("No real serial ports detected.")
        if pts_ports:
            print("Detected virtual tty interfaces:")
            for port in pts_ports:
                print(f"  - {port}")
        else:
            print("No virtual tty interfaces detected yet.")
        print()
        return choose_startup_configuration()

    baud_text = input("Baudrate [115200]: ").strip()
    baudrate = int(baud_text) if baud_text.isdigit() else 115200

    if choice == "2":
        return StartupConfig(mode="bridge", port=None, baudrate=baudrate)

    print("Detected real serial ports:")
    real_ports = detect_serial_ports()
    if real_ports:
        for i, port in enumerate(real_ports, start=1):
            print(f"  [{i}] {port}")
    else:
        print("  (none detected)")

    manual = input("Enter port path or leave empty to choose from list: ").strip()
    if manual:
        return StartupConfig(mode="serial", port=manual, baudrate=baudrate)

    if not real_ports:
        print("No ports found. You can try bridge mode instead.")
        return None

    idx_text = input("Select port number: ").strip()
    if not idx_text.isdigit():
        return None
    idx = int(idx_text)
    if not (1 <= idx <= len(real_ports)):
        return None
    return StartupConfig(mode="serial", port=real_ports[idx - 1], baudrate=baudrate)

    def close(self) -> None:
        self.running = False
        if self.serial is not None:
            try:
                self.serial.close()
            except Exception:
                pass
            self.serial = None
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except Exception:
                pass
            self.master_fd = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Serial packet monitor for the face-tracking backend")
    parser.add_argument("--port", type=str, default=None, help="Serial port path (real device or bridge slave)")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument("--list-ports", action="store_true", help="List detected serial ports and exit")
    parser.add_argument("--bridge", action="store_true", help="Create an internal pseudo-TTY bridge for backend testing")
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

    interactive = (
        args.port is None
        and not args.bridge
        and sys.stdin.isatty()
    )

    if interactive:
        config = choose_startup_configuration()
        if config is None:
            return
        if config.mode == "bridge":
            args.bridge = True
            args.baudrate = config.baudrate
        else:
            args.port = config.port
            args.baudrate = config.baudrate

    monitor = SerialMonitor(port=args.port, baudrate=args.baudrate)
    try:
        if args.bridge:
            slave = monitor.create_bridge()
            print(f"[bridge] connect backend to: {slave}")
            monitor.start()
        else:
            if args.port is None:
                print("No serial port specified. Re-run with no flags for interactive selection.")
                return
            monitor.connect()
            monitor.start()
    except KeyboardInterrupt:
        print("\n[stopped]")
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
