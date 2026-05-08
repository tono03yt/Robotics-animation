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
import threading
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
ANIM_RE = re.compile(r"^ANIM,(\w+),(.*)$")
AUDIO_RE = re.compile(r"^AUDIO,(.+)$")
DEBUG_RE = re.compile(r"^\[")


class Ansi:
    RESET = "\033[0m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def use_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def colorize(text: str, color: str) -> str:
    if not use_color():
        return text
    return f"{color}{text}{Ansi.RESET}"


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

    match = ANIM_RE.match(line)
    if match:
        return Packet(True, "ANIM", {"animation": match.group(1), "text": match.group(2)}, line)

    match = AUDIO_RE.match(line)
    if match:
        b64 = match.group(1)
        return Packet(True, "AUDIO", {"base64_len": len(b64)}, line)

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
        # None = show all; otherwise a set of kinds to display (e.g. {"POS","STAT"})
        self.display_kinds: Optional[set[str]] = None

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
        # filter by kind if requested
        if self.display_kinds is not None and packet.kind not in self.display_kinds:
            return
        status = colorize("✓", Ansi.GREEN) if packet.valid else colorize("✗", Ansi.RED)
        if packet.kind == "POS":
            print(
                f"{colorize('[rx]', Ansi.CYAN)} {status} {colorize('POS', Ansi.BLUE):<14} | x={packet.fields['x_error']:+.4f} "
                f"y={packet.fields['y_error']:+.4f} conf={packet.fields['confidence']:.2f}"
            )
        elif packet.kind == "STAT":
            print(f"{colorize('[rx]', Ansi.CYAN)} {status} {colorize('STAT', Ansi.YELLOW):<14} | servo_us={packet.fields['servo_us']} millis={packet.fields['millis']}")
        elif packet.kind == "ANIM":
            print(f"{colorize('[rx]', Ansi.CYAN)} {status} {colorize('ANIM', Ansi.MAGENTA):<14} | animation={packet.fields['animation']} text={packet.fields['text']}")
        elif packet.kind == "AUDIO":
            print(f"{colorize('[rx]', Ansi.CYAN)} {status} {colorize('AUDIO', Ansi.GREEN):<14} | base64_len={packet.fields['base64_len']}")
        elif packet.kind == "DEBUG":
            print(f"{colorize('[rx]', Ansi.CYAN)} {status} {colorize('DEBUG', Ansi.DIM):<14} | {packet.fields['text']}")
        else:
            print(f"{colorize('[rx]', Ansi.CYAN)} {status} {colorize(packet.kind, Ansi.RED):<14} | {packet.raw}")

    def _bridge_loop(self) -> None:
        assert self.master_fd is not None
        buffer = b""
        print(colorize(f"[bridge] backend should connect to: {self.bridge_slave_path}", Ansi.YELLOW))
        print(colorize("[bridge] monitoring master side; press Ctrl+C to stop", Ansi.DIM))
        try:
            while self.running:
                try:
                    chunk = os.read(self.master_fd, 4096)
                    if not chunk:
                        self._flush_input_queue()
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
                    self._flush_input_queue()
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
        print(colorize(f"[connected] {self.port} @ {self.baudrate}", Ansi.GREEN))
        try:
            while self.running:
                self._flush_input_queue()
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

    def _flush_input_queue(self) -> None:
        return

    def set_display_kinds(self, kinds: Optional[List[str]]) -> None:
        if kinds is None:
            self.display_kinds = None
            return
        self.display_kinds = set(k.upper() for k in kinds)

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
    print("\n" + "-" * 20)
    print("Step 1: Select Mode")
    print("-" * 20)
    print("1) Real serial device  - Connect to a physical Arduino or USB-to-serial adapter.")
    print("2) Internal bridge     - Create a virtual serial port for local testing (no hardware needed).")
    print("3) List interfaces     - Show all detected real and virtual serial ports.")
    print("q) Quit")

    choice = _ask_choice("Select mode [1/2/3/q]: ", {"1", "2", "3", "q"})
    if choice == "q":
        return None

    if choice == "3":
        print("-" * 40)
        real_ports = detect_serial_ports()
        pts_ports = detect_virtual_ttys()
        if real_ports:
            print("Detected real serial ports (e.g., Arduino, USB adapters):")
            for port in real_ports:
                print(f"  - {port}")
        else:
            print("No real serial ports detected.")
        if pts_ports:
            print("\nDetected virtual tty interfaces (for local testing):")
            for port in pts_ports:
                print(f"  - {port}")
        else:
            print("\nNo virtual tty interfaces detected yet.")
        print("-" * 40)
        return choose_startup_configuration()

    if choice == "1":
        ports = detect_serial_ports()
        if not ports:
            print("\nError: No real serial ports were detected.")
            print("Please ensure your device is connected and you have the correct permissions.")
            return None
        print("\nPlease select the serial device to connect to:")
        for i, port in enumerate(ports, 1):
            print(f"  [{i}] {port}")
        port_choice = input(f"Enter number (1-{len(ports)}): ").strip()
        if not port_choice.isdigit() or not (1 <= int(port_choice) <= len(ports)):
            print("Invalid selection.")
            return None
        selected_port = ports[int(port_choice) - 1]
        print(f"\n> You selected: Real serial device '{selected_port}'")
    else:  # choice == "2"
        selected_port = None
        print("\n> You selected: Internal bridge mode.")
        print("> A new virtual serial port will be created.")
        print("> The backend script should connect to this new port path.")

    baud_text = input("Baudrate [115200]: ").strip()
    baudrate = int(baud_text) if baud_text.isdigit() else 115200
    print(f"> Using baudrate: {baudrate}")

    if choice == "1":
        return StartupConfig(mode="serial", port=selected_port, baudrate=baudrate)
    return StartupConfig(mode="bridge", port=None, baudrate=baudrate)


def print_help() -> None:
    print("\n" + "=" * 60)
    print("SERIAL MONITOR HELP")
    print("=" * 60)
    print("This script emulates the robot's serial interface for testing the backend.")
    print()
    print("This client is read-only: it listens for serial packets from the backend.")
    print()
    print("Packet types:")
    print("  POS     - Face position data (x_error, y_error, confidence)")
    print("  STAT    - Servo status (servo_us, millis)")
    print("  ANIM    - Animation response from backend (animation, text)")
    print("  AUDIO   - Audio data (base64 encoded)")
    print("  DEBUG   - Debug messages")
    print()
    print("Display presets:")
    print("  1 (all)          - Show all packet types")
    print("  2 (tracking)     - Show POS + STAT (face tracking)")
    print("  3 (audio_test)   - Show ANIM + AUDIO (LLM/audio testing)")
    print("  4 (custom)       - Enter custom packet types")
    print()
    print("During monitoring:")
    print("  Ctrl+C           - Stop and change options")
    print("=" * 60 + "\n")


def ask_display_kinds_interactive() -> tuple[Optional[List[str]], bool]:
    print("\n" + "-" * 20)
    print("Step 2: Select Packets to Display")
    print("-" * 20)
    print("Choose a preset to filter which serial packets are shown.")
    print("  1) All          - Show all packets (useful for general debugging).")
    print("  2) Tracking     - Show POS and STAT packets (for face tracking testing).")
    print("  3) LLM/Audio    - Show TEXT, ANIM, and AUDIO packets (for AI testing).")
    print("  4) Custom       - Manually enter a list of packet types to show.")
    choice = _ask_choice("Select preset [1/2/3/4]: ", {"1", "2", "3", "4"})

    kinds: Optional[List[str]] = None

    if choice == "1":
        print("> Displaying all packets.")
        kinds = None
    elif choice == "2":
        print("> Displaying: POS, STAT")
        kinds = ["POS", "STAT"]
    elif choice == "3":
        print("> Displaying: ANIM, AUDIO")
        kinds = ["ANIM", "AUDIO"]
    else:  # choice == "4": custom
        print("\nEnter comma-separated packet types (e.g., POS, STAT, ANIM, AUDIO, TEXT, DEBUG):")
        txt = input("Types: ").strip()
        if not txt:
            print("> No types entered, will display all packets.")
            kinds = None
        else:
            kinds = [p.strip().upper() for p in txt.split(",") if p.strip()]
            print(f"> Displaying: {', '.join(kinds)}")
    return kinds, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Serial packet monitor for the face-tracking backend")
    parser.add_argument("--port", type=str, default=None, help="Serial port path (real device or bridge slave)")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument("--list-ports", action="store_true", help="List detected serial ports and exit")
    parser.add_argument("--bridge", action="store_true", help="Create an internal pseudo-TTY bridge for backend testing")
    parser.add_argument("--help-info", action="store_true", help="Show help and exit")
    args = parser.parse_args()

    if args.help_info:
        print_help()
        return

    # No --help-info, so don't print it unless requested.
    # print_help()

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
    enable_text_input = False
    # initial display kinds (interactive prompt if running interactively)
    if sys.stdin.isatty():
        kinds, enable_text_input = ask_display_kinds_interactive()
        monitor.set_display_kinds(kinds)

    try:
        while True:
            try:
                if args.bridge:
                    slave = monitor.create_bridge()
                    print(f"\n[bridge] connect backend to: {slave}")
                    print("[bridge] Press Ctrl+C to change options")
                else:
                    if args.port is None:
                        print("No serial port specified. Re-run with no flags for interactive selection.")
                        return
                    print(f"\n[serial] Press Ctrl+C to change options")
                # Run the monitor in the background when text input is enabled so the main
                # terminal can reliably accept typed TEXT packets.
                monitor.start()
                # normal exit from monitor (not Ctrl+C)
                break
            except KeyboardInterrupt:
                print("\n[stopped] — interrupted")
                # allow user to change filters or reselect startup
                if not sys.stdin.isatty():
                    break
                print("Options: [f] change filters  [s] reselect startup  [q] quit")
                action = input("Choice [f/s/q]: ").strip().lower()
                if action == "q":
                    monitor.close()
                    break
                if action == "f":
                    kinds, enable_text_input = ask_display_kinds_interactive()
                    monitor.set_display_kinds(kinds)
                    # restart reading with same connection; set running True and loop
                    monitor.running = True
                    continue
                if action == "s":
                    monitor.close()
                    cfg = choose_startup_configuration()
                    if cfg is None:
                        break
                    if cfg.mode == "bridge":
                        args.bridge = True
                        args.baudrate = cfg.baudrate
                        args.port = None
                    else:
                        args.bridge = False
                        args.port = cfg.port
                        args.baudrate = cfg.baudrate
                    monitor = SerialMonitor(port=args.port, baudrate=args.baudrate)
                    kinds, enable_text_input = ask_display_kinds_interactive()
                    monitor.set_display_kinds(kinds)
                    continue
                # unknown option -> quit
                monitor.close()
                break
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
