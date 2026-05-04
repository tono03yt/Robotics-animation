# Python Backend and Serial Testing

This folder contains the face-tracking backend and the serial packet monitor used to test Arduino communication.

## Files

- [face_tracking_test.py](face_tracking_test.py) — camera + face detection backend
- [serial_io_test_client.py](serial_io_test_client.py) — serial monitor and bridge helper

---

## Install

From the project root:

```bash
python3 -m venv DEV/python_backend/.venv
source DEV/python_backend/.venv/bin/activate
python -m pip install --upgrade pip
pip install opencv-contrib-python==4.10.0.84 mediapipe==0.10.14 numpy==1.26.4 pyserial
```

Required packages:
- opencv-contrib-python
- mediapipe
- numpy
- pyserial

---

## Backend: face_tracking_test.py

The backend detects the largest face and sends a normalized position packet to the serial device:

```text
POS,<x_error>,<y_error>,<confidence>\n
```

### Example

```text
POS,0.1250,-0.2500,0.91
```

- `x_error`: horizontal offset from center, normalized to roughly `-1.0 .. +1.0`
- `y_error`: vertical offset from center, normalized to roughly `-1.0 .. +1.0`
- `confidence`: face detection confidence from MediaPipe

### Run

```bash
python face_tracking_test.py
```

A startup window lets you choose:
- camera
- resolution
- face detector mode
- serial port
- baudrate

### Flags

- `--max-cameras N`
  - Number of camera indices to probe.
- `--camera-index N`
  - Skip the selector and open one camera directly.
- `--model-selection 0|1`
  - `0` = short-range detector, `1` = full-range detector.
- `--min-detection-confidence FLOAT`
  - Minimum confidence threshold.
- `--serial-port PATH`
  - Serial device path for Arduino or a virtual tty.
- `--serial-baudrate N`
  - Serial baudrate, default `115200`.

### Virtual / test interfaces

The backend can use a virtual tty path such as `/dev/pts/7`.

Ways to select it:
- type it manually in the startup window
- pass it with `--serial-port /dev/pts/7`

---

## Serial Monitor: serial_io_test_client.py

This script prints and decodes packets from the backend or Arduino.

### Start with no flags

Run the monitor without any flags to open the startup selector:

```bash
python serial_io_test_client.py
```

You can then choose:
- real serial device
- internal bridge mode
- baudrate
- manual tty path or detected interface

### Packet formats

#### Position packet from backend
```text
POS,<x_error>,<y_error>,<confidence>\n
```

#### Status packet from Arduino
```text
STAT,<servo_us>,<millis>\n
```

#### Debug line
```text
[anything starting with [ ]
```

### Example output

```text
[rx ✓] POS     | x=+0.1250 y=-0.2500 conf=0.91
[rx ✓] STAT    | servo_us=1500 millis=42
[rx ✓] DEBUG   | [mock] ready
```

### Flags

- `--port PATH`
  - Open a real serial port or a bridge slave tty.
- `--baudrate N`
  - Serial baudrate, default `115200`.
- `--list-ports`
  - Show detected serial devices.
- `--bridge`
  - Create an internal pseudo-TTY bridge for backend testing.

If you do not pass any flags, the script opens an interactive selector first.

### Bridge mode

Use this when you want to test the backend without an Arduino.

1. Start the monitor with no flags.
2. Choose **internal bridge** in the startup selector.
3. It prints a slave tty path like `/dev/pts/7`.
4. Start the backend with that path:
  ```bash
  python face_tracking_test.py --serial-port /dev/pts/7
  ```
5. The monitor will print the packets that the backend writes.

### Real serial mode

```bash
python serial_io_test_client.py --port /dev/ttyUSB0
```

This is useful for observing an Arduino directly.

---

## Serial protocol summary

### Backend to Arduino
```text
POS,<x_error>,<y_error>,<confidence>\n
```

Meaning:
- `x_error < 0` means the face is left of center
- `x_error > 0` means the face is right of center
- `y_error < 0` means the face is above center
- `y_error > 0` means the face is below center
- `confidence` is the detection confidence

### Arduino to monitor
```text
STAT,<servo_us>,<millis>\n
```

Optional debug lines may also appear.

---

## Recommended workflows

### 1) No Arduino, internal bridge
```bash
python serial_io_test_client.py
# choose internal bridge
# copy the printed /dev/pts/X path
python face_tracking_test.py --serial-port /dev/pts/X
```

### 2) Real Arduino
```bash
python face_tracking_test.py --serial-port /dev/ttyUSB0
```

### 3) Observe a device directly
```bash
python serial_io_test_client.py --port /dev/ttyUSB0
```

---

## Notes

- The serial port can only be opened by one process at a time.
- Bridge mode is the easiest way to test backend output without an Arduino.
- The monitor script only prints packets; it does not move servos itself.
