# Hardware Control Pipeline: IPC Receiver & Arduino

This folder contains the hardware execution layer of the face-tracking robot pipeline. It consists of two parts:
1. **The Arduino Sketch (`robot_code.ino`)**: Runs on the microcontroller and directly drives the servo motors.
2. **The IPC Receiver (`ipc_reciever_robot.c`)**: Runs on the Linux host machine. It receives coordinate data from the Python AI backend, runs a proportional math controller, and sends precise angle calculations to the Arduino.

---

## 1. The Arduino Code (`robot_code.ino`)

The Arduino acts purely as a "dumb" actuator. It listens to the USB serial connection for strings like `95,102\n` and instantly maps those to the Pan and Tilt servo PWM outputs. It does not perform any math, tracking, or interpolation itself.

### Hardware Setup (Nano 33 IoT)
- **Pan Servo**: Connect signal wire to **Pin 9**
- **Tilt Servo**: Connect signal wire to **Pin 10**
- **Power Warning**: The Nano 33 IoT uses **3.3V logic**. You **MUST** power your 5V servos from an external power supply. Connecting heavy servos to the Nano's `VUSB` or `3.3V` pins will brown out or permanently damage the microcontroller.

### Compiling and Flashing
You will need `arduino-cli` installed.

1. **Install the SAMD Core** (Required for Nano 33 IoT):
   ```bash
   arduino-cli core install arduino:samd
   ```

2. **Install the Servo Library**:
   ```bash
   arduino-cli lib install Servo
   ```

3. **Compile and Upload**:
   Find your port (e.g., `/dev/ttyACM0`) using `arduino-cli board list`, then run:
   ```bash
   arduino-cli compile --upload -p /dev/ttyACM0 --fqbn arduino:samd:nano_33_iot robot_code.ino
   ```

---

## 2. The IPC Receiver (`ipc_reciever_robot.c`)

This C script runs on your Linux machine (or Raspberry Pi). It creates a Unix Domain Socket file at `/tmp/robot_pipeline.sock`. The Python AI backend connects to this socket and streams raw `x, y` face coordinates. 

The C script parses these coordinates, applies a **Proportional Control Loop (P-Controller)** to calculate smooth servo tracking angles, and pushes those angles down the USB cable to the Arduino.

### Why a C Receiver?
Running the tight, time-sensitive control loop math in a lightweight C script prevents the servos from stuttering or lagging when the heavy Python AI backend drops frames or pauses to process audio/wake-words.

### Compiling the C Receiver
This requires the standard GCC compiler.
```bash
gcc -Wall -o ipc_reciever_robot ipc_reciever_robot.c
```

### Running the System
1. **Start the Receiver**:
   ```bash
   ./ipc_reciever_robot
   ```
2. **Select your Port**:
   The script will scan for connected Arduinos. Type the full path of your Arduino (e.g., `/dev/ttyACM0`) and press Enter. 
   *(Note: It features background retry logic. If the Arduino gets unplugged, it will wait and auto-reconnect when plugged back in).*

3. **Start the AI Backend**:
   Open a new terminal window, navigate to your Python backend directory, and run the Python tracker. It will automatically detect the socket created by the C script and begin streaming data.
