/*
 * ==============================================================================
 * Flashing Instructions (Linux Command Line using arduino-cli)
 * For Arduino Nano 33 IoT
 * ==============================================================================
 * 1. Install arduino-cli if you haven't already:
 *    curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
 * 
 * 2. Install the SAMD core (required for the Nano 33 IoT):
 *    arduino-cli core install arduino:samd
 * 
 * 3. Find your Arduino's port:
 *    arduino-cli board list
 *    (Look for something like /dev/ttyACM0 and arduino:samd:nano_33_iot)
 * 
 * 4. Compile the sketch:
 *    arduino-cli compile --fqbn arduino:samd:nano_33_iot arduino_nano_33_iot.ino
 * 
 * 5. Upload the code to your Arduino:
 *    arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:samd:nano_33_iot arduino_nano_33_iot.ino
 * ==============================================================================
 * HARDWARE NOTE: 
 * The Arduino Nano 33 IoT uses 3.3V logic pins, unlike the 5V Arduino Uno. 
 * Most 5V servos will still register a 3.3V PWM control signal from pins 9 & 10 
 * without issues. However, you MUST power the servos from an external 5V power 
 * supply. Do not power the servos directly from the Nano's 3.3V or VUSB pin to 
 * avoid browning out or damaging the board.
 * ==============================================================================
 */

#include <Arduino.h>
#include <Servo.h>

Servo panServo;
Servo tiltServo;

// PWM capable pins on Nano 33 IoT
const int panPin = 9;
const int tiltPin = 10;

// Pan: Right/Left
const int panMin = 60;
const int panMax = 120;
const int panCenter = 90;

// Tilt: Up/Down
// Keep the range small at first so it doesn't crash into the frame
const int tiltMin = 75;
const int tiltMax = 105;
const int tiltCenter = 90;

// Buffer for non-blocking serial read
const byte numChars = 32;
char receivedChars[numChars];
boolean newData = false;

void setup() {
  Serial.begin(115200);

  // Optional: For native USB devices like Nano 33 IoT, you can wait for 
  // the serial port to connect, but we skip it here so the program 
  // runs even if the PC connects later.

  panServo.attach(panPin);
  tiltServo.attach(tiltPin);

  panServo.write(panCenter);
  tiltServo.write(tiltCenter);

  delay(500);
  Serial.println("Arduino Ready");
}

// Read incoming serial data without blocking the main loop
void recvWithEndMarker() {
  static byte ndx = 0;
  char endMarker = '\n';
  char rc;

  while (Serial.available() > 0 && newData == false) {
    rc = Serial.read();

    if (rc != endMarker) {
      receivedChars[ndx] = rc;
      ndx++;
      if (ndx >= numChars) {
        ndx = numChars - 1;
      }
    } else {
      receivedChars[ndx] = '\0'; // Terminate the string
      ndx = 0;
      newData = true;
    }
  }
}

void loop() {
  recvWithEndMarker();

  if (newData) {
    // Parse the received string formatted as "pan,tilt"
    char *strtokIndx;

    strtokIndx = strtok(receivedChars, ","); // Get first part (pan)
    if (strtokIndx != NULL) {
      int panVal = atoi(strtokIndx);

      strtokIndx = strtok(NULL, ","); // Get second part (tilt)
      if (strtokIndx != NULL) {
        int tiltVal = atoi(strtokIndx);

        // Constrain and write to servos immediately.
        // The PC's proportional control loop (in C) already calculates 
        // the incremental tracking speed, so the Arduino just executes it.
        if (panVal >= panMin && panVal <= panMax) {
          panServo.write(panVal);
        }

        if (tiltVal >= tiltMin && tiltVal <= tiltMax) {
          tiltServo.write(tiltVal);
        }
      }
    }
    newData = false;
  }
}
