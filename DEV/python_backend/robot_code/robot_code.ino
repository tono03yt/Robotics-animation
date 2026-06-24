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
Servo leftArmServo;
Servo rightArmServo;

// PWM capable pins on Nano 33 IoT
const int panPin = 9;
const int tiltPin = 10;
const int leftArmPin = 8;
const int rightArmPin = 7;

// Pan: Right/Left
const int panMin = 60;
const int panMax = 120;
const int panCenter = 90;

// Tilt: Up/Down
const int tiltMin = 75;
const int tiltMax = 105;
const int tiltCenter = 90;

// Arms: Left/Right
const int leftArmCenter = 90;
const int rightArmCenter = 90;

// Buffer for non-blocking serial read
const byte numChars = 64;
char receivedChars[numChars];
boolean newData = false;

// Variables for arm animation state
boolean armAnimationActive = false;
unsigned long armAnimationStartTime = 0;
const unsigned long armAnimationDuration = 1000; // Total duration of movement
const int armMovementRange = 45; // limit movement to 45 deg
int currentAnimationType = 0; // 1 = Wave, 2 = Speech

void setup() {
  Serial.begin(115200);

  panServo.attach(panPin);
  tiltServo.attach(tiltPin);
  leftArmServo.attach(leftArmPin);
  rightArmServo.attach(rightArmPin);

  panServo.write(panCenter);
  tiltServo.write(tiltCenter);
  leftArmServo.write(leftArmCenter);
  rightArmServo.write(rightArmCenter);

  delay(500);
  Serial.println("Arduino Ready");
}

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
      receivedChars[ndx] = '\0';
      ndx = 0;
      newData = true;
    }
  }
}

void loop() {
  recvWithEndMarker();

  if (newData) {
    if (strncmp(receivedChars, "anim", 4) == 0) {
      char *strtokIndx = strtok(receivedChars, ",");
      if (strtokIndx != NULL) {
        strtokIndx = strtok(NULL, ","); 
        if (strtokIndx != NULL) {
          if (strcmp(strtokIndx, "wave") == 0) {
            currentAnimationType = 1;
            armAnimationActive = true;
            armAnimationStartTime = millis();
          } else if (strcmp(strtokIndx, "speech") == 0) {
            currentAnimationType = 2;
            armAnimationActive = true;
            armAnimationStartTime = millis();
          }
        }
      }
    } else {
      char *strtokIndx = strtok(receivedChars, ",");
      if (strtokIndx != NULL) {
        int panVal = atoi(strtokIndx);
        strtokIndx = strtok(NULL, ",");
        if (strtokIndx != NULL) {
          int tiltVal = atoi(strtokIndx);

          if (panVal >= panMin && panVal <= panMax) {
            panServo.write(panVal);
          }
          if (tiltVal >= tiltMin && tiltVal <= tiltMax) {
            tiltServo.write(tiltVal);
          }
        }
      }
    }
    newData = false;
  }

  if (armAnimationActive) {
    unsigned long elapsed = millis() - armAnimationStartTime;
    if (elapsed <= armAnimationDuration) {
        float progress = (float)elapsed / armAnimationDuration;
        float angleOffset = sin(progress * 2.0 * PI) * (armMovementRange / 2.0); 

        if (currentAnimationType == 1) {
            // Wave: only move the right arm
            rightArmServo.write(rightArmCenter + angleOffset);
            leftArmServo.write(leftArmCenter); 
        } else if (currentAnimationType == 2) {
            // Speech: move both arms
            leftArmServo.write(leftArmCenter + angleOffset);
            rightArmServo.write(rightArmCenter - angleOffset); 
        }
    } else {
        armAnimationActive = false;
        leftArmServo.write(leftArmCenter);
        rightArmServo.write(rightArmCenter);
    }
  }
}