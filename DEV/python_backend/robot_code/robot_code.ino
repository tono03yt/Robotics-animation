/*
* ==============================================================================
* Flashing Instructions (Linux Command Line using arduino-cli)
* For Arduino Nano 33 IoT

 arduino-cli compile --upload -p /dev/ttyACM0 --fqbn arduino:samd:nano_33_iot robot_code.ino

* ==============================================================================
*/

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

  // Initialize center positions
  panServo.write(90);
  tiltServo.write(90);
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

  // Handle incoming serial commands
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
          
          // Removed pan and tilt min/max constraints 
          panServo.write(panVal);
          tiltServo.write(tiltVal);
        }
      }
    }
    newData = false;
  }

  // Handle Animations outside of the newData check so it updates continuously
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
      // Animation finished, reset arms
      armAnimationActive = false;
      leftArmServo.write(leftArmCenter);
      rightArmServo.write(rightArmCenter);
    }
  }
}