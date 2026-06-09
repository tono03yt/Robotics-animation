#include <Arduino.h>
#include <Servo.h>

Servo panServo;
Servo tiltServo;

const int panPin = 9;
const int tiltPin = 10;

// Pan: يمين/شمال
const int panMin = 60;
const int panMax = 120;
const int panCenter = 90;

// Tilt: فوق/تحت
// خلي المدى صغير في الأول عشان ما يخبطش
const int tiltMin = 75;
const int tiltMax = 105;
const int tiltCenter = 90;

float currentPan = panCenter;
float targetPan = panCenter;

float currentTilt = tiltCenter;
float targetTilt = tiltCenter;

const float stepSize = 1.0;
const unsigned long moveInterval = 15;
unsigned long lastMoveTime = 0;

void setup() {
  Serial.begin(115200);

  panServo.attach(panPin);
  tiltServo.attach(tiltPin);

  panServo.write(panCenter);
  tiltServo.write(tiltCenter);

  delay(500);
  Serial.println("Arduino Ready");
}

void loop() {
  if (Serial.available()) {
    String data = Serial.readStringUntil('\n');

    int commaIndex = data.indexOf(',');

    if (commaIndex > 0) {
      int panVal = data.substring(0, commaIndex).toInt();
      int tiltVal = data.substring(commaIndex + 1).toInt();

      if (panVal >= panMin && panVal <= panMax) {
        targetPan = panVal;
      }

      if (tiltVal >= tiltMin && tiltVal <= tiltMax) {
        targetTilt = tiltVal;
      }
    }
  }

  unsigned long now = millis();

  if (now - lastMoveTime >= moveInterval) {
    lastMoveTime = now;

    if (currentPan < targetPan) {
      currentPan += stepSize;
      if (currentPan > targetPan) currentPan = targetPan;
    } else if (currentPan > targetPan) {
      currentPan -= stepSize;
      if (currentPan < targetPan) currentPan = targetPan;
    }

    if (currentTilt < targetTilt) {
      currentTilt += stepSize;
      if (currentTilt > targetTilt) currentTilt = targetTilt;
    } else if (currentTilt > targetTilt) {
      currentTilt -= stepSize;
      if (currentTilt < targetTilt) currentTilt = targetTilt;
    }

    panServo.write((int)currentPan);
    tiltServo.write((int)currentTilt);
  }
}