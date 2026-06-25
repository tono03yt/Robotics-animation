/*
 * ==============================================================================
 * Flashing Instructions (Linux Command Line using arduino-cli)
 * For Arduino Nano 33 IoT
 *
 * arduino-cli compile --upload -p /dev/ttyACM0 --fqbn arduino:samd:nano_33_iot robot_code.ino
 * ==============================================================================
 */

#include <Servo.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

struct SmoothServo {
  Servo servo;
  uint8_t pin;
  float currentDeg;
  float targetDeg;
  float minDeg;
  float maxDeg;
  int minUs;
  int maxUs;
};

SmoothServo panAxis;
SmoothServo tiltAxis;
SmoothServo leftArmAxis;
SmoothServo rightArmAxis;

const uint8_t PAN_PIN = 9;
const uint8_t TILT_PIN = 10;
const uint8_t LEFT_ARM_PIN = 8;
const uint8_t RIGHT_ARM_PIN = 7;

const float PAN_CENTER = 90.0f;
const float TILT_CENTER = 90.0f;
const float LEFT_ARM_CENTER = 90.0f;
const float RIGHT_ARM_CENTER = 90.0f;

const unsigned long CONTROL_INTERVAL_MS = 10;
const float SMOOTHING_FACTOR = 0.18f;
const float POSITION_EPSILON = 0.02f;

const byte NUM_CHARS = 64;
char receivedChars[NUM_CHARS];
bool newData = false;

enum AnimationType {
  ANIM_NONE = 0,
  ANIM_WAVE = 1,
  ANIM_SPEECH = 2
};

bool animationActive = false;
AnimationType currentAnimation = ANIM_NONE;
unsigned long animationStartMs = 0;
const unsigned long WAVE_DURATION_MS = 1000;
const unsigned long SPEECH_DURATION_MS = 1200;

static void initServoState(
  SmoothServo &s,
  uint8_t pin,
  float center,
  float minDeg,
  float maxDeg,
  int minUs = 500,
  int maxUs = 2500
) {
  s.pin = pin;
  s.currentDeg = center;
  s.targetDeg = center;
  s.minDeg = minDeg;
  s.maxDeg = maxDeg;
  s.minUs = minUs;
  s.maxUs = maxUs;
}

static float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

static void trimToken(char *s) {
  if (!s) return;

  while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') {
    memmove(s, s + 1, strlen(s));
  }

  int len = strlen(s);
  while (len > 0) {
    char c = s[len - 1];
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      s[len - 1] = '\0';
      len--;
    } else {
      break;
    }
  }
}

static int degToUs(const SmoothServo &s, float deg) {
  float clamped = clampf(deg, s.minDeg, s.maxDeg);
  float ratio = (clamped - s.minDeg) / (s.maxDeg - s.minDeg);
  return (int)lroundf(s.minUs + ratio * (float)(s.maxUs - s.minUs));
}

static void writeServoNow(SmoothServo &s) {
  s.servo.writeMicroseconds(degToUs(s, s.currentDeg));
}

static void serviceServo(SmoothServo &s) {
  float error = s.targetDeg - s.currentDeg;

  if (fabsf(error) <= POSITION_EPSILON) {
    s.currentDeg = s.targetDeg;
  } else {
    s.currentDeg += error * SMOOTHING_FACTOR;
  }

  s.currentDeg = clampf(s.currentDeg, s.minDeg, s.maxDeg);
  writeServoNow(s);
}

static void stopAnimation() {
  animationActive = false;
  currentAnimation = ANIM_NONE;
  leftArmAxis.targetDeg = LEFT_ARM_CENTER;
  rightArmAxis.targetDeg = RIGHT_ARM_CENTER;
}

static void startAnimation(AnimationType anim) {
  currentAnimation = anim;
  animationActive = true;
  animationStartMs = millis();
}

static void updateAnimationTargets() {
  if (!animationActive) {
    leftArmAxis.targetDeg = LEFT_ARM_CENTER;
    rightArmAxis.targetDeg = RIGHT_ARM_CENTER;
    return;
  }

  unsigned long elapsed = millis() - animationStartMs;

  if (currentAnimation == ANIM_WAVE) {
    if (elapsed >= WAVE_DURATION_MS) {
      stopAnimation();
      return;
    }

    float t = (float)elapsed / (float)WAVE_DURATION_MS;
    float wave = 18.0f * sinf(t * 4.0f * PI);

    leftArmAxis.targetDeg = LEFT_ARM_CENTER;
    rightArmAxis.targetDeg = clampf(RIGHT_ARM_CENTER + wave, 70.0f, 110.0f);
  }
  else if (currentAnimation == ANIM_SPEECH) {
    if (elapsed >= SPEECH_DURATION_MS) {
      stopAnimation();
      return;
    }

    float t = (float)elapsed / (float)SPEECH_DURATION_MS;
    float wiggle = 12.0f * sinf(t * 6.0f * PI);

    leftArmAxis.targetDeg = clampf(LEFT_ARM_CENTER + wiggle, 60.0f, 120.0f);
    rightArmAxis.targetDeg = clampf(RIGHT_ARM_CENTER - wiggle, 60.0f, 120.0f);
  }
  else {
    stopAnimation();
  }
}

static void handleVectorCommand(char *line) {
  char *panTok = strtok(line, ",");
  char *tiltTok = strtok(NULL, ",");

  if (!panTok || !tiltTok) return;

  trimToken(panTok);
  trimToken(tiltTok);

  float panVal = atof(panTok);
  float tiltVal = atof(tiltTok);

  panAxis.targetDeg = clampf(panVal, panAxis.minDeg, panAxis.maxDeg);
  tiltAxis.targetDeg = clampf(tiltVal, tiltAxis.minDeg, tiltAxis.maxDeg);
}

static void handleAnimCommand(char *line) {
  char *prefix = strtok(line, ",");
  char *animTok = strtok(NULL, ",");

  (void)prefix;

  if (!animTok) return;
  trimToken(animTok);

  if (strcmp(animTok, "wave") == 0) {
    startAnimation(ANIM_WAVE);
  } else if (strcmp(animTok, "speech") == 0) {
    startAnimation(ANIM_SPEECH);
  }
}

static void processCommand() {
  trimToken(receivedChars);

  if (strncmp(receivedChars, "anim", 4) == 0) {
    handleAnimCommand(receivedChars);
  } else {
    handleVectorCommand(receivedChars);
  }

  newData = false;
}

static void recvWithEndMarker() {
  static byte ndx = 0;
  const char endMarker = '\n';
  char rc;

  while (Serial.available() > 0 && !newData) {
    rc = Serial.read();

    if (rc != endMarker) {
      receivedChars[ndx] = rc;
      ndx++;
      if (ndx >= NUM_CHARS) {
        ndx = NUM_CHARS - 1;
      }
    } else {
      receivedChars[ndx] = '\0';
      ndx = 0;
      newData = true;
    }
  }
}

static void serviceMotionLoop() {
  static unsigned long lastStepMs = 0;
  unsigned long now = millis();

  if ((now - lastStepMs) < CONTROL_INTERVAL_MS) return;
  lastStepMs = now;

  updateAnimationTargets();

  serviceServo(panAxis);
  serviceServo(tiltAxis);
  serviceServo(leftArmAxis);
  serviceServo(rightArmAxis);
}

void setup() {
  Serial.begin(115200);

  initServoState(panAxis, PAN_PIN, PAN_CENTER, 20.0f, 160.0f);
  initServoState(tiltAxis, TILT_PIN, TILT_CENTER, 45.0f, 135.0f);
  initServoState(leftArmAxis, LEFT_ARM_PIN, LEFT_ARM_CENTER, 0.0f, 180.0f);
  initServoState(rightArmAxis, RIGHT_ARM_PIN, RIGHT_ARM_CENTER, 0.0f, 180.0f);

  panAxis.servo.attach(panAxis.pin);
  tiltAxis.servo.attach(tiltAxis.pin);
  leftArmAxis.servo.attach(leftArmAxis.pin);
  rightArmAxis.servo.attach(rightArmAxis.pin);

  writeServoNow(panAxis);
  writeServoNow(tiltAxis);
  writeServoNow(leftArmAxis);
  writeServoNow(rightArmAxis);

  delay(500);
  Serial.println("Arduino Ready");
}

void loop() {
  recvWithEndMarker();

  if (newData) {
    processCommand();
  }

  serviceMotionLoop();
}
