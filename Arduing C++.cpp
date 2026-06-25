#include <Arduino.h>
#include <Servo.h>

/*
  Robot Head Controller
  ---------------------
  Linux/Python:
    - Camera processing
    - Face detection
    - Sends normalized error: ERR:x,y

  Arduino:
    - Receives error values
    - Runs PD controller
    - Applies deadband, limits, smoothing, and failsafe
    - Controls pan/tilt servos
*/

// ================= Servo objects =================
Servo panServo;
Servo tiltServo;

// ================= Pins =================
#define PAN_PIN   9
#define TILT_PIN  10

// ================= Mechanical limits =================
// Bewegungsgrezen der Servos
const float PAN_MIN    = 60.0;
const float PAN_MAX    = 120.0;
const float PAN_CENTER = 90.0;

const float TILT_MIN    = 75.0;
const float TILT_MAX    = 105.0;
const float TILT_CENTER = 90.0;

// Bewegungsrichtung
// Change sign if the servo moves in the wrong direction
const float PAN_DIR  = 1.0;
const float TILT_DIR = -1.0;

// ================= Controller parameters =================
// Python sends errX and errY normalized from -1.0 to +1.0

// PD controller for pan axis
float Kp_pan = 85.0;
float Ki_pan = 0.0;
float Kd_pan = 8.0;

// PD controller for tilt axis
float Kp_tilt = 65.0;
float Ki_tilt = 0.0;
float Kd_tilt = 6.0;

// Deadband to reduce small oscillations
const float DEADBAND_X = 0.04;
const float DEADBAND_Y = 0.04;

// Maximum target angle speed
const float MAX_TARGET_SPEED = 120.0;  // degree/second

// Servo smoothing speed
const float SERVO_MOVE_SPEED = 90.0;   // degree/second

// Failsafe time
// If no serial data is received, the robot returns to center
const unsigned long FAILSAFE_TIME_MS = 600;

// ================= State variables =================
float currentPan  = PAN_CENTER;
float currentTilt = TILT_CENTER;

float targetPan  = PAN_CENTER;
float targetTilt = TILT_CENTER;

float errX = 0.0;
float errY = 0.0;

float prevErrX = 0.0;
float prevErrY = 0.0;

float integralX = 0.0;
float integralY = 0.0;

unsigned long lastSerialTime = 0;
unsigned long lastControlTime = 0;
unsigned long lastServoMoveTime = 0;

bool faceDetected = false;

// ================= Helper functions =================
float clampFloat(float value, float minValue, float maxValue)
{
  if (value < minValue)
  {
    return minValue;
  }

  if (value > maxValue)
  {
    return maxValue;
  }

  return value;
}

float applyDeadband(float value, float deadband)
{
  if (abs(value) < deadband)
  {
    return 0.0;
  }

  return value;
}

// ================= Servo movement =================
void moveServoSmoothly(void)
{
  unsigned long now = millis();
  float dt = (now - lastServoMoveTime) / 1000.0;

  if (dt <= 0.0)
  {
    return;
  }

  lastServoMoveTime = now;

  float maxStep = SERVO_MOVE_SPEED * dt;

  // Pan movement
  float panDiff = targetPan - currentPan;

  if (abs(panDiff) <= maxStep)
  {
    currentPan = targetPan;
  }
  else
  {
    if (panDiff > 0)
    {
      currentPan += maxStep;
    }
    else
    {
      currentPan -= maxStep;
    }
  }

  // Tilt movement
  float tiltDiff = targetTilt - currentTilt;

  if (abs(tiltDiff) <= maxStep)
  {
    currentTilt = targetTilt;
  }
  else
  {
    if (tiltDiff > 0)
    {
      currentTilt += maxStep;
    }
    else
    {
      currentTilt -= maxStep;
    }
  }

  currentPan  = clampFloat(currentPan, PAN_MIN, PAN_MAX);
  currentTilt = clampFloat(currentTilt, TILT_MIN, TILT_MAX);

  panServo.write((int)currentPan);
  tiltServo.write((int)currentTilt);
}

// ================= Controller reset =================
void resetController(void)
{
  errX = 0.0;
  errY = 0.0;

  prevErrX = 0.0;
  prevErrY = 0.0;

  integralX = 0.0;
  integralY = 0.0;
}

// ================= Controller update =================
void updateController(void)
{
  unsigned long now = millis();
  float dt = (now - lastControlTime) / 1000.0;

  if (dt <= 0.0)
  {
    return;
  }

  lastControlTime = now;

  // Failsafe: no face or no serial data
  if (!faceDetected || (now - lastSerialTime > FAILSAFE_TIME_MS))
  {
    faceDetected = false;
    resetController();

    targetPan = PAN_CENTER;
    targetTilt = TILT_CENTER;

    return;
  }

  float eX = applyDeadband(errX, DEADBAND_X);
  float eY = applyDeadband(errY, DEADBAND_Y);

  // Integral part, currently Ki = 0
  integralX += eX * dt;
  integralY += eY * dt;

  // Anti-windup
  integralX = clampFloat(integralX, -0.5, 0.5);
  integralY = clampFloat(integralY, -0.5, 0.5);

  // Derivative part
  float derivativeX = (eX - prevErrX) / dt;
  float derivativeY = (eY - prevErrY) / dt;

  prevErrX = eX;
  prevErrY = eY;

  // PID / PD controller output
  float panSpeedCommand =
    PAN_DIR * ((Kp_pan * eX) + (Ki_pan * integralX) + (Kd_pan * derivativeX));

  float tiltSpeedCommand =
    TILT_DIR * ((Kp_tilt * eY) + (Ki_tilt * integralY) + (Kd_tilt * derivativeY));

  panSpeedCommand = clampFloat(panSpeedCommand, -MAX_TARGET_SPEED, MAX_TARGET_SPEED);
  tiltSpeedCommand = clampFloat(tiltSpeedCommand, -MAX_TARGET_SPEED, MAX_TARGET_SPEED);

  // Integrate angular speed command into target angle
  targetPan += panSpeedCommand * dt;
  targetTilt += tiltSpeedCommand * dt;

  targetPan = clampFloat(targetPan, PAN_MIN, PAN_MAX);
  targetTilt = clampFloat(targetTilt, TILT_MIN, TILT_MAX);
}

// ================= Serial communication =================
void readSerialCommand(void)
{
  if (!Serial.available())
  {
    return;
  }

  String line = Serial.readStringUntil('\n');
  line.trim();

  if (line.length() == 0)
  {
    return;
  }

  // Python sends NOFACE if no face is detected
  if (line == "NOFACE")
  {
    faceDetected = false;
    return;
  }

  // Expected format:
  // ERR:0.25,-0.10
  if (line.startsWith("ERR:"))
  {
    int commaIndex = line.indexOf(',');

    if (commaIndex > 4)
    {
      String xString = line.substring(4, commaIndex);
      String yString = line.substring(commaIndex + 1);

      errX = xString.toFloat();
      errY = yString.toFloat();

      errX = clampFloat(errX, -1.0, 1.0);
      errY = clampFloat(errY, -1.0, 1.0);

      faceDetected = true;
      lastSerialTime = millis();
    }
  }
}

// ================= Setup =================
void setup(void)
{
  Serial.begin(115200);
  Serial.setTimeout(5);

  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);

  currentPan = PAN_CENTER;
  currentTilt = TILT_CENTER;

  targetPan = PAN_CENTER;
  targetTilt = TILT_CENTER;

  panServo.write((int)currentPan);
  tiltServo.write((int)currentTilt);

  lastSerialTime = millis();
  lastControlTime = millis();
  lastServoMoveTime = millis();

  delay(500);
}

// ================= Main loop =================
void loop(void)
{
  readSerialCommand();
  updateController();
  moveServoSmoothly();
}