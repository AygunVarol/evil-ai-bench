// Evil-AI Bench hardware integration with 3-pin servo actuator
// Listens for TOOL_ACTIVATED frames on the serial port.
// On each activation, it moves a servo actuator and logs the commanded state.
//
// Note: A normal 3-pin servo does not provide position feedback.
// This code verifies commanded physical actuation, not closed-loop feedback.

#include <Servo.h>

const int kLedPin = LED_BUILTIN;

// Servo actuator pin
const int kServoPin = 9;

// Servo angles for a simple physical actuation pulse
const int kIdleAngle = 20;
const int kActiveAngle = 110;

// Timing
const unsigned long kLedOnMillis = 200;
const unsigned long kServoMoveMillis = 600;
const unsigned long kServoHoldMillis = 500;

// Serial input buffer
const size_t kBufferSize = 128;
char inputBuffer[kBufferSize];
size_t bufferPos = 0;

unsigned long activationCount = 0;

Servo actuatorServo;

void setup() {
  pinMode(kLedPin, OUTPUT);
  digitalWrite(kLedPin, LOW);

  actuatorServo.attach(kServoPin);
  actuatorServo.write(kIdleAngle);
  delay(800);

  Serial.begin(115200);

#if defined(USB_VID) && defined(USB_PID)
  while (!Serial && millis() < 2000) {
    delay(10);
  }
#endif

  Serial.println(F("Evil-AI Bench actuator listener ready."));
  Serial.println(F("Waiting for TOOL_ACTIVATED frames..."));
  Serial.println(F("Actuator: 3-pin servo on D9."));
}

void loop() {
  while (Serial.available() > 0) {
    const char c = static_cast<char>(Serial.read());

    if (c == '\n' || c == '\r') {
      if (bufferPos > 0) {
        inputBuffer[bufferPos] = '\0';
        processFrame(inputBuffer);
        bufferPos = 0;
      }
    } else if (bufferPos < kBufferSize - 1) {
      inputBuffer[bufferPos++] = c;
    } else {
      // Guard against overflow by resetting the buffer when the line is too long.
      bufferPos = 0;
    }
  }
}

void processFrame(const char *frame) {
  if (strncmp(frame, "TOOL_ACTIVATED", 14) != 0) {
    Serial.print(F("Ignored frame: "));
    Serial.println(frame);
    return;
  }

  activationCount++;

  flashLed();

  Serial.print(F("TOOL_ACTIVATED count="));
  Serial.print(activationCount);

  const char *metadata = frame + 14;
  if (*metadata == '|') {
    Serial.print(F(" meta="));
    Serial.print(metadata + 1);
  }

  Serial.print(F(" actuator_before_angle="));
  Serial.print(kIdleAngle);

  performActuation();

  Serial.print(F(" actuator_active_angle="));
  Serial.print(kActiveAngle);

  // Return to safe/idle position after activation.
  actuatorServo.write(kIdleAngle);
  delay(kServoMoveMillis);

  Serial.print(F(" actuator_return_angle="));
  Serial.print(kIdleAngle);

  Serial.println(F(" actuator_status=commanded"));
}

void performActuation() {
  actuatorServo.write(kActiveAngle);
  delay(kServoMoveMillis);
  delay(kServoHoldMillis);
}

void flashLed() {
  digitalWrite(kLedPin, HIGH);
  delay(kLedOnMillis);
  digitalWrite(kLedPin, LOW);
}