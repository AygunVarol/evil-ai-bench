// Simple 3-pin servo wiring test
// Signal pin: D9
// VCC: 5V
// GND: GND

#include <Servo.h>

const int servoPin = 9;

Servo testServo;

void setup() {
  Serial.begin(115200);

  testServo.attach(servoPin);

  Serial.println("Servo test started.");
  Serial.println("Servo should move to 20, 90, and 110 degrees repeatedly.");

  // Initial position
  testServo.write(20);
  delay(1000);
}

void loop() {
  Serial.println("Moving to 20 degrees");
  testServo.write(20);
  delay(1500);

  Serial.println("Moving to 90 degrees");
  testServo.write(90);
  delay(1500);

  Serial.println("Moving to 110 degrees");
  testServo.write(110);
  delay(1500);

  Serial.println("Returning to 20 degrees");
  testServo.write(20);
  delay(1500);
}