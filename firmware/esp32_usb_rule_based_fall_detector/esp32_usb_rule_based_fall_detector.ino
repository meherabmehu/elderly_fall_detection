#include <Arduino.h>
#include <Wire.h>
#include <math.h>

// AI-free rule-based fall detector
// Human sensor input + Serial Monitor manual test commands

const uint8_t MPU6050_ADDR = 0x68;

const int SDA_PIN = 21;
const int SCL_PIN = 22;
const int BUZZER_PIN = 4;
const int LED_PIN = 2;

const float FREEFALL_THRESHOLD_G = 0.65f;
const float IMPACT_THRESHOLD_G = 2.50f;
const float ROTATION_THRESHOLD_DPS = 250.0f;

const unsigned long SAMPLE_INTERVAL_MS = 20;
const unsigned long FREEFALL_MIN_MS = 80;
const unsigned long FALL_CONFIRM_WINDOW_MS = 1500;
const unsigned long ALARM_DURATION_MS = 3000;
const unsigned long ALARM_COOLDOWN_MS = 5000;

unsigned long last_sample_time = 0;
unsigned long freefall_start_time = 0;
unsigned long alarm_start_time = 0;
unsigned long last_alarm_time = 0;

bool sensor_ready = false;
bool freefall_detected = false;
bool alarm_active = false;

void write_register(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission(true);
}

uint8_t read_register(uint8_t reg) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return 0xFF;
  if (Wire.requestFrom(MPU6050_ADDR, (uint8_t)1, (uint8_t)true) != 1) return 0xFF;
  return Wire.read();
}

bool i2c_exists(uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission() == 0;
}

void scan_i2c_bus() {
  Serial.println("Scanning I2C bus...");
  int found = 0;

  for (uint8_t address = 1; address < 127; address++) {
    if (i2c_exists(address)) {
      Serial.print("I2C device found at address: 0x");
      if (address < 16) Serial.print("0");
      Serial.println(address, HEX);
      found++;
    }
  }

  Serial.print("Total I2C devices found: ");
  Serial.println(found);
}

bool initialize_mpu6050() {
  if (!i2c_exists(MPU6050_ADDR)) {
    return false;
  }

  Serial.print("WHO_AM_I: 0x");
  uint8_t who = read_register(0x75);
  if (who < 16) Serial.print("0");
  Serial.println(who, HEX);

  // Wake sensor
  write_register(0x6B, 0x00);

  // Accelerometer +/-16G
  write_register(0x1C, 0x18);

  // Gyroscope +/-2000 DPS
  write_register(0x1B, 0x18);

  // Digital low-pass filter
  write_register(0x1A, 0x04);

  // 50 Hz sample rate
  write_register(0x19, 19);

  delay(100);
  return true;
}

int16_t read_int16() {
  uint8_t high_byte = Wire.read();
  uint8_t low_byte = Wire.read();
  return (int16_t)((high_byte << 8) | low_byte);
}

bool read_sensor(
  float &ax,
  float &ay,
  float &az,
  float &gx,
  float &gy,
  float &gz
) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x3B);

  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU6050_ADDR, (uint8_t)14, (uint8_t)true) != 14) return false;

  int16_t raw_ax = read_int16();
  int16_t raw_ay = read_int16();
  int16_t raw_az = read_int16();
  int16_t raw_temp = read_int16();
  int16_t raw_gx = read_int16();
  int16_t raw_gy = read_int16();
  int16_t raw_gz = read_int16();

  (void)raw_temp;

  // +/-16G = 2048 LSB/G
  ax = (float)raw_ax / 2048.0f;
  ay = (float)raw_ay / 2048.0f;
  az = (float)raw_az / 2048.0f;

  // +/-2000 DPS = 16.4 LSB/DPS
  gx = (float)raw_gx / 16.4f;
  gy = (float)raw_gy / 16.4f;
  gz = (float)raw_gz / 16.4f;

  return true;
}

void start_alarm(const char *reason) {
  if (alarm_active) return;

  alarm_active = true;
  alarm_start_time = millis();
  last_alarm_time = alarm_start_time;

  digitalWrite(LED_PIN, HIGH);
  digitalWrite(BUZZER_PIN, HIGH);

  Serial.println();
  Serial.println("==============================");
  Serial.print("FALL ALERT: ");
  Serial.println(reason);
  Serial.println("LED = ON");
  Serial.println("BUZZER = ON");
  Serial.println("==============================");
}

void stop_alarm() {
  alarm_active = false;
  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
  Serial.println("ALARM: OFF");
}

void reset_detector() {
  freefall_detected = false;
  freefall_start_time = 0;
  stop_alarm();
  Serial.println("Detector state reset.");
}

void manual_alarm_test() {
  Serial.println("Manual LED/buzzer test started.");

  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    digitalWrite(BUZZER_PIN, HIGH);
    delay(300);
    digitalWrite(LED_PIN, LOW);
    digitalWrite(BUZZER_PIN, LOW);
    delay(300);
  }

  Serial.println("Manual LED/buzzer test finished.");
}

void print_help() {
  Serial.println();
  Serial.println("Serial commands:");
  Serial.println("  test   = blink LED and beep buzzer 3 times");
  Serial.println("  alarm  = force fall alert ON");
  Serial.println("  off    = turn alarm OFF");
  Serial.println("  fall   = simulate a possible fall event");
  Serial.println("  reset  = reset detector state");
  Serial.println("  status = show detector status");
  Serial.println("  help   = show commands");
  Serial.println();
}

void handle_serial_commands() {
  if (!Serial.available()) return;

  String command = Serial.readStringUntil('\n');
  command.trim();
  command.toLowerCase();

  if (command == "test" || command == "t") {
    manual_alarm_test();
  }
  else if (command == "alarm" || command == "a" || command == "fall" || command == "f") {
    start_alarm("SERIAL MANUAL TEST");
  }
  else if (command == "off" || command == "o") {
    stop_alarm();
  }
  else if (command == "reset" || command == "r") {
    reset_detector();
  }
  else if (command == "status" || command == "s") {
    Serial.print("Sensor ready: ");
    Serial.println(sensor_ready ? "YES" : "NO");
    Serial.print("Free-fall state: ");
    Serial.println(freefall_detected ? "ACTIVE" : "IDLE");
    Serial.print("Alarm state: ");
    Serial.println(alarm_active ? "ON" : "OFF");
  }
  else if (command == "help" || command == "h") {
    print_help();
  }
  else if (command.length() > 0) {
    Serial.print("Unknown command: ");
    Serial.println(command);
    print_help();
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(100);
  delay(1500);

  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);

  Serial.println();
  Serial.println("==============================");
  Serial.println("RULE-BASED FALL DETECTOR");
  Serial.println("AI MODEL: DISABLED");
  Serial.println("==============================");

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  Serial.println("SDA: GPIO 21");
  Serial.println("SCL: GPIO 22");
  Serial.println("MPU6050 address: 0x68");

  scan_i2c_bus();

  if (!initialize_mpu6050()) {
    Serial.println("ERROR: MPU6050 initialization failed");

    while (true) {
      digitalWrite(LED_PIN, HIGH);
      delay(250);
      digitalWrite(LED_PIN, LOW);
      delay(250);
    }
  }

  sensor_ready = true;

  Serial.println("MPU6050 READY");
  Serial.println("Rule: free-fall followed by impact/rotation");
  Serial.println("System ready.");

  print_help();
}

void loop() {
  handle_serial_commands();

  if (!sensor_ready) {
    delay(1000);
    return;
  }

  unsigned long now = millis();

  if (now - last_sample_time < SAMPLE_INTERVAL_MS) {
    return;
  }

  last_sample_time = now;

  float ax, ay, az, gx, gy, gz;

  if (!read_sensor(ax, ay, az, gx, gy, gz)) {
    Serial.println("ERROR: MPU6050 read failed");
    return;
  }

  float acceleration_magnitude = sqrt(
    ax * ax +
    ay * ay +
    az * az
  );

  float rotation_magnitude = sqrt(
    gx * gx +
    gy * gy +
    gz * gz
  );

  Serial.print("ax=");
  Serial.print(ax, 3);
  Serial.print(" ay=");
  Serial.print(ay, 3);
  Serial.print(" az=");
  Serial.print(az, 3);
  Serial.print(" gx=");
  Serial.print(gx, 1);
  Serial.print(" gy=");
  Serial.print(gy, 1);
  Serial.print(" gz=");
  Serial.print(gz, 1);
  Serial.print(" | A=");
  Serial.print(acceleration_magnitude, 3);
  Serial.print(" G | R=");
  Serial.print(rotation_magnitude, 1);
  Serial.println(" DPS");

  // Stage 1: possible free-fall/unloading
  if (acceleration_magnitude <= FREEFALL_THRESHOLD_G) {
    if (!freefall_detected) {
      freefall_detected = true;
      freefall_start_time = now;
      Serial.println("Possible free-fall detected; waiting for confirmation...");
    }
  }

  // Stage 2: confirm with impact or rapid rotation
  if (freefall_detected) {
    unsigned long freefall_duration =
      now - freefall_start_time;

    bool freefall_long_enough =
      freefall_duration >= FREEFALL_MIN_MS;

    bool within_confirmation_window =
      freefall_duration <= FALL_CONFIRM_WINDOW_MS;

    bool impact_detected =
      acceleration_magnitude >= IMPACT_THRESHOLD_G;

    bool rotation_detected =
      rotation_magnitude >= ROTATION_THRESHOLD_DPS;

    if (
      freefall_long_enough &&
      within_confirmation_window &&
      (impact_detected || rotation_detected) &&
      !alarm_active &&
      now - last_alarm_time >= ALARM_COOLDOWN_MS
    ) {
      start_alarm("FREE-FALL + IMPACT/ROTATION CONFIRMED");
      freefall_detected = false;
    }

    if (!within_confirmation_window) {
      freefall_detected = false;
      Serial.println("Fall confirmation window expired.");
    }
  }

  if (
    alarm_active &&
    now - alarm_start_time >= ALARM_DURATION_MS
  ) {
    stop_alarm();
  }
}
