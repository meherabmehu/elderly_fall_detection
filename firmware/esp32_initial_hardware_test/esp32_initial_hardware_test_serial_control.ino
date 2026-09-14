#include <Arduino.h>
#include <Wire.h>
#include <math.h>

// ESP32 + MPU6050 direct-register hardware test
// AI model is disabled in this version.
// Serial commands: alarm, off, test, status, help

const uint8_t MPU6050_ADDR = 0x68;

const int SDA_PIN = 21;
const int SCL_PIN = 22;
const int BUZZER_PIN = 4;
const int LED_PIN = 2;

const float FALL_THRESHOLD_G = 2.5f;

const unsigned long SAMPLE_INTERVAL_MS = 20;
const unsigned long ALARM_DURATION_MS = 1000;
const unsigned long ALARM_COOLDOWN_MS = 5000;

unsigned long last_sample_time = 0;
unsigned long alarm_start_time = 0;
unsigned long last_alarm_time = 0;

bool sensor_ready = false;
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

  if (Wire.endTransmission(false) != 0) {
    return 0xFF;
  }

  if (Wire.requestFrom(MPU6050_ADDR, (uint8_t)1, (uint8_t)true) != 1) {
    return 0xFF;
  }

  return Wire.read();
}

bool i2c_device_exists(uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission() == 0;
}

void scan_i2c_bus() {
  Serial.println("Scanning I2C bus...");

  int found = 0;

  for (uint8_t address = 1; address < 127; address++) {
    if (i2c_device_exists(address)) {
      Serial.print("I2C device found at address: 0x");

      if (address < 16) {
        Serial.print("0");
      }

      Serial.println(address, HEX);
      found++;
    }
  }

  Serial.print("Total I2C devices found: ");
  Serial.println(found);
}

bool initialize_mpu6050_direct() {
  if (!i2c_device_exists(MPU6050_ADDR)) {
    Serial.println("ERROR: MPU6050 is not responding at 0x68");
    return false;
  }

  uint8_t who_am_i = read_register(0x75);

  Serial.print("WHO_AM_I: 0x");
  if (who_am_i < 16) {
    Serial.print("0");
  }
  Serial.println(who_am_i, HEX);

  // Wake up MPU6050
  write_register(0x6B, 0x00);
  delay(100);

  // Accelerometer +/-16G
  write_register(0x1C, 0x18);

  // Gyroscope +/-2000 degree/second
  write_register(0x1B, 0x18);

  // Digital low-pass filter
  write_register(0x1A, 0x04);

  // 1 kHz / (1 + 19) = 50 Hz
  write_register(0x19, 19);
  delay(100);

  return true;
}

int16_t read_int16() {
  uint8_t high_byte = Wire.read();
  uint8_t low_byte = Wire.read();
  return (int16_t)((high_byte << 8) | low_byte);
}

bool read_mpu6050(float sample[6]) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(0x3B);

  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  if (Wire.requestFrom(MPU6050_ADDR, (uint8_t)14, (uint8_t)true) != 14) {
    return false;
  }

  int16_t raw_ax = read_int16();
  int16_t raw_ay = read_int16();
  int16_t raw_az = read_int16();
  int16_t raw_temperature = read_int16();
  int16_t raw_gx = read_int16();
  int16_t raw_gy = read_int16();
  int16_t raw_gz = read_int16();

  (void)raw_temperature;

  // +/-16G = 2048 LSB/G
  sample[0] = (float)raw_ax / 2048.0f;
  sample[1] = (float)raw_ay / 2048.0f;
  sample[2] = (float)raw_az / 2048.0f;

  // +/-2000 DPS = 16.4 LSB per DPS
  sample[3] = (float)raw_gx / 16.4f;
  sample[4] = (float)raw_gy / 16.4f;
  sample[5] = (float)raw_gz / 16.4f;

  return true;
}

void start_alarm(const char* reason, float magnitude = 0.0f) {
  alarm_active = true;
  alarm_start_time = millis();
  last_alarm_time = alarm_start_time;

  digitalWrite(LED_PIN, HIGH);
  digitalWrite(BUZZER_PIN, HIGH);

  Serial.println();
  Serial.println("==============================");
  Serial.print("ALARM STARTED: ");
  Serial.println(reason);

  if (magnitude > 0.0f) {
    Serial.print("Magnitude: ");
    Serial.print(magnitude, 3);
    Serial.println(" G");
  }

  Serial.println("LED = ON");
  Serial.println("BUZZER = ON");
  Serial.println("==============================");
}

void stop_alarm() {
  alarm_active = false;

  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  Serial.println("ALARM STOPPED");
}

void manual_test_alarm() {
  Serial.println("Manual alarm test started");

  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    digitalWrite(BUZZER_PIN, HIGH);
    delay(300);

    digitalWrite(LED_PIN, LOW);
    digitalWrite(BUZZER_PIN, LOW);
    delay(300);
  }

  Serial.println("Manual alarm test finished");
}

void print_help() {
  Serial.println();
  Serial.println("Available serial commands:");
  Serial.println("  alarm  - turn LED and buzzer ON");
  Serial.println("  off    - turn LED and buzzer OFF");
  Serial.println("  test   - beep and blink 3 times");
  Serial.println("  status - show current system status");
  Serial.println("  help   - show this help");
  Serial.println();
}

void print_status() {
  Serial.println();
  Serial.println("SYSTEM STATUS");
  Serial.print("Sensor ready: ");
  Serial.println(sensor_ready ? "YES" : "NO");
  Serial.print("Alarm active: ");
  Serial.println(alarm_active ? "YES" : "NO");
  Serial.println("AI model: DISABLED");
  Serial.println();
}

void handle_serial_commands() {
  if (!Serial.available()) {
    return;
  }

  String command = Serial.readStringUntil('\n');
  command.trim();
  command.toLowerCase();

  if (command == "alarm" || command == "a") {
    start_alarm("MANUAL SERIAL COMMAND");
  }
  else if (command == "off" || command == "o") {
    stop_alarm();
  }
  else if (command == "test" || command == "t") {
    manual_test_alarm();
  }
  else if (command == "status" || command == "s") {
    print_status();
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
  Serial.println("ESP32 FALL DETECTOR");
  Serial.println("INITIAL LIVE HARDWARE TEST");
  Serial.println("AI MODEL: DISABLED");
  Serial.println("==============================");

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  Serial.println("SDA: GPIO 21");
  Serial.println("SCL: GPIO 22");
  Serial.println("MPU6050 address: 0x68");

  scan_i2c_bus();

  if (!initialize_mpu6050_direct()) {
    Serial.println("ERROR: MPU6050 initialization failed");

    while (true) {
      digitalWrite(LED_PIN, HIGH);
      delay(250);
      digitalWrite(LED_PIN, LOW);
      delay(250);
    }
  }

  sensor_ready = true;

  Serial.println();
  Serial.println("MPU6050 READY");
  Serial.println("Sampling frequency: 50 Hz");
  Serial.println("Basic fall threshold: 2.5 G");
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

  float sample[6];

  if (!read_mpu6050(sample)) {
    Serial.println("ERROR: MPU6050 read failed");
    return;
  }

  float ax = sample[0];
  float ay = sample[1];
  float az = sample[2];
  float gx = sample[3];
  float gy = sample[4];
  float gz = sample[5];

  float magnitude = sqrt(
    ax * ax +
    ay * ay +
    az * az
  );

  Serial.print("ax=");
  Serial.print(ax, 3);
  Serial.print(" ay=");
  Serial.print(ay, 3);
  Serial.print(" az=");
  Serial.print(az, 3);
  Serial.print(" gx=");
  Serial.print(gx, 3);
  Serial.print(" gy=");
  Serial.print(gy, 3);
  Serial.print(" gz=");
  Serial.print(gz, 3);
  Serial.print(" magnitude=");
  Serial.println(magnitude, 3);

  // Basic non-AI threshold test
  if (
    magnitude >= FALL_THRESHOLD_G &&
    !alarm_active &&
    now - last_alarm_time >= ALARM_COOLDOWN_MS
  ) {
    start_alarm("ACCELERATION THRESHOLD", magnitude);
  }

  if (
    alarm_active &&
    now - alarm_start_time >= ALARM_DURATION_MS
  ) {
    stop_alarm();
  }
}
