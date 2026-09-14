#include <Arduino.h>
#include <Wire.h>
#include <math.h>

// Direct MPU6050 I2C test for ESP32 + GY-521
const uint8_t MPU6050_ADDR = 0x68;

const int SDA_PIN = 21;
const int SCL_PIN = 22;
const int BUZZER_PIN = 4;
const int LED_PIN = 2;

const float GRAVITY = 9.80665f;
const float GYRO_RAD_TO_DEG = 57.2957795f;
const float FALL_THRESHOLD_G = 2.5f;

const unsigned long SAMPLE_INTERVAL_MS = 20;
const unsigned long ALARM_DURATION_MS = 1000;
const unsigned long ALARM_COOLDOWN_MS = 5000;

unsigned long last_sample_time = 0;
unsigned long alarm_start_time = 0;

bool alarm_active = false;
bool sensor_ready = false;

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

bool check_i2c_device(uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission() == 0;
}

void scan_i2c_bus() {
  Serial.println("Scanning I2C bus...");

  int found = 0;

  for (uint8_t address = 1; address < 127; address++) {
    if (check_i2c_device(address)) {
      Serial.print("I2C device found at address: 0x");
      if (address < 16) Serial.print("0");
      Serial.println(address, HEX);
      found++;
    }
  }

  Serial.print("Total I2C devices found: ");
  Serial.println(found);
}

bool initialize_mpu6050_direct() {
  if (!check_i2c_device(MPU6050_ADDR)) {
    Serial.println("ERROR: I2C address 0x68 is not responding");
    return false;
  }

  uint8_t who_am_i = read_register(0x75);

  Serial.print("MPU6050 WHO_AM_I register: 0x");
  if (who_am_i < 16) Serial.print("0");
  Serial.println(who_am_i, HEX);

  // Wake the sensor from sleep mode
  write_register(0x6B, 0x00);
  delay(100);

  // Configure accelerometer: +/-16 g
  write_register(0x1C, 0x18);

  // Configure gyroscope: +/-2000 degree/s
  write_register(0x1B, 0x18);

  // Digital low-pass filter
  write_register(0x1A, 0x04);

  // Sample-rate divider: 1 kHz / (1 + 19) = 50 Hz
  write_register(0x19, 19);

  delay(100);

  uint8_t power_register = read_register(0x6B);

  Serial.print("Power register after wake-up: 0x");
  Serial.println(power_register, HEX);

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

  // +/-16 g = 2048 LSB/g
  sample[0] = (float)raw_ax / 2048.0f;
  sample[1] = (float)raw_ay / 2048.0f;
  sample[2] = (float)raw_az / 2048.0f;

  // +/-2000 degree/s = 16.4 LSB/(degree/s)
  sample[3] = ((float)raw_gx / 16.4f);
  sample[4] = ((float)raw_gy / 16.4f);
  sample[5] = ((float)raw_gz / 16.4f);

  return true;
}

void start_alarm(float magnitude) {
  if (alarm_active) return;

  alarm_active = true;
  alarm_start_time = millis();

  digitalWrite(LED_PIN, HIGH);
  digitalWrite(BUZZER_PIN, HIGH);

  Serial.println("==============================");
  Serial.println("BASIC ALARM: POSSIBLE FALL");
  Serial.print("Acceleration magnitude: ");
  Serial.print(magnitude, 3);
  Serial.println(" G");
  Serial.println("==============================");
}

void stop_alarm() {
  alarm_active = false;
  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
  Serial.println("ALARM: OFF");
}

void setup() {
  Serial.begin(115200);
  delay(1500);

  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);

  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);

  Serial.println();
  Serial.println("==============================");
  Serial.println("ESP32 FALL DETECTOR");
  Serial.println("DIRECT MPU6050 HARDWARE TEST");
  Serial.println("AI MODEL: DISABLED");
  Serial.println("==============================");

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  Serial.println("SDA: GPIO 21");
  Serial.println("SCL: GPIO 22");
  Serial.println("MPU6050 address: 0x68");

  scan_i2c_bus();

  if (!initialize_mpu6050_direct()) {
    Serial.println("ERROR: MPU6050 direct initialization failed");

    while (true) {
      digitalWrite(LED_PIN, HIGH);
      delay(250);
      digitalWrite(LED_PIN, LOW);
      delay(250);
    }
  }

  sensor_ready = true;

  Serial.println();
  Serial.println("MPU6050 DIRECT MODE READY");
  Serial.println("Accelerometer range: +/-16G");
  Serial.println("Gyroscope range: +/-2000 DPS");
  Serial.println("Sampling frequency: 50 Hz");
  Serial.println("AI model is disabled for this initial test");
  Serial.println("System ready.");
  Serial.println();
}

void loop() {
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

  if (
    magnitude >= FALL_THRESHOLD_G &&
    !alarm_active &&
    now - alarm_start_time >= ALARM_COOLDOWN_MS
  ) {
    start_alarm(magnitude);
  }

  if (
    alarm_active &&
    now - alarm_start_time >= ALARM_DURATION_MS
  ) {
    stop_alarm();
  }
}
