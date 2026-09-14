/*
 * Final INT8 fall-detection firmware - Wokwi build with KFall replay support.
 *
 * This is the SAME runtime as firmware/esp32_ai_final_live_deployment:
 *
 *   50 Hz sampling -> 100x6 rolling window (stride 25)
 *   -> per-window instance normalisation in float
 *   -> INT8 quantise -> TFLite Micro inference
 *   -> p_bkg / p_alert / p_fall -> threshold + k-consecutive confirmation
 *   -> red LED + active buzzer
 *
 * Two sample sources, selected by REPLAY_MODE:
 *
 *   REPLAY_MODE 1  KFall trial S06T20R01 is injected at the RAW-COUNT level
 *                  (values from replay_data.h are converted to the int16 counts
 *                  an MPU6050 at +/-16 g / +/-2000 dps would report), so the
 *                  replay exercises exactly the same code path as live data.
 *                  The trial loops; a summary is printed at the end of each pass.
 *
 *   REPLAY_MODE 0  live MPU6050 on D21/D22 (also works in Wokwi with the
 *                  wokwi-mpu6050 part at 0x68).
 *
 * Trial ground truth (KFall label file SA06_label.xlsx):
 *   onset frame 130, impact frame 208 at native 100 Hz
 *   -> at the 50 Hz working rate: onset sample 65, impact sample 104.
 *   Every alarm prints its lead time against sample 104
 *   (positive = alarm fired BEFORE the labelled impact).
 *
 * IMPORTANT - normalisation contract:
 *   The final model expects PER-WINDOW instance normalisation, computed in float
 *   BEFORE quantisation. It must stay identical to fdlib.preprocess.instance_normalise
 *   (mean per channel over the 100-sample window, divide by std + 1e-6).
 *   It does NOT use the frozen global constants in norm_constants.h.
 *
 * Works on both ESP32 Arduino core 2.x and 3.x timer APIs (see setup()).
 */

#include <Arduino.h>
#include <Wire.h>
#include <math.h>

#include "model.h"       // final INT8 model (23,080 bytes, signature 2574e6104c95)
#include "replay_data.h" // KFall S06T20R01, 268 samples x 6 channels, 50 Hz, g / dps

// The deployed (physically verified) Arduino sketch builds against the legacy
// TensorFlowLite_ESP32 library; Wokwi/PlatformIO builds typically resolve the
// newer tflite-micro-arduino-examples library, which renamed the umbrella
// header and removed MicroErrorReporter. Support both, identically.
#ifdef __has_include
#  if __has_include(<TensorFlowLite_ESP32.h>)
#    include <TensorFlowLite_ESP32.h>
#  elif __has_include(<TensorFlowLite.h>)
#    include <TensorFlowLite.h>
#  else
#    error "Install a TensorFlow Lite Micro Arduino library (see wokwi/README.md)"
#  endif
#else
#  include <TensorFlowLite_ESP32.h>
#endif

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Newer tflite-micro removed MicroErrorReporter; the legacy TensorFlowLite_ESP32
// library still requires it in the interpreter constructor.
#ifdef __has_include
#  if __has_include("tensorflow/lite/micro/micro_error_reporter.h")
#    include "tensorflow/lite/micro/micro_error_reporter.h"
#    define FD_TFLM_LEGACY_API 1
#  else
#    define FD_TFLM_LEGACY_API 0
#  endif
#else
#  include "tensorflow/lite/micro/micro_error_reporter.h"
#  define FD_TFLM_LEGACY_API 1
#endif

#ifndef REPLAY_MODE
#define REPLAY_MODE 1
#endif

// ------------------------------------------------------------------ contract
// These MUST match fdlib.config. A mismatch is silent and fatal.
static const int kSampleRateHz = 50;
static const int kWindowLen    = 100;   // 2.0 s
static const int kStride       = 25;    // 0.5 s
static const int kChannels     = 6;
static const int kNumClasses   = 3;     // bkg, alert, fall

// KFall S06T20R01 ground truth at the 50 Hz working rate.
static const int kImpactSample = 104;   // resample_index(208, 100 Hz, 50 Hz)
static const int kOnsetSample  = 65;    // resample_index(130, 100 Hz, 50 Hz)

// Operating point - identical defaults to the final live firmware.
//   DEFAULT (0.95, k=2): demo-robust; sens 0.886 / spec 0.975 / mean lead 302 ms
//   PRE-IMPACT OPTIMAL (0.95, k=1): sens 0.958 / spec 0.836 / mean lead 539 ms
//   Change live over serial: `thr 0.7`, `k 4` (trial-level sweep: models/final_int8/final_model_card.md)
static float   gThreshold   = 0.95f;
static int     gAgreeK      = 2;
static const uint32_t kCooldownMs   = 5000;
static const uint32_t kAlarmHoldMs  = 3000;
static const uint32_t kReplayGapMs  = 5000;   // pause between trial passes

// ------------------------------------------------------------------- pins
static const uint8_t kMpuAddr   = 0x68;
static const int     kPinSDA    = 21;
static const int     kPinSCL    = 22;
static const int     kPinBuzzer = 4;
static const int     kPinLed    = 2;

// +/-16 g and +/-2000 dps, as configured below and as training assumes
static const float kAccelScale = 1.0f / 2048.0f;
static const float kGyroScale  = 1.0f / 16.4f;

// --------------------------------------------------------- rule-based baseline
static const float FREEFALL_THRESHOLD_G   = 0.65f;
static const float IMPACT_THRESHOLD_G     = 2.50f;
static const float ROTATION_THRESHOLD_DPS = 250.0f;
static const uint32_t FREEFALL_MIN_MS        = 80;
static const uint32_t FALL_CONFIRM_WINDOW_MS = 1500;

// ---------------------------------------------------------------- ring buffer
static volatile int16_t  gRing[kWindowLen][kChannels];
static volatile uint32_t gWriteIdx   = 0;
static volatile uint32_t gNewSamples = 0;
static volatile bool     gSampleDue  = false;
static hw_timer_t *gTimer = nullptr;

// ------------------------------------------------------------------- TFLite
constexpr int kArenaSize = 24 * 1024;   // desktop estimate ~8 KB; measure, then tune down
static uint8_t gArena[kArenaSize];

#if FD_TFLM_LEGACY_API
static tflite::MicroErrorReporter gErrorReporter;
#endif
static tflite::MicroInterpreter  *gInterpreter = nullptr;
static TfLiteTensor              *gInput  = nullptr;
static TfLiteTensor              *gOutput = nullptr;

// ------------------------------------------------------------- C3 measurement
static uint64_t gLatencySumUs = 0;
static uint64_t gLatencyMinUs = UINT64_MAX;
static uint64_t gLatencyMaxUs = 0;
static uint32_t gInferenceCount = 0;
static uint64_t gPrepSumUs = 0;       // window copy + normalise + quantise
static uint32_t gArenaHighWater = 0;

// ------------------------------------------------------------------- state
enum Mode { MODE_CNN, MODE_RULE, MODE_BOTH };
static Mode gMode = MODE_CNN;

static bool     gAlarmActive   = false;
static uint32_t gAlarmStartMs  = 0;
static uint32_t gLastAlarmMs   = 0;
static int      gConsecutive   = 0;
static uint32_t gCnnAlarms = 0, gRuleAlarms = 0;

static bool     gFreefallDetected = false;
static uint32_t gFreefallStartMs  = 0;

static float gLastP[kNumClasses] = {0, 0, 0};

// ----------------------------------------------------------------- replay
static uint32_t gReplayIdx     = 0;
static bool     gReplayHolding = false;
static uint32_t gReplayHoldStart = 0;
static uint32_t gReplayPass    = 0;
static uint32_t gPassAlarms    = 0;
static int32_t  gPassFirstAlarmEdge = -1;

// ----------------------------------------------------------------- MPU6050
static void writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(kMpuAddr);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission(true);
}

static uint8_t readRegister(uint8_t reg) {
  Wire.beginTransmission(kMpuAddr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return 0xFF;
  if (Wire.requestFrom((uint8_t)kMpuAddr, (uint8_t)1, (uint8_t)true) != 1) return 0xFF;
  return Wire.read();
}

static bool i2cExists(uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission() == 0;
}

static bool initialiseMpu6050() {
  if (!i2cExists(kMpuAddr)) return false;
  Serial.print("WHO_AM_I: 0x");
  Serial.println(readRegister(0x75), HEX);

  writeRegister(0x6B, 0x00);  // wake
  writeRegister(0x1C, 0x18);  // ACCEL_CONFIG: +/-16 g
  writeRegister(0x1B, 0x18);  // GYRO_CONFIG:  +/-2000 dps
  writeRegister(0x1A, 0x04);  // DLPF -- analogue anti-alias ahead of the 50 Hz sampler
  writeRegister(0x19, 19);    // sample rate 1000/(1+19) = 50 Hz
  delay(100);
  return true;
}

static int16_t readInt16() {
  uint8_t hi = Wire.read();
  uint8_t lo = Wire.read();
  return (int16_t)((hi << 8) | lo);
}

static bool readSensorRaw(int16_t *out6) {
  Wire.beginTransmission(kMpuAddr);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint8_t)kMpuAddr, (uint8_t)14, (uint8_t)true) != 14) return false;
  out6[0] = readInt16();  // ax
  out6[1] = readInt16();  // ay
  out6[2] = readInt16();  // az
  (void)readInt16();      // temperature, discarded
  out6[3] = readInt16();  // gx
  out6[4] = readInt16();  // gy
  out6[5] = readInt16();  // gz
  return true;
}

// ------------------------------------------------------------------ replay
// Convert one replay sample (g / dps, 50 Hz) into the int16 counts an MPU6050
// in this firmware's configuration would produce for the same signal, so the
// replay and live paths are identical downstream of the sensor read.
static bool readReplayRaw(int16_t *out6) {
  if (gReplayHolding) {
    if (millis() - gReplayHoldStart >= kReplayGapMs) {
      gReplayHolding = false;
      gReplayIdx = 0;
      gWriteIdx = 0;
      gNewSamples = 0;
      gConsecutive = 0;
      gPassAlarms = 0;
      gPassFirstAlarmEdge = -1;
      Serial.println();
      Serial.printf("==== REPLAY RESTART (pass %u) ====\n", (unsigned)(gReplayPass + 1));
    } else {
      return false;   // inter-trial pause; nothing sampled
    }
  }
  for (int c = 0; c < kChannels; c++) {
    float scale = (c < 3) ? kAccelScale : kGyroScale;
    long raw = lroundf(replay_data[gReplayIdx][c] / scale);
    if (raw > 32767) raw = 32767;
    if (raw < -32768) raw = -32768;
    out6[c] = (int16_t)raw;
  }
  gReplayIdx++;
  if (gReplayIdx >= (uint32_t)REPLAY_SAMPLES) {
    gReplayHolding = true;
    gReplayHoldStart = millis();
    gReplayPass++;
    Serial.println();
    Serial.println("==== REPLAY TRIAL COMPLETE ====");
    Serial.printf("dataset          : KFall S06T20R01 (%d samples @ 50 Hz)\n", REPLAY_SAMPLES);
    Serial.printf("ground truth     : onset sample %d, impact sample %d\n", kOnsetSample, kImpactSample);
    Serial.printf("threshold / k    : %.2f / %d\n", gThreshold, gAgreeK);
    Serial.printf("alarms this pass : %u\n", (unsigned)gPassAlarms);
    if (gPassFirstAlarmEdge >= 0) {
      Serial.printf("first alarm edge : %d  ->  lead time %d ms %s impact\n",
                    (int)gPassFirstAlarmEdge,
                    (int)abs(kImpactSample - gPassFirstAlarmEdge) * (1000 / kSampleRateHz),
                    (gPassFirstAlarmEdge <= kImpactSample) ? "BEFORE" : "AFTER");
    }
    Serial.printf("restarting in %u s...\n", (unsigned)(kReplayGapMs / 1000));
  }
  return true;
}

// --------------------------------------------------------------------- ISR
void IRAM_ATTR onSampleTimer() { gSampleDue = true; }

// ------------------------------------------------------------------ alarm
static void alarmLeadLine() {
  if (gWriteIdx == 0) return;
  int32_t rightEdge = (int32_t)(gWriteIdx - 1);
  Serial.printf("alarm right edge : %d  ->  lead time %d ms %s impact\n",
                (int)rightEdge,
                (int)abs(kImpactSample - rightEdge) * (1000 / kSampleRateHz),
                (rightEdge <= kImpactSample) ? "BEFORE" : "AFTER");
}

static void startAlarm(const char *reason) {
  if (gAlarmActive) return;
  uint32_t now = millis();
  if (now - gLastAlarmMs < kCooldownMs) return;
  gAlarmActive  = true;
  gAlarmStartMs = now;
  gLastAlarmMs  = now;
  digitalWrite(kPinLed, HIGH);
  digitalWrite(kPinBuzzer, HIGH);
  Serial.println("========================================");
  Serial.print("FALL ALERT: ");
  Serial.println(reason);
  Serial.printf("p_bkg=%.3f p_alert=%.3f p_fall=%.3f\n", gLastP[0], gLastP[1], gLastP[2]);
#if REPLAY_MODE
  alarmLeadLine();
#endif
  Serial.println("========================================");
}

static void stopAlarm() {
  gAlarmActive = false;
  digitalWrite(kPinLed, LOW);
  digitalWrite(kPinBuzzer, LOW);
  Serial.println("ALARM: OFF");
}

// ------------------------------------------------------------ rule baseline
static void ruleStep(float accMag, float rotMag, uint32_t now) {
  if (accMag <= FREEFALL_THRESHOLD_G && !gFreefallDetected) {
    gFreefallDetected = true;
    gFreefallStartMs  = now;
  }
  if (!gFreefallDetected) return;

  uint32_t dur = now - gFreefallStartMs;
  bool longEnough = dur >= FREEFALL_MIN_MS;
  bool inWindow   = dur <= FALL_CONFIRM_WINDOW_MS;
  bool impact     = accMag >= IMPACT_THRESHOLD_G;
  bool rotation   = rotMag >= ROTATION_THRESHOLD_DPS;

  if (longEnough && inWindow && (impact || rotation)) {
    gRuleAlarms++;
    if (gMode == MODE_RULE || gMode == MODE_BOTH) {
      startAlarm("RULE: free-fall + impact/rotation");
    } else {
      Serial.println("[rule] would fire (mode=cnn)");
    }
    gFreefallDetected = false;
  } else if (!inWindow) {
    gFreefallDetected = false;
  }
}

// -------------------------------------------------------------- CNN inference
static void runInference() {
  const float inScale = gInput->params.scale;
  const int   inZero  = gInput->params.zero_point;
  const uint32_t start = gWriteIdx % kWindowLen;

  static float win[kWindowLen][kChannels];
  int64_t tp0 = esp_timer_get_time();

  // pass 1 -- oldest-first copy, raw counts to g and deg/s
  for (int t = 0; t < kWindowLen; t++) {
    uint32_t r = (start + t) % kWindowLen;
    for (int c = 0; c < kChannels; c++) {
      win[t][c] = (float)gRing[r][c] * (c < 3 ? kAccelScale : kGyroScale);
    }
  }

  // passes 2 and 3 -- per-window instance normalisation, then quantise.
  // Must stay identical to fdlib.preprocess.instance_normalise.
  for (int c = 0; c < kChannels; c++) {
    float mean = 0.0f;
    for (int t = 0; t < kWindowLen; t++) mean += win[t][c];
    mean /= (float)kWindowLen;

    float var = 0.0f;
    for (int t = 0; t < kWindowLen; t++) {
      const float d = win[t][c] - mean;
      var += d * d;
    }
    var /= (float)kWindowLen;
    const float inv = 1.0f / (sqrtf(var) + 1e-6f);   // matches the Python eps

    for (int t = 0; t < kWindowLen; t++) {
      int q = (int)lroundf(((win[t][c] - mean) * inv) / inScale) + inZero;
      if (q < -128) q = -128;
      if (q >  127) q =  127;
      gInput->data.int8[t * kChannels + c] = (int8_t)q;
    }
  }
  gPrepSumUs += (uint64_t)(esp_timer_get_time() - tp0);

  int64_t t0 = esp_timer_get_time();
  if (gInterpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Invoke failed");
    return;
  }
  int64_t t1 = esp_timer_get_time();

  uint64_t us = (uint64_t)(t1 - t0);
  gLatencySumUs += us;
  if (us < gLatencyMinUs) gLatencyMinUs = us;
  if (us > gLatencyMaxUs) gLatencyMaxUs = us;
  gInferenceCount++;
  uint32_t used = (uint32_t)gInterpreter->arena_used_bytes();
  if (used > gArenaHighWater) gArenaHighWater = used;

  const float outScale = gOutput->params.scale;
  const int   outZero  = gOutput->params.zero_point;
  for (int i = 0; i < kNumClasses; i++) {
    gLastP[i] = ((int)gOutput->data.int8[i] - outZero) * outScale;
  }
  const float pAlarm = gLastP[1] + gLastP[2];   // alert + fall, as in the Python metrics

#if REPLAY_MODE
  Serial.printf("window @ edge %u  p_bkg=%.4f p_alert=%.4f p_fall=%.4f\n",
                (unsigned)(gWriteIdx - 1), gLastP[0], gLastP[1], gLastP[2]);
#endif

  // k-consecutive confirmation. A genuine fall spans several overlapping windows;
  // isolated false positives do not.
  if (pAlarm >= gThreshold) {
    gConsecutive++;
  } else {
    gConsecutive = 0;
  }
  if (gConsecutive >= gAgreeK) {
    gConsecutive = 0;
    gCnnAlarms++;
    gPassAlarms++;
    if (gPassFirstAlarmEdge < 0) gPassFirstAlarmEdge = (int32_t)(gWriteIdx - 1);
    if (gMode == MODE_CNN || gMode == MODE_BOTH) {
      startAlarm("CNN: alert/fall probability sustained");
    } else {
      Serial.println("[cnn] would fire (mode=rule)");
    }
  }
}

// ------------------------------------------------------------ serial commands
static void printMeasure() {
  Serial.println("---- MEASURE ----");
  Serial.printf("inferences        : %u\n", gInferenceCount);
  if (gInferenceCount) {
    Serial.printf("invoke mean       : %.3f ms\n",
                  (double)gLatencySumUs / gInferenceCount / 1000.0);
    Serial.printf("invoke min/max    : %.3f / %.3f ms\n",
                  gLatencyMinUs / 1000.0, gLatencyMaxUs / 1000.0);
    Serial.printf("preprocess mean   : %.3f ms\n",
                  (double)gPrepSumUs / gInferenceCount / 1000.0);
    Serial.printf("end-to-end mean   : %.3f ms\n",
                  ((double)gPrepSumUs + gLatencySumUs) / gInferenceCount / 1000.0);
  }
  Serial.printf("arena high water  : %u bytes (%.1f KB) of %u\n",
                gArenaHighWater, gArenaHighWater / 1024.0, (unsigned)kArenaSize);
  Serial.printf("model flash       : %u bytes (%.1f KB)\n",
                g_model_len, g_model_len / 1024.0);
  Serial.printf("free heap         : %u bytes\n", (unsigned)ESP.getFreeHeap());
  Serial.printf("cnn alarms        : %u\n", gCnnAlarms);
  Serial.printf("rule alarms       : %u\n", gRuleAlarms);
  Serial.printf("threshold / k     : %.2f / %d\n", gThreshold, gAgreeK);
  Serial.println("-----------------");
}

static void printHelp() {
  Serial.println();
  Serial.println("Serial commands:");
  Serial.println("  test      = blink LED and beep buzzer 3 times");
  Serial.println("  alarm     = force the alarm on");
  Serial.println("  off       = turn the alarm off");
  Serial.println("  reset     = reset detector state and counters");
  Serial.println("  status    = sensor and detector status");
  Serial.println("  measure   = latency, arena high-water mark, alarm counts");
  Serial.println("  mode cnn|rule|both = which detector fires the alarm");
  Serial.println("  thr <0..1>  = CNN decision threshold");
  Serial.println("  k <1..8>    = consecutive windows required before firing");
  Serial.println("  help      = this list");
  Serial.println();
}

static void manualTest() {
  Serial.println("Manual LED/buzzer test started.");
  for (int i = 0; i < 3; i++) {
    digitalWrite(kPinLed, HIGH); digitalWrite(kPinBuzzer, HIGH); delay(300);
    digitalWrite(kPinLed, LOW);  digitalWrite(kPinBuzzer, LOW);  delay(300);
  }
  Serial.println("Manual LED/buzzer test finished.");
}

static void handleSerial() {
  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  cmd.toLowerCase();

  if (cmd == "test" || cmd == "t")        { manualTest(); }
  else if (cmd == "alarm" || cmd == "a")  { gLastAlarmMs = 0; startAlarm("SERIAL MANUAL TEST"); }
  else if (cmd == "off" || cmd == "o")    { stopAlarm(); }
  else if (cmd == "reset" || cmd == "r")  {
    gFreefallDetected = false; gConsecutive = 0;
    gInferenceCount = 0; gLatencySumUs = 0; gPrepSumUs = 0;
    gLatencyMinUs = UINT64_MAX; gLatencyMaxUs = 0;
    gCnnAlarms = gRuleAlarms = 0; gArenaHighWater = 0;
    stopAlarm();
    Serial.println("Detector state and counters reset.");
  }
  else if (cmd == "status" || cmd == "s") {
    Serial.printf("mode         : %s\n",
                  gMode == MODE_CNN ? "cnn" : gMode == MODE_RULE ? "rule" : "both");
    Serial.printf("alarm        : %s\n", gAlarmActive ? "ON" : "OFF");
    Serial.printf("windows seen : %u\n", gInferenceCount);
#if REPLAY_MODE
    Serial.printf("replay sample: %u / %d\n", (unsigned)gReplayIdx, REPLAY_SAMPLES);
#endif
  }
  else if (cmd == "measure" || cmd == "m") { printMeasure(); }
  else if (cmd.startsWith("mode ")) {
    String m = cmd.substring(5); m.trim();
    if (m == "cnn")       { gMode = MODE_CNN;  Serial.println("mode = cnn"); }
    else if (m == "rule") { gMode = MODE_RULE; Serial.println("mode = rule"); }
    else if (m == "both") { gMode = MODE_BOTH; Serial.println("mode = both"); }
    else Serial.println("mode must be cnn, rule or both");
  }
  else if (cmd.startsWith("thr ")) {
    gThreshold = cmd.substring(4).toFloat();
    Serial.printf("threshold = %.2f\n", gThreshold);
  }
  else if (cmd.startsWith("k ")) {
    gAgreeK = cmd.substring(2).toInt();
    if (gAgreeK < 1) gAgreeK = 1;
    Serial.printf("k = %d\n", gAgreeK);
  }
  else if (cmd == "help" || cmd == "h") { printHelp(); }
  else if (cmd.length() > 0) {
    Serial.print("Unknown command: "); Serial.println(cmd);
    printHelp();
  }
}

// ------------------------------------------------------------------- setup
void setup() {
  Serial.begin(115200);
  Serial.setTimeout(100);
  delay(1500);

  pinMode(kPinBuzzer, OUTPUT);
  pinMode(kPinLed, OUTPUT);
  digitalWrite(kPinBuzzer, LOW);
  digitalWrite(kPinLed, LOW);

  Serial.println();
  Serial.println("========================================");
  Serial.println("PRE-IMPACT FALL DETECTOR -- FINAL INT8 MODEL");
#if REPLAY_MODE
  Serial.println("MODE: KFALL S06T20R01 REPLAY");
#else
  Serial.println("MODE: LIVE MPU6050");
#endif
  Serial.println("========================================");

  Wire.begin(kPinSDA, kPinSCL);
  Wire.setClock(400000);

  if (!initialiseMpu6050()) {
#if REPLAY_MODE
    Serial.println("WARNING: no MPU6050 found - replay mode continues without a sensor");
#else
    Serial.println("ERROR: MPU6050 initialisation failed");
    while (true) {
      digitalWrite(kPinLed, HIGH); delay(250);
      digitalWrite(kPinLed, LOW);  delay(250);
    }
#endif
  } else {
    Serial.println("MPU6050 READY (+/-16 g, +/-2000 dps, DLPF on, 50 Hz)");
  }

  const tflite::Model *model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("FATAL: model schema mismatch");
    while (true) delay(1000);
  }
  static tflite::AllOpsResolver resolver;
#if FD_TFLM_LEGACY_API
  static tflite::MicroInterpreter interpreter(model, resolver, gArena, kArenaSize,
                                              &gErrorReporter);
#else
  static tflite::MicroInterpreter interpreter(model, resolver, gArena, kArenaSize);
#endif
  gInterpreter = &interpreter;
  if (gInterpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("FATAL: AllocateTensors failed -- increase kArenaSize");
    while (true) delay(1000);
  }
  gInput  = gInterpreter->input(0);
  gOutput = gInterpreter->output(0);

  Serial.printf("model flash    : %u bytes (%.1f KB)\n", g_model_len, g_model_len / 1024.0);
  Serial.printf("arena at init  : %u of %u bytes\n",
                (unsigned)gInterpreter->arena_used_bytes(), (unsigned)kArenaSize);
  Serial.printf("input  scale %.8f zero_point %d\n",
                gInput->params.scale, gInput->params.zero_point);
  Serial.printf("output scale %.8f zero_point %d\n",
                gOutput->params.scale, gOutput->params.zero_point);
  Serial.println("normalisation  : PER-WINDOW instance norm (not frozen constants)");

#if REPLAY_MODE
  Serial.println();
  Serial.println("REPLAY MODE ENABLED");
  Serial.println("Dataset: KFall");
  Serial.println("Trial: S06T20R01");
  Serial.printf("Replay samples: %d @ 50 Hz\n", REPLAY_SAMPLES);
  Serial.printf("Ground truth: onset sample %d, impact sample %d\n", kOnsetSample, kImpactSample);
#endif

  // 50 Hz hardware timer -- not millis() polling, which lets jitter accumulate.
#if defined(ESP_ARDUINO_VERSION_MAJOR) && (ESP_ARDUINO_VERSION_MAJOR >= 3)
  // ESP32 Arduino core 3.x: timer frequency 1 MHz, one tick = 1 us.
  gTimer = timerBegin(1000000);
  timerAttachInterrupt(gTimer, &onSampleTimer);
  timerAlarm(gTimer, 1000000 / kSampleRateHz, true, 0);
#else
  // ESP32 Arduino core 2.x: 80 prescaler -> 1 MHz tick.
  gTimer = timerBegin(0, 80, true);
  timerAttachInterrupt(gTimer, &onSampleTimer, true);
  timerAlarmWrite(gTimer, 1000000 / kSampleRateHz, true);
  timerAlarmEnable(gTimer);
#endif

  Serial.println("System ready.");
  printHelp();
}

// -------------------------------------------------------------------- loop
void loop() {
  handleSerial();

  if (gAlarmActive && (millis() - gAlarmStartMs >= kAlarmHoldMs)) stopAlarm();

  if (gSampleDue) {
    gSampleDue = false;
    int16_t s[kChannels];
#if REPLAY_MODE
    bool got = readReplayRaw(s);
#else
    bool got = readSensorRaw(s);
#endif
    if (got) {
      uint32_t idx = gWriteIdx % kWindowLen;
      for (int c = 0; c < kChannels; c++) gRing[idx][c] = s[c];
      gWriteIdx++;
      gNewSamples++;

      // rule baseline runs every sample, on the same data as the CNN
      const float ax = s[0] * kAccelScale, ay = s[1] * kAccelScale, az = s[2] * kAccelScale;
      const float gx = s[3] * kGyroScale,  gy = s[4] * kGyroScale,  gz = s[5] * kGyroScale;
      ruleStep(sqrtf(ax * ax + ay * ay + az * az),
               sqrtf(gx * gx + gy * gy + gz * gz), millis());
    }
  }

  if (gNewSamples >= (uint32_t)kStride && gWriteIdx >= (uint32_t)kWindowLen) {
    gNewSamples = 0;
    runInference();
    if (gInferenceCount % 1000 == 0) printMeasure();
  }
}
