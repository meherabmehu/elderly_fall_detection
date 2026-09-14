# Elderly Fall Detection on ESP32 — Project Handoff Summary

## 1. Project identity

**Project name:** Elderly Fall Detection on ESP32

**Main idea:** A low-cost waist-worn inertial-sensor system for detecting dangerous fall-related movement and pre-impact alert states in older adults.

**Primary hardware:**
- ESP32 DevKit V1
- MPU6050 / GY-521
- Active buzzer
- Red LED with 220 ohm resistor
- USB power for the current physical setup

**Primary repository:**
`https://github.com/meherabmehu/elderly_fall_detection`

**Reference repository used for the final technical architecture:**
`https://github.com/arifshekhk8/preimpact-fall-detection-tinyml`

---

## 2. Project objective

The system should read six inertial channels from an MPU6050:

```text
ax, ay, az, gx, gy, gz
```

It should process the sensor stream on an ESP32 and identify:

```text
non-fall / normal movement
alert / pre-impact transition
fall / impact-related movement
```

When the model or rule-based detector reaches its operating condition, the system should activate:

```text
red LED
active buzzer
```

The device is intended as a research prototype for elderly fall detection and pre-impact warning. It is not a certified medical device.

---

## 3. Dataset and training work completed

Datasets studied and used in the project:

- SisFall Enhanced
- KFall
- FallAllD
- UMAFall

The Kaggle pipeline includes:

- dataset discovery and verification
- real-format dataset parsers
- channel harmonization
- resampling to 50 Hz
- windowing
- subject/grouped evaluation
- E1 within-dataset experiments
- E2 leave-one-dataset-out evaluation
- E3 pre-impact/lead-time evaluation
- E5 ablations
- FP32 training
- full INT8 quantization
- desktop verification of the INT8 model
- C header generation for embedded deployment

Frozen preprocessing contract:

```text
Sampling rate: 50 Hz
Window: 2 seconds
Window size: 100 x 6
Stride: 25 samples
Channels: ax, ay, az, gx, gy, gz
```

---

## 4. Important final-model decision

The reference repository's final deployment model uses:

```text
Per-window instance normalization
```

Normalization is performed in float before INT8 quantization. It is not kept as an in-graph normalization layer because in-graph instance normalization damaged full INT8 quantization.

This is critical:

```text
Training preprocessing and firmware preprocessing must remain identical.
```

Do not mix the older global-normalization firmware with the reference repository's final model.

Final reference model facts:

```text
Parameters: approximately 7,947
INT8 model size: approximately 22.54 KB
Estimated tensor arena: approximately 8 KB
FP32/INT8 agreement: approximately 99.44%
```

---

## 5. Results summary

### Cross-dataset generalization

Within-dataset learned-model macro-F1 was approximately:

```text
0.95–0.98
```

Leave-one-dataset-out performance was substantially lower and asymmetric:

```text
FallAllD unseen test: approximately 0.34 F1
KFall unseen test: approximately 0.84 F1
```

The key finding is that cross-dataset generalization is much harder than within-dataset evaluation suggests.

### Pre-impact results

Depending on the trial-level operating point:

```text
Sensitivity: approximately 0.91–0.96
Specificity: approximately 0.84–0.97
Mean lead time: approximately 539–722 ms
```

The threshold and consecutive-window parameter create a trade-off:

- higher threshold/k: fewer false alarms but lower sensitivity and shorter lead time
- lower threshold/k: more sensitivity and earlier alarms but more false alarms

### Deployment results

```text
INT8 model size: approximately 22.54 KB
Estimated tensor arena: approximately 8 KB
FP32 to INT8 accuracy change: within the project budget
```

Physical latency, peak RAM high-water mark, battery life, current, and sensor-to-buzzer latency still need to be measured on the real ESP32.

---

## 6. Physical hardware status

The USB-powered physical setup has been tested successfully.

Verified:

```text
MPU6050 I2C address: 0x68
SDA: GPIO 21 / D21
SCL: GPIO 22 / D22
MPU6050 VCC: ESP32 3V3
MPU6050 GND: ESP32 GND
MPU6050 AD0: GND
Buzzer: GPIO 4 / D4
LED: GPIO 2 / D2 through 220 ohm resistor
```

The physical MPU6050 initially failed with the Adafruit initialization path even though the I2C scanner found address 0x68. A direct-register MPU6050 reader was therefore created and successfully produced acceleration and gyroscope values.

Stationary readings were approximately:

```text
acceleration magnitude: about 1 G
```

---

## 7. AI-free baseline completed

The AI-free rule-based firmware was created and physically tested.

It uses:

```text
free-fall/unloading detection
impact detection
rapid rotation detection
LED output
buzzer output
serial manual commands
```

Manual commands include:

```text
test
alarm
off
status
reset
help
```

A safe demonstration mode was also created where a controlled board tilt can activate the alarm for presentation purposes.

---

## 8. AI integration completed

The final AI firmware was integrated into the ESP32/Wokwi workflow.

The AI runtime pipeline is:

```text
MPU6050 raw register reading
→ 50 Hz sampling
→ 100 x 6 rolling window
→ per-window instance normalization
→ INT8 quantization
→ TensorFlow Lite Micro inference
→ p_bkg / p_alert / p_fall
→ threshold and consecutive-window confirmation
→ LED/buzzer alarm
```

The physical AI firmware successfully loaded the model and printed output such as:

```text
p_bkg=0.066
p_alert=0.781
p_fall=0.156
```

It also successfully triggered:

```text
FALL ALERT: CNN: alert/fall probability sustained
```

---

## 9. Wokwi replay completed

A real KFall trial was extracted:

```text
S06T20R01.csv
SA06_label.xlsx
```

Trial information:

```text
Task: T20
Trial: R01
Fall onset frame: 130
Fall impact frame: 208
Native rate: 100 Hz
Replay rate: 50 Hz
```

The replay was converted into:

```text
replay_data.h
```

Wokwi successfully replayed the KFall trial through the trained INT8 model and produced high fall probabilities, for example:

```text
p_fall=0.8906
p_fall=0.9063
p_fall=0.9336
```

The buzzer and LED activated successfully.

---

## 10. Current USB hardware configuration

The project currently uses USB power only.

```text
USB cable → ESP32
```

No VIN, TP4056 or LiPo battery is currently required for the physical test.

The future battery circuit is optional:

```text
LiPo + → TP4056 B+
LiPo - → TP4056 B-
TP4056 OUT+ → ESP32 VIN
TP4056 OUT- → ESP32 GND
```

The battery circuit should be added only after USB-powered testing is stable.

---

## 11. Technical package contents

The complete ZIP package is:

```text
ELDERLY_FALL_DETECTION_COMPLETE.zip
```

Important locations inside it:

```text
repository/              organized project repository
technical_artifacts/     training, results and deployment source
firmware/                live and replay firmware
wokwi/                   Wokwi/PlatformIO project
models/                  trained model artifacts
reports/                 project report, BOM and pin documents
presentations/           English and Bangla presentation scripts
paper_support/           results, tables, citations and measurement procedure
```

Raw datasets are not included because of access and redistribution restrictions.

---

## 12. Next work required

The project is technically functional, but the following work remains for a strong conference submission:

1. Physical ESP32 inference latency measurement over 1,000 runs.
2. Tensor arena high-water measurement.
3. Sensor-to-buzzer end-to-end latency measurement.
4. Current draw and battery-life measurement.
5. Safe supervised human-activity trials.
6. False-alarm measurement during walking, sitting, standing and stairs.
7. Operating-point selection based on the intended claim.
8. Final figures and tables using only verified results.
9. Clear discussion of dataset limitations and simulated-fall limitations.
10. Do not invent hardware numbers; keep them marked pending until measured.

---

## 13. Recommended continuation order

```text
1. Keep the current USB physical hardware unchanged.
2. Run live AI firmware and collect serial output.
3. Use the measure command after at least 1,000 inferences.
4. Record latency, arena high-water mark and current.
5. Run safe normal-activity trials.
6. Run controlled soft-surface tests only.
7. Select the final threshold and consecutive-window value.
8. Update results and hardware measurement files.
9. Freeze the final firmware and model artifacts.
10. Prepare the conference manuscript from verified tables only.
```

---

## 14. Important warnings for future continuation

- Do not mix the older model with the reference repository's normalization logic.
- Do not use global normalization if the selected final model uses per-window instance normalization.
- Do not commit raw KFall data to a public repository.
- Do not claim physical latency or battery life before measuring them.
- Do not describe Wokwi replay as real-world validation.
- Keep the AI-free rule baseline and AI CNN mode available for comparison.
- Use USB power until the physical circuit is stable.
