# Hardware guide

Everything needed to assemble, verify and test the physical device.

## 1. Bill of materials (verified build, USB-powered)

| Component | Qty | Est. price (BDT, Dhaka) | Purpose |
|---|---:|---:|---|
| ESP32 DevKit V1 (30-pin) | 1 | 450 | main MCU |
| MPU6050 / GY-521 IMU | 2 | 400 | waist sensor + spare |
| Active buzzer | 1 | 40 | alarm audio |
| Red LED + 220 Ω resistor | 1 | 20 | alarm visual |
| Breadboard + jumper wires + headers | 1 set | 190 | prototyping |
| **Total (verified build)** | | **~1,100** | USB cable assumed available |

Optional / next step (battery version):

| Component | Qty | Est. price (BDT) | Purpose |
|---|---:|---:|---|
| TP4056 charger with protection | 1 | 100 | LiPo charge/protection |
| 3.7 V 1000 mAh LiPo | 1 | 250 | portable power + battery test |
| Elastic belt/Velcro | 1 | 150 | repeatable waist mounting |
| USB current meter | 1 | 350 | direct power measurement |
| ABS enclosure | 1 | 200 | final wearable build |

Full BOM document: `hardware/esp32_fall_detector_minimum_bom.docx`.

## 2. Pin map (single source of truth)

| Component | Pin | ESP32 | Notes |
|---|---|---|---|
| MPU6050 VCC | — | 3V3 | **3.3 V only — never 5 V** |
| MPU6050 GND | — | GND | common ground |
| MPU6050 SDA | — | GPIO 21 / D21 | I²C data |
| MPU6050 SCL | — | GPIO 22 / D22 | I²C clock, run at 400 kHz |
| MPU6050 AD0 | — | GND | selects address `0x68` |
| MPU6050 INT/XDA/XCL | — | (unconnected) | unused by all firmware here |
| Buzzer + | — | GPIO 4 / D4 | use a transistor if the buzzer draws ≳12 mA |
| Buzzer − | — | GND | |
| LED anode | — | GPIO 2 / D2 through 220 Ω | series resistor mandatory |
| LED cathode | — | GND | shorter leg / flat side |

All grounds must be common. Diagrams:
`hardware/esp32_fall_detector_pin_diagram.svg`,
`hardware/realistic_esp32_wiring_annotated.png`,
`hardware/esp32_mpu6050_breadboard_setup.png`,
`hardware/wiring_diagram.png`, connection sheets
`hardware/complete_pin_connections.docx` (full) and
`hardware/usb_direct_pin_connections.docx` (USB build).

## 3. Power

- **Verified build: USB only.** `USB port → ESP32`. No VIN/TP4056/LiPo.
- Optional battery circuit (add only after USB testing is stable):

```text
LiPo + -> TP4056 B+        TP4056 OUT+ -> ESP32 VIN (5V pin — verify YOUR board)
LiPo - -> TP4056 B-        TP4056 OUT- -> ESP32 GND
```

Use a protected TP4056 module. Never connect a LiPo directly to 3V3; never
reverse polarity.

## 4. Bring-up sequence (the order that worked)

1. **Wire per §2**, double-check 3V3 (not 5 V) and common grounds.
2. Flash `firmware/esp32_initial_hardware_test/` — confirm the I²C scan finds
   `0x68`, read WHO_AM_I (`0x68`), watch `ax ay az gx gy gz` on serial; at
   rest the acceleration vector magnitude should sit ≈ 1 g.
3. If the Adafruit `imu.begin()` path fails although the scanner sees 0x68
   (it did on our build), use register-level init — see
   `firmware/esp32_mpu6050_direct_test/`. All final firmware uses the
   register-level path for exactly this reason.
4. Flash `firmware/esp32_usb_rule_based_fall_detector/`; type `test`, `status`;
   move the board & confirm free-fall→impact detection fires.
5. Flash the final AI firmware per `docs/DEPLOYMENT.md`; pass the stationary
   sanity gate (`p_bkg` dominant, no alarm at rest).
6. Optional demo mode for presentations:
   `firmware/esp32_fall_detector_demo_final/` (tilt-to-trigger; clearly a
   demo, not a detection claim).

## 5. Mounting

Waist/low-back position matching the training corpora. The device is
orientation-sensitive in the sense that the model expects the canonical axis
convention learned from the datasets (gravity on −Y); per-window instance
normalisation absorbs static DC offsets; genuinely different placements (neck,
wrist) degrade accuracy — measured in E5 ablations (waist 0.769 vs wrist
0.732 vs neck 0.815 macro-F1 on FallAllD).

## 6. Safety and test protocol

- Do **not** perform real falls for testing. Mattress drops of the *device*
  (not a person) and the MEASUREMENT.md §4b protocol are the designed
  verification.
- The system is a research prototype, not a medical device; never its sole
  safeguard for a fall-risk person.
- Add the TP4056 + LiPo only after USB-powered tests are stable; disconnect
  the buzzer during battery-life runs (it dominates the draw otherwise).
