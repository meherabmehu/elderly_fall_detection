# firmware/ — ESP32 sketches

Pin map and power: `docs/HARDWARE.md`. Flashing walkthrough and the
**final supported deployment combination**: `docs/DEPLOYMENT.md`.

## Quick map

| Folder | What | Verified |
|---|---|---|
| `esp32_ai_final_live_deployment/` | **Final firmware**: register-level MPU6050 @ 50 Hz, TFLite Micro INT8 CNN + rule baseline, hardware timer, k-of-n + threshold, instrumented (`measure`). Arduino core **3.x**. | ✅ physical build (boot, inference, alarm; quantitative numbers pending) |
| `fall_detector_core2x/` | Same logic for Arduino core **2.x** timers (reference build). | build + logic identical to above |
| `esp32_ai_final_millis_sampling/` | Final logic with `millis()` polling instead of a timer — compatibility fallback where timer API differs. | — (use only if the timer API cannot be used) |
| `esp32_usb_rule_based_fall_detector/` | AI-free two-stage detector (free-fall → impact/rotation), USB-powered. | ✅ physical build |
| `esp32_fall_detector_demo_final/` | Safe presentation demo (tilt gesture fires alarm effects). Not a detector. | ✅ physical build (demo use) |
| `esp32_initial_hardware_test/` | Bring-up: I²C scan, WHO_AM_I, live raw values, serial commands incl. manual LED/buzzer test. | ✅ physical build |
| `esp32_mpu6050_direct_test/` | Register-level MPU reader + simple magnitude threshold — the diagnostic that bypassed the Adafruit-init failure. | ✅ physical build |
| `legacy_global_norm/` | GEN-1 firmware (legacy model + frozen constants), live and replay variants. | superseded — see its README |

`MEASUREMENT.md` (from the reference repository) is the on-device measurement
protocol for the pending Table IV numbers; companion tooling in `tools/`.

Common contract for all final sketches: SDA 21, SCL 22, buzzer 4, LED 2,
MPU6050 at 0x68, ±16 g / ±2000 dps, DLPF on, 50 Hz, window 100×6, stride 25,
per-window instance normalisation in float, alarm = p_alert + p_fall.
