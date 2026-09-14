# ESP32 deployment guide

## Final supported deployment combination

The only combination supported for deployment and paper claims:

| Component | File | Identity |
|---|---|---|
| INT8 model | `models/final_int8/model.tflite` | 23,080 bytes, sha256 `8e7cb130…` |
| Model C header | `models/final_int8/model.h` | sha256 `a27b6fc9…`, preprocess signature `2574e6104c95` |
| Normalisation | **per-window instance normalisation, in float** (implemented inside the firmware; identical to `fdlib.preprocess.instance_normalise`) | — |
| Firmware (Arduino core 3.x) | `firmware/esp32_ai_final_live_deployment/` | verified on the physical build |
| Firmware (Arduino core 2.x) | `firmware/fall_detector_core2x/` | identical logic, legacy timer API |
| Wokwi project | `wokwi/final_model/` | REPLAY_MODE 1 default; compiles both modes |
| Operating point | threshold 0.95, k = 2 default; 0.7 / 4 = model-card recommendation | live-switchable |

**Never** combine:
- `models/final_int8/*` with frozen/global normalisation, or
- `models/legacy_synthetic/*` (19,928 B, synthetic-trained) with instance-normalised firmware.
`firmware/legacy_global_norm/` and `wokwi/legacy_synthetic_model/` exist only
as the historical first demo (see their READMEs).

---

## Option A — Arduino IDE (as physically verified)

1. Copy the folder `firmware/esp32_ai_final_live_deployment/` somewhere; the
   sketch folder name must match the `.ino` name. Keep **all three** files in
   it: `.ino`, `model.h`, `norm_constants.h` (the last is included as a record;
   the final firmware does *not* use the frozen constants).
2. Install Arduino IDE board package **esp32 by Espressif** (core 3.x tested;
   for core 2.x use `firmware/fall_detector_core2x/` instead).
3. Install library **TensorFlowLite_ESP32** (tanakamasayuki) from the Library
   Manager. This is the legacy API the verified sketch targets
   (`MicroErrorReporter`). Do not substitute the Nano-33-oriented
   `Arduino_TensorFlowLite` — on ESP32 builds it fails at its `peripherals/`
   layer, which is why `wokwi/final_model/` pins the tanakamasayuki library.
4. Board "ESP32 Dev Module", 115200 baud, flash.
5. Expected boot:

```text
PRE-IMPACT FALL DETECTOR -- INT8 CNN + RULE BASELINE
WHO_AM_I: 0x68
MPU6050 READY (+/-16 g, +/-2000 dps, DLPF on, 50 Hz)
model flash    : 23080 bytes (22.5 KB)
arena at init  : ..... of 24576 bytes
input  scale ...  zero_point ...
normalisation  : PER-WINDOW instance norm (not frozen constants)
System ready.
```

6. Sanity gate **before any measurement**: leave the board still — it must not
   alarm; rotate it slowly — still nothing. If it alarms while stationary,
   stop: axes or scaling are wrong and no later number means anything.

## Option B — PlatformIO

`wokwi/final_model/` is a complete PlatformIO project (also usable without the
simulator):

```bash
pio run -e esp32dev_live         # builds the live-MPU6050 firmware
pio run -e esp32dev_live -t upload --upload-port COMx
pio device monitor -p COMx -b 115200
```

## Serial command reference (final firmware)

| Command | Effect |
|---|---|
| `test` | blink LED + beep buzzer 3× (wiring check) |
| `alarm` / `off` | force alarm on / turn off |
| `status` | sensor, mode, flags, windows seen |
| `measure` | latency stats, arena high-water, alarm counts, threshold/k |
| `reset` | reset state and counters |
| `mode cnn\|rule\|both` | which detector drives the alarm |
| `thr 0.7` | set CNN threshold |
| `k 4` | set consecutive-window confirmation |
| `help` | list commands |

`measure` prints automatically every 1,000 inferences (~8 minutes at the
0.5 s stride).

## Firmware variants and what they are for

| Folder | Mode | Notes |
|---|---|---|
| `firmware/esp32_ai_final_live_deployment/` | live AI (CNN + rule), core 3.x timer | **physically verified** build |
| `firmware/fall_detector_core2x/` | live AI, core 2.x timer API | reference build of the same logic |
| `firmware/esp32_ai_final_millis_sampling/` | live AI with `millis()` polling instead of the timer | compatibility fallback; keep the same three-file layout (sketch + `model.h` + `norm_constants.h` as a record) |
| `firmware/esp32_usb_rule_based_fall_detector/` | AI-free two-stage rule detector | physically verified baseline |
| `firmware/esp32_fall_detector_demo_final/` | AI-free demo (safe tilt gesture) | presentation use only |
| `firmware/esp32_initial_hardware_test/` | bring-up: I²C scan, raw values, manual alarm | first-flash test |
| `firmware/esp32_mpu6050_direct_test/` | register-level MPU6050 reader + threshold | diagnoses the Adafruit-init failure |
| `firmware/legacy_global_norm/` | legacy model firmware (live/replay) | **superseded** — see README inside |

## Measuring the pending Table IV numbers

Follow `firmware/MEASUREMENT.md` exactly; capture the session with
`tools/serial_capture.py` and fold the results into
`experiments/hardware_measurement/hardware_measurements.json` with
`tools/parse_measure_log.py`. Until then every latency/RAM/current/battery
figure stays `PENDING`.
