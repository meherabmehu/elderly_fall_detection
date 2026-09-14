# firmware/legacy_global_norm/ — superseded legacy firmware

GEN-1 firmware for the legacy synthetic-trained model. Kept so the August 2026
demo remains reproducible. **Do not deploy, do not cite its numbers as
dataset results.**

| File | Content |
|---|---|
| `live_main.cpp` | live MPU6050 firmware (Adafruit_MPU6050), frozen global normalisation, threshold 0.60, fires immediately |
| `replay_main.cpp` | same, with `#define REPLAY_MODE 1` — replays KFall S06T20R01 from `replay_data.h` |
| `model_data.h` | legacy INT8 model as a C array (8,219-param model, 19,928 B) |
| `model_config.h` | frozen normalisation constants + input scale/zero-point. **Comment bug:** `ACCEL_RANGE 16384` is described as ±16 g; 16384 LSB/g is the ±2 g scale (2048 is ±16 g). The macro is unused by these sketches, so it never executed — do not copy it forward |
| `replay_data.h` | KFall S06T20R01 excerpt used by the replay sketch (byte-identical to the current replay data) |

The PlatformIO/Wokwi project for this firmware is in
`wokwi/legacy_synthetic_model/` (it includes its own copies of the headers so
the project builds standalone).

If you are looking for the firmware to run: use
`firmware/esp32_ai_final_live_deployment/`
(see `docs/DEPLOYMENT.md`).
