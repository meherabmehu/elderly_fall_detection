# Wokwi simulation guide

Wokwi verifies **firmware + model integration** (I²C wiring, window
scheduling, INT8 inference, alarm logic) without physical hardware. It does
**not** verify physical latency, RAM, sensor noise, battery behaviour or
fall-detection accuracy in the real world.

## Projects in this repository

| Project | Model | Normalisation | Status |
|---|---|---|---|
| `wokwi/final_model/` | final INT8 (23,080 B) | per-window instance norm (float) | **recommended** — compiles cleanly; simulator execution pending |
| `wokwi/legacy_synthetic_model/` | legacy (19,928 B, synthetic-trained) | frozen global constants | kept for provenance; the Aug-2026 verified demo ran this one |

## `wokwi/final_model/` layout

```text
diagram.json      ESP32 DevKit V1 + MPU6050 (0x68) + buzzer + LED (+220 Ω)
platformio.ini    env esp32dev (REPLAY_MODE=1, default) and esp32dev_live (=0)
wokwi.toml        .pio/build/esp32dev/firmware.(bin|elf)
src/main.cpp      the final firmware with replay support (compiles both modes)
src/model.h       final INT8 model as a C array (byte-identical to models/final_int8/model.h)
src/replay_data.h KFall S06T20R01, 268 samples x 6 ch, 50 Hz, g + dps
```

The library pin (`tanakamasayuki/TensorFlowLite_ESP32@^1.0.0`) matches the
legacy TFLM API used by the physically verified Arduino sketch; the source
also carries a compatibility shim for newer tflite-micro packages that removed
`MicroErrorReporter`.

## Run it (VS Code, ~10 minutes)

1. Install **PlatformIO IDE** and the **Wokwi** extension.
2. Open the `wokwi/final_model/` folder.
3. Build:

   ```bash
   pio run                 # REPLAY_MODE=1
   # or for live-sensor simulation:
   pio run -e esp32dev_live
   ```

4. `Ctrl+Shift+P → Wokwi: Start Simulator` (select `wokwi.toml`).
5. Expected serial output:

```text
PRE-IMPACT FALL DETECTOR -- FINAL INT8 MODEL
MODE: KFALL S06T20R01 REPLAY
MPU6050 READY (+/-16 g, +/-2000 dps, DLPF on, 50 Hz)
model flash    : 23080 bytes (22.5 KB)
normalisation  : PER-WINDOW instance norm (not frozen constants)
REPLAY MODE ENABLED
Dataset: KFall
Trial: S06T20R01
Replay samples: 268 @ 50 Hz
Ground truth: onset sample 65, impact sample 104
window @ edge 99  p_bkg=0.0781 p_alert=0.9219 p_fall=0.0000
window @ edge 124 p_bkg=0.0000 p_alert=0.0000 p_fall=0.9961
...
FALL ALERT: CNN: alert/fall probability sustained
alarm right edge : 149  ->  lead time 900 ms AFTER impact
```

(Exact alarm edges depend on the live `thr`/`k` settings; per-window
probabilities are deterministic and match
`results/replay_verification/`.) The trial then loops every ~5 seconds.

6. Change the operating point live in the simulator serial monitor:
   `thr 0.9`, `k 1` (this trial's stem fires 100 ms *before* impact at ≤ 0.9,
   k=1 — see `results/replay_verification/REPLAY_VERIFICATION.md`).

## Replay data provenance

`src/replay_data.h` was extracted from KFall `sensor_data/.../S06T20R01.csv`
with labels `label_data/SA06_label.xlsx` (Task T20, Trial R01), resampled
from 100 Hz to 50 Hz into g and deg/s. It is a single-trial excerpt kept for
integration demo purposes; the full KFall dataset is **not** redistributed
(see `datasets/README.md`).

## Notes and honest caveats

- The simulator executes the exact `.bin` produced by `pio run` — so a Wokwi
  run exercises the same firmware bytes that would flash onto silicon, minus
  physical I²C behaviour.
- The wokwi-mpu6050 part is idealised (no noise/offset), so live-mode Wokwi
  behaviour should not be used for accuracy claims.
- Record the simulator run you intend to cite (serial log via clipboard or
  the extension's capture), then file it next to
  `results/replay_verification/` so the claim stays auditable.
