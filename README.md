# Elderly Fall Detection on ESP32

A low-cost, waist-worn inertial-sensor system that detects **falls and
pre-impact alert states** with a compact INT8 convolutional network running
on an **ESP32 DevKit V1**, raising a red LED and an active buzzer when a fall
is detected.

Built from a TinyML research pipeline (four public datasets, subject-grouped
and leave-one-dataset-out evaluation, full INT8 quantisation) and verified on
real hardware and in Wokwi simulation.

---

## Status at a glance

| Claim | Status | Evidence |
|---|---|---|
| MPU6050 + ESP32 hardware bring-up (I²C 0x68, GPIO 21/22, LED D2, buzzer D4) | **Verified on physical hardware** | `firmware/esp32_*_test*`, `docs/HARDWARE.md` |
| AI-free rule-based fall detector (free-fall + impact/rotation) | **Verified on physical hardware** | `firmware/esp32_usb_rule_based_fall_detector/`, `reports/fall_detector_ai_free_report_english.pdf` |
| Final INT8 model loads and infers on the physical ESP32, prints `p_bkg/p_alert/p_fall`, raises the alarm | **Verified on physical hardware** (USB-powered) | `firmware/esp32_ai_final_live_deployment/`, `docs/PROJECT_HANDOFF_SUMMARY.md` §8 |
| Dataset results (Tables I–IV, ablations, lead-time, false alarms) on real public datasets | **Verified — Kaggle reference runs** (see provenance) | `results/kaggle_reference/`, `results/tables/` |
| Desktop replay verification: committed INT8 model detects KFall trial `S06T20R01` end-to-end | **Verified by this repo's tooling, reproducible now** | `results/replay_verification/`, `tools/replay_desktop_check.py` |
| Final-model firmware compiles for ESP32 (replay + live modes) | **Verified build (PlatformIO, this repo's CI step done locally)** | `wokwi/final_model/` |
| On-device latency, tensor-arena high-water mark, current draw, battery life | **PENDING — requires the bench** | `experiments/hardware_measurement/`, `firmware/MEASUREMENT.md` |
| Real-world false-alarm rate (1-hour ADL wear), mattress drop trials | **PENDING — requires supervised trials** | `experiments/hardware_measurement/on_device_trials.md` |

Nothing in this repository turns a simulation result into a hardware
measurement, or a reference-pipeline result into a new experiment. See
[`docs/RESULTS_PROVENANCE.md`](docs/RESULTS_PROVENANCE.md) for the per-number
audit trail.

---

## System overview

```text
MPU6050 / GY-521 (waist)
        |  6 channels: ax ay az gx gy gz
        |  I2C @ 400 kHz, 50 Hz
        v
ESP32 DevKit V1
        |  100 x 6 rolling window (2.0 s, stride 25 = 0.5 s)
        |  per-window instance normalisation in float
        |  INT8 quantisation
        v
TensorFlow Lite Micro  (22.54 KB INT8 separable CNN, 7,947 params)
        |
        v
p_bkg / p_alert / p_fall  ->  threshold + k-consecutive confirmation
        |
        +--> red LED (GPIO 2, 220 ohm)
        +--> active buzzer (GPIO 4)
```

Also on board: an **AI-free rule-based detector** (free-fall → impact/rotation
sequence) that runs on the same samples, so the CNN can be compared against a
classical threshold detector on identical motion (`mode cnn|rule|both`).

## The frozen technical contract

These values are the contract between training and firmware — a mismatch is
silent and fatal:

```text
Sampling rate   : 50 Hz (MPU6050 divider 19, DLPF on; hardware timer, no delay())
Channels        : ax, ay, az (g), gx, gy, gz (deg/s); accel ±16 g, gyro ±2000 dps
Window          : 100 samples x 6 channels (2.0 s)
Stride          : 25 samples (0.5 s)
Classes         : bkg / alert / fall (alarm = p_alert + p_fall)
Normalisation   : PER-WINDOW instance normalisation, in float, before quantisation
Model           : separable CNN, 7,947 params, INT8 23,080 bytes (22.54 KB)
Preprocess sig  : 2574e6104c95
Default point   : threshold 0.95, k = 2  (switch live: `thr`, `k`)
```

**Do not** mix the final model with global/frozen-constant normalisation, or
the legacy model with instance normalisation. The final supported deployment
combination is pinned down in
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md#final-supported-deployment-combination).

## Results summary

From the reference Kaggle runs on the four real datasets
(provenance in `results/kaggle_reference/`, audit in `docs/RESULTS_PROVENANCE.md`):

- Within-dataset macro-F1 is saturated (0.95–0.98); the proposed 7,947-param
  model reaches 0.9502 grouped-5-fold (Table I).
- **Leave-one-dataset-out is much harder and asymmetric:** F1 0.34 on unseen
  FallAllD vs 0.84 on unseen KFall; per-window instance normalisation is the
  only adaptation rung that helps (Table II).
- **Pre-impact:** 663 ms mean lead at threshold 0.5, detecting 91.8% of 2,346
  KFall fall trials pre-impact — at ~293 false alarms/hour; the
  threshold/*k* trade-off and the honest false-alarm analysis are in
  `results/tables/debounce_analysis.md` (Table III + trial-level evaluations).
- **Deployment:** INT8 22.54 KB, ~8 KB tensor arena (desktop estimate),
  FP32→INT8 agreement 99.4% — measured on desktop; the on-device latency/RAM/
  current/battery numbers are **pending** (`experiments/hardware_measurement/`).
- **This repo's replay verification:** the committed INT8 model replayed
  through the exact firmware preprocessing flags KFall `S06T20R01`
  pre-impact (+100 ms at threshold ≤ 0.9, k=1) and scores 0.9961 fall
  probability across the impact windows — see
  `results/replay_verification/REPLAY_VERIFICATION.md`.

## Hardware

| Component | Connection | Notes |
|---|---|---|
| MPU6050 VCC | ESP32 3V3 | **never 5 V** |
| MPU6050 GND | GND | common ground |
| MPU6050 SDA | GPIO 21 / D21 | I²C data |
| MPU6050 SCL | GPIO 22 / D22 | I²C clock 400 kHz |
| MPU6050 AD0 | GND | address 0x68 (verified on the physical build) |
| Buzzer + | GPIO 4 / D4 | active buzzer |
| LED anode | GPIO 2 / D2 via 220 Ω | red LED |
| Power | USB | battery (TP4056 + 1000 mAh LiPo) is the optional next step |

Wiring diagrams, BOM and pin sheets: [`hardware/`](hardware/),
build/test guide: [`docs/HARDWARE.md`](docs/HARDWARE.md).

## Quick start

### 1. See the model run in simulation (Wokwi, ~10 min)

```text
VS Code + PlatformIO + Wokwi extension
open wokwi/final_model/
pio run                       # builds REPLAY_MODE=1 (KFall S06T20R01)
Wokwi: Start Simulator        # Ctrl+Shift+P
```

The trial replays through the trained INT8 model; when the fall phase enters
the window the buzzer and LED fire, and the serial console prints lead time
against the labelled impact. Details: [`docs/WOKWI.md`](docs/WOKWI.md).

### 2. Flash the physical device (Arduino IDE)

Copy `firmware/esp32_ai_final_live_deployment/` (with its `model.h` and
`norm_constants.h`) into a sketch folder, install the **TensorFlowLite_ESP32**
library, flash, open Serial at **115200**. Type `help`, then `test`, then
`measure` after ~8 minutes for the device instrumentation. Full guide:
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

### 3. Re-run the desktop replay verification (no hardware)

```bash
python -m venv .venv && .venv/bin/pip install tensorflow
.venv/bin/python tools/replay_desktop_check.py \
    --model models/final_int8/model.tflite \
    --replay wokwi/final_model/src/replay_data.h \
    --norm instance --threshold 0.95 --k 2
```

### 4. Reproduce the training/evaluation pipeline

The full pipeline (datasets → windows → E1/E2/E3/E5 → INT8 export) is scripted
under [`training/`](training/) and runs on Kaggle GPU. Raw datasets are **not**
in this repository; fetch instructions and expected layouts are in
[`datasets/README.md`](datasets/README.md).
Guide: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Repository structure

```text
├── docs/                  all documentation (start: KNOWLEDGE_MAP.md)
├── datasets/              dataset acquisition + expected layouts (no raw data)
├── training/              fdlib pipeline: library, Kaggle notebooks, scripts
├── results/               Kaggle reference results, tables, replay verification
├── models/                final_int8/ (deployed) + legacy_synthetic/ (superseded)
├── firmware/              ESP32 sketches: final live AI, rule-based, demo, bring-up
├── wokwi/                 Wokwi projects: final_model/ (recommended) + legacy
├── experiments/           pending hardware measurement templates + future logs
├── tools/                 replay checker, serial capture, measure-log parser
├── hardware/              wiring diagrams, BOM, pin connection sheets
├── reports/               project reports (AI-free report, proposal report, …)
├── presentations/         20-minute English / Bangla presentation scripts
├── paper_support/         manuscript source, paper-ready tables/figures, refs
└── legacy/                superseded synthetic-pipeline snapshot (provenance)
```

## Limitations — read before citing

1. **On-device resource numbers are pending.** Latency, tensor-arena
   high-water mark, current and battery life are instrumented in the firmware
   and have a documented measurement protocol, but have **not yet been
   measured on the bench**. They appear as `PENDING` everywhere, on purpose.
2. **Wokwi replay ≠ real-world validation.** It verifies model/firmware
   integration on a real dataset trial. It says nothing about sensor noise,
   mounting, or battery behaviour.
3. **Datasets are simulations of falls, not elderly falls.** Public fall
   corpora are mostly young-adult, supervised, soft-surface falls; see
   `docs/LIMITATIONS.md`.
4. **The legacy model (models/legacy_synthetic/) was trained on synthetic
   data.** Its results are kept for provenance only and must not be reported
   as dataset results.
5. Cross-dataset generalisation is genuinely hard (Table II); do not quote
   within-dataset numbers as deployment expectations.

Full list: [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Provenance and attribution

The training pipeline and most quantitative methodology results originate from
the reference repository
[`arifshekhk8/preimpact-fall-detection-tinyml`](https://github.com/arifshekhk8/preimpact-fall-detection-tinyml)
(co-authored by this project's team). This repository adds the ESP32/Wokwi
integration, hardware bring-up, replay verification tooling, and the
documentation/measurement framework for the conference submission. Details:
[`ATTRIBUTION.md`](ATTRIBUTION.md), [`docs/RESULTS_PROVENANCE.md`](docs/RESULTS_PROVENANCE.md).

## Citations

Using the datasets obliges citing Sucerquia et al. 2017 (SisFall), Musci et
al. 2018/2020 (SisFall Enhanced annotations), Yu et al. 2021 (KFall), Saleh et
al. 2020 (FallAllD) and Casilari et al. 2017 (UMAFall). BibTeX:
[`paper_support/refs.bib`](paper_support/refs.bib).

## Safety

This is a research prototype, **not a certified medical device**. Test only
with safe, supervised, soft-surface protocols. Do not deliberately fall, and
never use this device as the sole safeguard for a person at risk of falling.
