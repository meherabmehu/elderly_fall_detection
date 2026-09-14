# Project knowledge map

How every part of this repository fits together, where each result came from,
and the consistency audit that was run before publication. Read this before
trusting or changing anything.

---

## 1. The project in one paragraph

A waist-worn ESP32 + MPU6050 prototype that runs a 7,947-parameter INT8
separable CNN over 2-second windows of 6-channel IMU data at 50 Hz and fires a
buzzer + LED when the model's alert/fall probability stays above a threshold
for *k* consecutive windows. The model is trained and evaluated on four public
datasets (SisFall, KFall, FallAllD, UMAFall) with subject-grouped and
leave-one-dataset-out protocols. The system goal is **pre-impact** alerting:
firing during the fall, before the body hits the ground.

---

## 2. The two generations (this is the single most important thing to know)

The project exists in two generations. Both are in this repo; they have
**different models, different normalisation, different training data**, and
must never be mixed.

| | **GEN-2 (final)** | **GEN-1 (legacy, superseded)** |
|---|---|---|
| Purpose | the supported deployment + paper claims | historical record, first working Wokwi demo |
| Training data | real SisFall + KFall + FallAllD (+ UMAFall as held-out test only) | synthetic mirrors of the four datasets (legacy pipeline's own generator) |
| Pipeline | `training/src/fdlib/` + `training/kaggle/nb00–nb07` | `legacy/src/*.py` + `legacy/notebooks/` |
| Model | `models/final_int8/model.tflite` — 23,080 B, 7,947 params | `models/legacy_synthetic/model.tflite` — 19,928 B, 8,219 params |
| Normalisation | **per-window instance norm, in float, before INT8 quantisation** | frozen global mean/std (`frozen_normalization.json`) |
| Firmware | `firmware/esp32_ai_final_live_deployment/`, `firmware/fall_detector_core2x/`, `wokwi/final_model/` | `firmware/legacy_global_norm/`, `wokwi/legacy_synthetic_model/` |
| Every number in `results/kaggle_reference/` | ✔ | ✘ |
| Every number in `legacy/results/` | synthetic data — not publishable as dataset results | ✔ (its own results only) |

Rule of thumb: **if a file mentions `2574e6104c95` (preprocess signature),
GEN-2.** If it mentions `frozen_normalization` / `model_config.h` /
`model_data.h` / `NORM_MEAN`, GEN-1.

---

## 3. GEN-2 pipeline map (training → device)

```text
Raw datasets (never committed)
   SisFall 3,691 trials / 25 subjects (mirror has 25 of 38; no temporal labels)
   KFall   5,075 trials / 32 subjects (video-grounded onset+impact frames)
   FallAllD 1,798 trials / 14 subjects (238 Hz waist IMU)
   UMAFall   577 trials / 18 subjects (pocket smartphone, accelerometer only)
        |
        v   training/src/fdlib/datasets/*.py   (real-format parsers)
Trials @ 50 Hz, g and deg/s, namespaced subject IDs ("kfall:SA06")
        |
        v   fdlib.preprocess (7-step contract)
   units -> anti-alias decimate to 50 Hz -> axis canonicalisation
   -> window 100x6 stride 25 -> label (pre-impact rule) -> [norm fitted on TRAIN]
        |
        v   fdlib.windowing  (labels: bkg / alert / fall;
        |                      pre-impact window = right edge in [onset, impact);
        |                      fall span = impact..impact+2 s)
windows_*.npz  (signature 2574e6104c95)
        |
        +--> nb02 E1 within-dataset baselines (Table I, SisFall post-fall)
        +--> nb03 E2 leave-one-dataset-out + adaptation ladder (Table II)
        +--> nb04 E3 pre-impact lead time, KFall (Table III, trial-level evals,
        |          debounce analysis)
        +--> nb05 E5 ablations: window, rate, channels, placement
        +--> nb06 E4 export variants: in-graph vs pipeline instance norm
        |          (in-graph collapses under INT8: F1 0.668 → 0.359)
        +--> nb07 final model = SisFall+KFall+FallAllD, instance norm in the
               PIPELINE, 7,947 params, INT8 23,080 B, agreement 99.44%
               + KFall pre-impact baselines (Table I(b)) + operating-point sweep
        |
        v   fdlib.tflite_export (rep. dataset from TRAIN ONLY)
models/final_int8/{model.tflite, model.h, norm_constants.h, norm_constants.json}
        |
        v   firmware*/ wokwi/   (same normalisation, in C, in float)
ESP32 inference -> threshold + k-consecutive -> LED + buzzer
```

Every GEN-2 result CSV carries the preprocess signature and per-fold rows
(`training/src/fdlib/experiment.py` appends per fold and resumes after
crashes), so each table number traces to a fold and to a Kaggle run in
`results/kaggle_reference/experiment_log.csv`.

---

## 4. What the ESP32 firmware actually does

`firmware/esp32_ai_final_live_deployment/esp32_ai_final_live_deployment.ino`
(Arduino core 3.x timer API; `firmware/fall_detector_core2x/fall_detector.ino`
is the matching core 2.x build of the identical logic):

1. Register-level MPU6050 driver (no library): wake `0x6B`, ±16 g `0x1C=0x18`,
   ±2000 dps `0x1B=0x18`, DLPF `0x1A=0x04`, divider `0x19=19` → 50 Hz.
2. 50 Hz hardware timer → ring buffer of raw `int16` counts.
3. Every 25 new samples after the first 100: one inference —
   raw counts → g/deg/s → per-window per-channel instance normalisation
   (mean, std+1e-6, identical to `fdlib.preprocess.instance_normalise`) →
   INT8 quantise with the model's input scale/zero-point → `Invoke()`.
4. `p_alarm = p_alert + p_fall`; alarm when ≥ threshold for *k* consecutive
   windows; 5 s cooldown, 3 s alarm hold.
5. The **rule-based detector runs on the same samples** (free-fall < 0.65 g
   for ≥ 80 ms, then impact > 2.5 g or rotation > 250 dps within 1.5 s) →
   on-device classical baseline (`mode cnn|rule|both`).
6. Self-instrumentation: invoke/preprocess/end-to-end timing, arena
   high-water, alarm counters → `measure` command (`firmware/MEASUREMENT.md`).

---

## 5. Evidence map (claims → sources)

| # | Claim | Kind | Source artifact(s) |
|---|---|---|---|
| R1 | Within-dataset macro-F1 0.95–0.98, proposed 0.9502 ± 0.0063 | Kaggle reference | `results/kaggle_reference/results_e1.csv`, `results/tables/table_I.md` |
| R2 | LODO F1 0.34 FallAllD / 0.84 KFall; instance norm the only adaptation that helps | Kaggle reference | `results_e2.csv`, `results/tables/table_II.md` |
| R3 | KFall pre-impact: mean lead 588–663 ms (t=0.5–0.7), 89–92% pre-impact, 215–293 FA/h | Kaggle reference | `results_e3.csv`, `results/tables/table_III.md` |
| R4 | Trial-level deployed model: sens 0.91/spec 0.97/lead 722 ms @ (0.7, k=4); 0.96/0.84/539 ms @ (0.95, k=1) | Kaggle reference | `final_model_card.md`, `final_operating_points.csv` |
| R5 | Ablations: 2 s/50 Hz window good; accel-only drops ~1.3 F1 pts; wrist placement much worse | Kaggle reference | `results_e5.csv`, `results/tables/table_ablations.md` |
| R6 | INT8 22.54 KB, arena ~8 KB est., FP32→INT8 Δ −0.29 pp, agreement 99.44%; in-graph norm collapses (30.95 pp) | Kaggle reference | `quantisation_report.json`, `quantisation_comparison.json` |
| P1 | Hardware bring-up (0x68, ±1 g at rest, direct-register MPU6050 read) | Physical, this project | `firmware/esp32_mpu6050_direct_test/`, `esp32_initial_hardware_test/`, handoff §6 |
| P2 | AI-free rule detector works end-to-end on the bench | Physical, this project | `firmware/esp32_usb_rule_based_fall_detector/`, `reports/fall_detector_ai_free_report_english.pdf` |
| P3 | Final INT8 firmware boots, loads model, prints class probabilities, fires alarm on live motion | Physical, this project | `docs/PROJECT_HANDOFF_SUMMARY.md` §8; firmware itself |
| S1 | KFall S06T20R01 replay fires buzzer/LED with high fall probability | Simulation (Wokwi), legacy model | `wokwi/legacy_synthetic_model/`, handoff §9 |
| S2 | Same replay through the FINAL model: p_fall 0.9961 across impact windows, +100 ms lead at (≤0.9, k=1) | Desktop verification, this repo | `results/replay_verification/` (reproduce: `tools/replay_desktop_check.py`) |
| B1 | Final-model ESP32 firmware compiles (replay + live) | Build check, this repo | `wokwi/final_model/` (PlatformIO build log) |
| M1 | On-device latency, arena high-water, sensor-to-buzzer latency, current, battery | NOT MEASURED — pending | `experiments/hardware_measurement/hardware_measurements.json` (nulls) |
| M2 | Real 1-hour ADL false-alarm rate; 20-drop mattress test | NOT RUN — pending | `experiments/hardware_measurement/on_device_trials.md` (blank) |

---

## 6. Consistency audit (issues found and how they were resolved)

1. **Two models, two normalisation schemes.** Resolved by
   documenting GEN-1 as legacy/superseded and GEN-2 as the only supported
   deployment. The final firmware does per-window instance norm; the frozen
   global constants (`norm_constants.h`) are kept as a record and **not
   compiled in**.
2. **Package README named `esp32_ai_final_complete_main.cpp` as the "KFall
   replay" file.** It is actually a *live* firmware with millis-polling
   sampling (no replay logic). Resolved: it ships as
   `firmware/esp32_ai_final_millis_sampling/` (a compatibility variant), and a
   true replay firmware now exists in `wokwi/final_model/src/main.cpp`
   (`REPLAY_MODE 1`).
3. **The verified Wokwi replay ran the legacy (synthetic-trained) model**, not
   the final one — so "Wokwi replay verified" did not by itself verify the
   deployment model. Resolved: (a) final-model replay firmware added and
   compile-verified; (b) desktop replay verification of the final model
   committed (`results/replay_verification/`); (c) on-device/Wokwi execution
   of the final-model replay explicitly marked pending.
4. **Legacy `model_config.h` comment claims ±16 g = 16384 LSB/g.** 16384
   LSB/g is the ±2 g scale; ±16 g is 2048 LSB/g (the final firmware and the
   handoff both use 2048). The legacy macro is unused by its own sketches
   (they read through the Adafruit library / replay data in g), so it is a
   latent comment bug, not a live one — documented here and in
   `firmware/legacy_global_norm/README.md`.
5. **`fdlib.config` stride comment says "50 samples"; the value is 25**
   (25 samples = 0.5 s at 50 Hz = 75% overlap). Code is correct; comment is
   stale. Documented; not "fixed" to keep the reference library byte-identical
   with its Kaggle runs.
6. **All copies of GEN-2 model headers are byte-identical** (sha256
   `a27b6fc9…` for `model.h`, model.tflite `8e7cb130…`): in `models/final_int8/`,
   `firmware/esp32_ai_final_live_deployment/`, `wokwi/final_model/src/`,
   `results/kaggle_reference/`. Verified, and re-verifiable via
   `docs/ARTIFACT_INVENTORY.md`.
7. **`README` snapshot numbers in the legacy repo (8,219 params, 19.46 KB,
   1.07 pp ΔF1) are synthetic-pipeline numbers**, not final results — the
   legacy snapshot keeps them, clearly labelled, under `legacy/`.
8. **UMAFall exclusion reason differs between docs.** Settled: UMAFall's waist
   SensorTag logs at 20 Hz (below the 50 Hz working rate) *and* its 200 Hz
   pocket smartphone logs no gyroscope → the final model (trained with gyro
   channels) excludes it from training; it remains the E2 fold-D stress test
   (accel-only, placement shift) reported once.

---

## 7. Where things live

See `docs/ARTIFACT_INVENTORY.md` for the full tree with provenance. Golden
rules: numbers belong to `results/` (with provenance), pending numbers to
`experiments/hardware_measurement/` (null templates), procedure to `docs/` and
`firmware/MEASUREMENT.md`, nothing unpublished without a row in
`docs/RESULTS_PROVENANCE.md`.
