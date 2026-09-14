# Artifact inventory

Complete map of the repository: what exists, what it is for, and where it came
from. Provenance keys: **[REF]** = reference pipeline/repo; **[HW]** = this
project's physical-hardware work; **[NEW]** = created for this repository;
**[LEGACY]** = superseded generation, kept for provenance.

```text
├── README.md                          [NEW]   project overview + status claims
├── ATTRIBUTION.md                     [NEW]   provenance & ownership
├── .gitignore                         [NEW]
│
├── datasets/README.md                 [NEW]   acquisition links, expected layouts, citation duties (no raw data — by design)
│
├── docs/                              [NEW]   knowledge map, methodology, architecture, deployment,
│   │                                          hardware, Wokwi, reproducibility, experiments, results provenance,
│   │                                          paper guide, limitations, artifact inventory
│   └── PROJECT_HANDOFF_SUMMARY.md     [HW]    the project's status summary that seeded this repo
│
├── training/                          [REF]   the real-data ML pipeline (fdlib)
│   ├── src/fdlib/                     [REF]   config/preprocess/windowing/cv/models/baselines/adapt/metrics/experiment/tflite_export + 4 dataset parsers
│   ├── kaggle/nb00…nb07/              [REF]   probe, preprocess, E1, E2, E3, E5, export, final
│   ├── scripts/                       [REF]   sync_fdlib, run_kernel, build_tables, debounce_analysis, trial_level_eval
│   ├── Makefile, requirements.txt     [REF]
│   └── PROJECT_PLAN.md, SHOWCASE.md   [REF]   execution plan + demo-day run sheet
│
├── results/
│   ├── kaggle_reference/              [REF]   every CSV/JSON/figure/report from the Kaggle runs (Tables I–IV artifacts)
│   ├── tables/                        [REF]   paper-ready markdown tables (I, Ib, II, III, IV, ablations, trial-level, debounce)
│   └── replay_verification/           [NEW]   desktop INT8 replay JSONs + REPLAY_VERIFICATION.md (this repo's own verified numbers)
│
├── models/
│   ├── final_int8/                    [REF]   model.tflite 23,080 B (sha 8e7cb130…), model.h (a27b6fc9…),
│   │                                          norm_constants.h/.json, final_model_card.md, quantisation reports
│   └── legacy_synthetic/              [LEGACY] 19,928 B synthetic-trained model + frozen global constants
│
├── firmware/
│   ├── README.md                      [NEW]
│   ├── MEASUREMENT.md                 [REF]   on-device measurement protocol
│   ├── esp32_ai_final_live_deployment [HW]    final firmware, Arduino core 3.x — VERIFIED on physical ESP32
│   ├── fall_detector_core2x/          [REF]   same logic, core 2.x timer API
│   ├── esp32_ai_final_millis_sampling [HW]    millis-polling compatibility variant
│   ├── esp32_usb_rule_based_fall_detector [HW] AI-free two-stage detector — VERIFIED
│   ├── esp32_fall_detector_demo_final [HW]    safe tilt demo
│   ├── esp32_initial_hardware_test/   [HW]    bring-up sketch — VERIFIED
│   ├── esp32_mpu6050_direct_test/     [HW]    register-level sensor checker — VERIFIED
│   └── legacy_global_norm/            [LEGACY] legacy model firmware (live/replay) — do not deploy
│
├── wokwi/
│   ├── README.md                      [NEW]
│   ├── final_model/                   [NEW/REF] PlatformIO+Wokwi project for the FINAL model:
│   │                                          src/main.cpp [NEW] replay + live (REPLAY_MODE), TFLM compat shim;
│   │                                          src/model.h [REF, byte-identical], src/replay_data.h [REF];
│   │                                          diagram.json, platformio.ini [NEW envs], wokwi.toml [NEW]
│   └── legacy_synthetic_model/        [LEGACY] the Aug-2026 verified replay project (old model)
│
├── experiments/hardware_measurement/  [REF]   hardware_measurements.json (null template), on_device_trials.md (blank)
├── experiments/README.md              [NEW]
│
├── tools/                             [NEW]   replay_desktop_check, serial_capture, parse_measure_log (+ README)
│
├── hardware/                          [HW]    pin diagram SVG, annotated wiring PNGs, BOM + connection sheets (docx)
├── reports/                           [HW]    AI-free reports (pdf), proposal report (docx)
├── presentations/                     [HW]    20-min scripts (English pdf, Bangla docx/pdf)
│
├── paper_support/
│   ├── manuscript/ieee/               [REF]   IEEEtran draft (main.tex, refs.bib, figs, ieee.csl)
│   ├── manuscript/OUTLINE.md          [REF]   section-by-section plan
│   ├── manuscript/VERIFICATION.md     [REF]   number-audit practice (3 errors found & fixed)
│   ├── Fall_Detection_Experimental_Plan_v2.pdf [REF]
│   ├── refs.bib                       [REF]
│   ├── figures/                       [REF]   lead time, sanity traces, alert intervals, method figure
│   ├── tables/                        [NEW]   dataset summary, operating points, hardware-resource table
│   └── checklists/                    [NEW]   submission + reproducibility checklists
│
└── legacy/                            [LEGACY] snapshot of the synthetic-pipeline project (own src/, notebooks,
                                               models, results, firmware, wokwi) — superseded, see legacy/README.md
```

## Identity checksums (re-verify any time)

```text
8e7cb13067fca18a  models/final_int8/model.tflite          (23,080 B)
a27b6fc99a775ce2  models/final_int8/model.h
                  = firmware/esp32_ai_final_live_deployment/model.h
                  = wokwi/final_model/src/model.h
b0621006f65a2311  models/final_int8/norm_constants.h
7727bc47b66cc2fc  wokwi/final_model/src/replay_data.h     (268 x 6, 50 Hz)
68acb0ca172bde37  models/legacy_synthetic/model.tflite   (19,928 B)
e61c2717a56a6f71  models/legacy_synthetic/proposed_fp32.keras
```

Preprocess signature of every GEN-2 artifact: `2574e6104c95`.
Total workspace footprint: ~16 MB across ~240 files (no raw datasets, no build
output, no secrets).
