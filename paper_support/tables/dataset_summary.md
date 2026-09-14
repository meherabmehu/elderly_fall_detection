# Dataset summary (as used)

Numbers from `results/kaggle_reference/nb01_report.md` (parsed corpora) and
`gate_report.json`; roles settled by the nb00 probe.

| Dataset | Native rate | Trials (mirror) | Falls | ADL | Subjects | Sensor used | Role in this work |
|---|---:|---:|---:|---:|---:|---|---|
| SisFall | 200 Hz | 3,691 | 1,798 | 1,893 | 25 of 38 | waist IMU (acc+gyro) | training corpus; E1 post-fall baseline; E2 fold C |
| KFall | 100 Hz | 5,075 | 2,346 | 2,729 | 32 | low-back IMU (acc+gyro) | **primary pre-impact source**; E1(b) pre-impact baselines; E2 fold B |
| FallAllD | 238 Hz | 1,798 | 466 | 1,332 | 14 | waist IMU (acc+gyro) | independent lab; E2 fold A; placement ablation (waist/neck/wrist) |
| UMAFall | 200 Hz pocket | 577 | 202 | 375 | 18 | pocket smartphone (**accel only**) | E2 fold D stress test, reported once; excluded from deploy-model training |

Window corpus after preprocessing (50 Hz, 100×6, stride 25): **200,596
windows** from 89 subjects (norm constants fitted on the 71 training
subjects) — `results/kaggle_reference/norm_constants.json`.

Label quality notes (must be stated in the paper):

- KFall onset/impact frames are video-grounded — the authoritative pre-impact
  source; its loader forward-fills sparse task-code annotations before
  indexing (a silent mislabelling trap, documented in
  `training/src/fdlib/datasets/kfall.py`).
- SisFall has no temporal labels in the mirror; impact is estimated as the
  acceleration-magnitude peak (proxy).
- The SisFall Enhanced three-class annotations are not recoverable from the
  Kaggle mirror (null-controlled ceiling 22.5%) — pre-impact claims rest on
  KFall alone.
