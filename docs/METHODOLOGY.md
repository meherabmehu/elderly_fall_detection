# Methodology

End-to-end description of the research pipeline as actually implemented in
`training/src/fdlib/` and executed by the Kaggle notebooks `training/kaggle/nb00–nb07`
(provenance for every number: `docs/RESULTS_PROVENANCE.md`).

## 1. Problem formulation

Waist-mounted 6-axis IMU stream → per-0.5 s decision: **bkg / alert / fall**.
The deployable quantity is *lead time*: how many milliseconds before the
labelled impact the first alarm fires. Post-hoc ("has fallen") detection is
treated only as a baseline, because an alarm after impact cannot prevent
injury.

## 2. Datasets and roles

| Dataset | Native rate | Subjects (mirror) | Trials | Role |
|---|---:|---:|---:|---|
| SisFall | 200 Hz | 25 of 38 | 3,691 | training corpus; E1 post-fall baseline; LODO fold |
| KFall | 100 Hz | 32 | 5,075 | **primary pre-impact source** (video-grounded onset & impact frames); LODO fold |
| FallAllD | 238 Hz | 14 | 1,798 | independent hardware/lab; LODO fold; placement ablation |
| UMAFall | 200 Hz (pocket) | 18 | 577 | held-out stress test, reported once (fold D) |

Key dataset findings (settled by the nb00 probe, documented in
`results/kaggle_reference/nb00_probe_report.md`):

- The SisFall *Enhanced* mirror is pre-windowed shuffled tensors — no subject
  IDs; its 3-class annotations are not recoverable (null-controlled ceiling
  22.5%), so **KFall alone carries the pre-impact claim**.
- The SisFall mirror has 25 of 38 subjects (13 of 15 older participants
  missing).
- UMAFall's waist SensorTag runs at 20 Hz; only the pocket smartphone reaches
  200 Hz and it logs **no gyroscope** → UMAFall windows are accelerometer-only
  and are excluded from *training* the deployed model.

SisFall has no temporal labels: fall impact indices are estimated as the
acceleration-magnitude peak (standard proxy; stated as a limitation). KFall
labels are video-grounded frames — the authoritative source.

## 3. Frozen preprocessing contract (`fdlib/preprocess.py`, `windowing.py`)

Seven steps, executed identically everywhere — the same library code trains
the model and generates the C contract the firmware implements:

1. **Channel selection** — waist/low-back IMU; `ax ay az gx gy gz`;
2. **Unit conversion** — g and deg/s (SisFall raw ADC via documented scale
   factors; FallAllD ±8 g/±2000 dps over 16 bits);
3. **Anti-alias decimation to 50 Hz** — `scipy.signal.decimate`
   (integer factors, staged ≤10), plus linear remainder where the ratio is
   not integral (238 Hz); never plain slicing;
4. **Axis canonicalisation** — measured per dataset with a gravity-axis vote,
   then a signed-permutation rotation onto "gravity on −Y" (right-handed;
   FallAllD and UMAFall rotated, SisFall/KFall already canonical);
5. **Windowing** — 100×6 windows, stride 25 (75% overlap);
6. **Labelling (pre-impact rule)** — a window is ALERT if its right edge lies
   in [onset, impact); FALL from impact to impact+2 s; BKG otherwise. SisFall
   (post-fall task): windows containing the impact;
7. **Caching** — `.npz` stamped with `preprocess_signature()` =
   `2574e6104c95`; loading a corpus stamped differently raises an error.

Every cached artifact carries that signature; mixing preprocessing versions is
structurally impossible.

## 4. Model (`fdlib/models.py`)

Compact separable CNN (target TinyML budget — 60 KB flash, 120 KB arena,
<50 ms inference):

```text
Input (100x6)
 → Conv1D(24, k=7, s=2, no bias) + BatchNorm + ReLU
 → SeparableConv1D(48, k=5) + BatchNorm + ReLU + MaxPool(2)
 → SeparableConv1D(64, k=3) + BatchNorm + ReLU
 → GlobalAveragePooling1D
 → Dense(32, ReLU) + Dropout(0.3)
 → Dense(3, softmax)
```

7,947 trainable parameters. Comparators: 1D-CNN (102,211), CNN-LSTM (47,811),
Random Forest, RBF-SVM on hand-crafted window features, SMV threshold.

**Instance normalisation lives in the pipeline, not the graph.** With the op
inside the graph, INT8 quantisation collapses macro-F1 0.668 → 0.359
(per-window mean/variance intermediates exceed one global scale); computed in
float before quantisation the delta is −0.22 pp. The firmware performs the
same two-pass per-window normalisation in float.

## 5. Training and evaluation protocols

- Splits **by subject or by dataset, never by window** (75% overlap makes
  window-splits meaningless): GroupKFold(5) for baselines/ablations; full LOSO
  headline; LODO{A–D} for cross-dataset; inner validation split by subject for
  early stopping.
- Class imbalance handled by **class weights**, not oversampling (near-
  identical windows across boundaries would leak).
- Adam @ 1e-3, cosine decay, batch 128, max 100 epochs, early stop on
  validation **macro-F1** (accuracy is ~97% by always saying "no fall").
- Seed 1337 everywhere.

Metrics: sensitivity/specificity/macro-F1/AUC at window and trial levels;
**lead time** = impact frame − first-alarm right edge (positive = pre-impact);
**false alarms per hour of ADL** (the wearable-reality metric: at a 0.5 s
stride the device decides 7,200 times/hour, so even 99% window-specificity
≈ 72 alarms/hour).

Experiments: **E1** within-dataset baselines (SisFall post-fall; KFall
pre-impact), **E2** leave-one-dataset-out with an adaptation ladder
(none → instance-norm → CORAL → DANN, never touching target labels),
**E3** pre-impact lead time + false-alarm curves + debounce sweep,
**E4** quantisation/export verification, **E5** ablations (window, rate,
channels, placement), **E6/E7** = hardware measurement programme (pending).

## 6. Deployment contract

- Full-integer INT8 (`TFLITE_BUILTINS_INT8`, int8 in/out), representative
  dataset of 200 windows drawn **from the training split only**.
- Desktop verification before any flash: FP32 vs INT8 on the same held-out
  subjects must agree (observed 0.9944); then `model.h` C array + `model.tflite`
  are committed and never hand-edited.
- Firmware contract: 50 Hz hardware timer; ±16 g / ±2000 dps; DLPF on;
  per-window instance norm in float; `p_alarm = p_alert + p_fall`; threshold +
  *k*-consecutive confirmation (default 0.95 / 2; switchable live).
- Operating point chosen from the 814-trial sweep
  (`models/final_int8/final_model_card.md`), not conveniently.

## 7. Reproducibility anchors

`config.SEED=1337`; dataset Kaggle sources + access dates in
`training/PROJECT_PLAN.md`; per-Kaggle-run log
`results/kaggle_reference/experiment_log.csv`; library version hash
`results/kaggle_reference/fdlib_version.json`; artifact checksums in
`docs/ARTIFACT_INVENTORY.md`.
