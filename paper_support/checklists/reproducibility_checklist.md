# Method reproducibility checklist (paper-facing)

Based on `training/src/fdlib/config.py` and the final deployment contract.
Every item is checkable in the repository.

## Data and preprocessing

- [x] Working rate **50 Hz**; anti-alias decimation via `scipy.signal.decimate`
      (integer factors, staged ≥ factor-10 cascades), linear fallback for
      non-integer ratios (238 Hz FallAllD)
- [x] Channels `ax ay az gx gy gz` in g and deg/s; IMU ranges ±16 g / ±2000 dps
- [x] Window **100×6 (2.0 s)**, stride **25 (0.5 s, 75% overlap)**
- [x] Axis canonicalisation via measured gravity-axis vote + signed
      permutation; rotations recorded per dataset
- [x] Labels: bkg = 0, alert = 1, fall = 2; pre-impact rule: right edge in
      [onset, impact) = alert; impact..+2 s = fall
- [x] SisFall temporal labels = peak-acceleration proxy (stated); KFall
      labels video-grounded (primary); Enhanced annotations unrecoverable
      (null-controlled) — disclosed in text
- [x] Preprocess signature `2574e6104c95` stamped into every corpus/artifact

## Splits and training

- [x] GroupKFold(5) by subject for baselines/ablations; full LOSO headline;
      LODO {A–D} for cross-dataset; inner validation split by subject
- [x] Class weighting inverse-frequency; **no oversampling** (leakage at 75% overlap)
- [x] Seeds fixed (`1337`); Adam 1e-3 → cosine; batch 128; ≤100 epochs;
      early stop on validation macro-F1, patience 15
- [x] Model: separable CNN 7,947 params (arch in `docs/METHODOLOGY.md` §4)
- [x] Deploy model trained on SisFall+KFall+FallAllD; UMAFall excluded
      (no gyroscope) — stated where the deploy model is reported

## Quantisation and deployment

- [x] Full-integer INT8 (int8 in/out); representative dataset = 200 windows
      from the **training split only**
- [x] Desktop FP32-vs-INT8 verification on held-out subjects as a pre-flash
      gate (agreement 0.9944)
- [x] Instance normalisation in the float pipeline (not in-graph) — collapsed
      F1 0.668→0.359 otherwise; firmware re-implements the identical function
- [x] Operating point from the 20-row trial-level sweep, not convenience
- [ ] On-device latency/RAM/current/battery — **PENDING**
      (`experiments/hardware_measurement/`)

## Reporting

- [x] Lead time reported as a distribution (mean/median/std) with pre-impact
      fraction, not a single mean
- [x] False alarms reported per hour of ADL (window-level specificity alone
      called near-meaningless at 7,200 decisions/hour)
- [x] Cross-dataset gap reported without hiding behind within-dataset numbers
- [x] Every manuscript number traces to a committed artifact
      (`manuscript/VERIFICATION.md` practice)
