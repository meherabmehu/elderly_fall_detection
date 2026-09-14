# Experiments and evaluation design

All experiment *results* live under `results/`; this file explains what each
experiment is, why it exists, and what evidence it produced. Cross-references
use the file paths, so every claim stays one lookup away from its data.

## E0 — dataset reality probe (`nb00`)

Before trusting any dataset, probe it. Findings that changed the design:
SisFall Enhanced annotations unrecoverable (null-controlled); SisFall mirror
has 25/38 subjects; UMAFall waist channel = 20 Hz (below working rate),
pocket smartphone = 200 Hz without gyro. Output:
`results/kaggle_reference/nb00_probe_report.md`.

## E1 — within-dataset baselines (`nb02`, `nb07` Part A)

Purpose: prove competitiveness before claiming anything about generalisation.
Six comparators under 5-fold subject-grouped CV; proposed model also full
LOSO. Two variants: post-fall task on SisFall (`results_e1.csv`,
`tables/table_I.md`) and pre-impact task on KFall (`results_e1_preimpact.csv`,
`tables/table_Ib.md`). Result: saturation — every learned model ≥ 0.95 macro
F1 within-dataset; the proposed 7,947-param model is not the largest but is
within ~2.7 points of the 102k-param CNN at 7.8% of its size.

## E2 — leave-one-dataset-out (`nb03`) — contribution C1

Train on two datasets, test on a third; no target labels, ever. Adaptation
ladder in ascending cost: none → instance-norm → CORAL → DANN. Results in
`results_e2.csv` / `tables/table_II.md`:

- Untrained-domain F1: FallAllD 0.344, KFall 0.835, SisFall 0.799, UMAFall
  0.475 (the last also a placement + sensor-modality shift).
- **Only** instance normalisation helps (~+3–4 pp on the three hard folds).
  CORAL hurts everywhere; DANN is worse than nothing on all four folds.
- Conclusion: the cross-dataset gap is large, asymmetric, and cheap training-
  pipeline normalisation goes further than classical domain adaptation here.

## E3 — pre-impact detection (`nb04`) — contribution C2

Metric of record: lead time (ms between first alarm and labelled impact),
reported as a distribution over 2,346 KFall fall trials under pooled LOSO.
Outputs: `results_e3.csv`, `e3_lead_times_windows.csv`, `e3_operating_points.csv`,
`tables/table_III.md`, `tables/trial_level_eval.md`, `tables/debounce_analysis.md`.

Headline (window threshold sweep): 663 ms mean lead @ t=0.5 with 89–92% of
detected falls pre-impact — at ~293 false alarms/hour; 449 ms @ t=0.9 at
102 FA/hr. The debounce analysis shows ≤ 1 FA/hr costs nearly all lead time
(10.7% pre-impact at 0.44 FA/hr): a detector that is pleasant to wear today
would largely revert to post-impact timing. This is stated, not hidden.

## E4 — quantisation & export (`nb06`, `nb07` Part B)

Full-integer INT8; desktop FP32-vs-INT8 verification on held-out subjects;
the in-graph vs pipeline normalisation comparison that decided the deployment
contract. Outputs: `quantisation_report.json`, `quantisation_comparison.json`,
`model.tflite` + `model.h` (committed), `final_model_card.md`
(operating-point sweep → default deployment recommendation 0.7/k=4).

## E5 — ablations (`nb05`)

- Window length: 2.0 s clearly best pre- and post-impact.
- Sampling rate: 100 Hz +0.3 F1 over 50 Hz — rejected (doubles inference
  compute & RAM); 25 Hz noticeably worse.
- Channels: accelerometer-only −1.4 F1 — the MPU6050's gyro is justified.
- Placement (FallAllD): waist 0.769, neck 0.815, wrist 0.732 — waist is a
  reasonable, not optimal, placement; wrist is poor.

## E6 — hardware measurement programme (PENDING)

Protocol: `firmware/MEASUREMENT.md`. Target record:
`experiments/hardware_measurement/hardware_measurements.json` (latency,
arena high-water, sensor-to-buzzer latency, current, battery),
`on_device_trials.md` (1-hour ADL false-alarm rate; 20-drop mattress test,
CNN vs rule). These fill Table IV's `PENDING-HW` cells and become the
"on-device trials" results subsection when done.

## E7 — controlled functional trial protocol (future, PENDING)

A safe, supervised, soft-surface trial protocol on real participants is a
future institutional (ethics-reviewed) step, not something to improvise.
Until then, all "real use" claims are limited to: bench checks, simulated
motion, datasets and Wokwi integration.
