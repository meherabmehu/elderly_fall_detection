# What the AI actually adds ("AI impact")

The device carries **two detectors on the same sensor stream**: a classical
two-stage rule baseline (free-fall → impact/rotation thresholds) and the INT8
CNN. This document quantifies what the neural network contributes, measured —
not asserted. All numbers trace to committed artifacts
(`docs/RESULTS_PROVENANCE.md`).

## 1. On the pre-impact task (the product claim)

KFall, 5-fold subject-grouped CV, macro-F1
(`results/kaggle_reference/results_e1_preimpact.csv`):

| Detector | macro-F1 | sensitivity |
|---|---:|---:|
| SMV threshold (2 fitted parameters — the mathematical form of the on-device rule) | **0.3262** | 0.512 |
| **Proposed INT8 CNN (what ships)** | **0.8742** | 0.9361 |

The threshold detector barely works pre-impact: an alarm that must fire
*before* impact cannot wait for the impact spike, and pre-impact motion has
no simple magnitude signature. The CNN's temporal window learns it.

## 2. Against the published KFall benchmark (same protocol, trial-level)

`results/tables/trial_level_eval.md` — Yu et al. (2021) threshold method vs
the CNN scored per trial on the same dataset:

| Method | Sensitivity | Specificity | Lead time |
|---|---:|---:|---:|
| Threshold (Yu 2021 benchmark) | 0.9550 | 0.8343 | 333 ± 160 ms |
| INT8 CNN, this work (nb04 pooled LOSO, t=0.95, k=1) | **0.9962** | **0.9465** | **393 ± 306 ms** |

+4.1 points sensitivity, +11.2 points specificity, longer mean lead — on the
authors' own scoring definition.

## 3. On the physical device, same motion, head-to-head

The rule baseline **runs on the ESP32 alongside the CNN** (`mode cnn|rule|both`),
so the comparison is on identical live samples, not across papers. Two designed
experiments (`firmware/MEASUREMENT.md` §4, blank templates in
`experiments/hardware_measurement/on_device_trials.md`):

- **20 mattress drops** of the device in mixed directions plus near-falls
  (heavy sitting, sofa drops) — which detector fires when;
- **1 hour of ordinary ADL** — measured false alarms/hour for each detector,
  replacing the simulated 293 FA/hr figure with a real one.

PENDING until the bench run. This is the on-device instance of §1's
comparison.

## 4. Inside the deployment pipeline, the AI engineering mattered

The biggest deployment finding is about *where the neural network's
normalisation lives* (`models/final_int8/quantisation_comparison.json`):

| Variant | FP32 F1 | INT8 F1 | INT8 agreement |
|---|---:|---:|---:|
| Instance norm inside the neural graph | 0.668 | **0.359** | 69% |
| Instance norm in the pipeline (as deployed) | 0.701 | **0.704** | 99.4% |

A naive INT8 port would have destroyed 31 points of macro-F1. The deployed
variant loses nothing (INT8 slightly *up*: −0.29 pp) in 22.54 KB of flash —
7,947 parameters inference on a BDT 450 MCU at 50 Hz.

## 5. What the AI does NOT fix (equally important)

- **False alarms**: at deployment-relevant operating points the CNN still
  raises 100–293 alarms/hour of pure ADL activity (E3 window-level); the
  debounce analysis shows the lead-time vs false-alarm trade is today decided
  by the 0.5 s decision rate, not by the model.
- **Cross-dataset shift**: on an unseen dataset F1 falls to 0.34 (FallAllD).
  The rule detector shifts less but starts from much worse.

## Bottom line for the paper

The AI impact is: **detection *before* impact, on a sensor and at a price the
rule baseline cannot operate at** — 0.874 vs 0.326 macro-F1 on the pre-impact
task, exceeding the published benchmark under its own protocol, running on a
22.54 KB INT8 model with 99.4% FP32↔INT8 agreement — while the false-alarm
and generalisation limits are measured and reported, not hidden.
