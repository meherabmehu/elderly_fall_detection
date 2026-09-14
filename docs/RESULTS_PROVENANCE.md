# Results provenance

One row per publishable number: what it is, where it came from, and what it
may be used for. The categories are strict:

- **[KAGGLE-REF]** — measured on real public datasets by the reference Kaggle
  pipeline (`training/`), artifacts in `results/kaggle_reference/`.
- **[DESKTOP-VERIFY]** — produced by tooling in *this* repository from
  committed artifacts; re-runnable (`tools/replay_desktop_check.py`).
- **[HW-VERIFIED]** — observed on the physical ESP32 build (USB-powered).
- **[SIM-ONLY]** — simulation/integration evidence only (Wokwi).
- **[PENDING]** — not measured yet; any table containing it must say so.

---

## Recognition performance (all [KAGGLE-REF])

| Number | Value | Source file |
|---|---|---|
| Proposed model, SisFall post-fall, grouped-5 macro-F1 | 0.9502 ± 0.0063 (LOSO 0.9533) | `results_e1.csv`, `tables/table_I.md` |
| Proposed model, KFall pre-impact, grouped-5 macro-F1 | 0.874 (baselines 0.91–0.92; RF best) | `results_e1_preimpact.csv`, `tables/table_Ib.md` |
| E2 fold A: train SisFall+KFall, test FallAllD | F1 0.344 → 0.388 (instance norm) | `results_e2.csv`, `tables/table_II.md` |
| E2 fold B: train SisFall+FallAllD, test KFall | F1 0.835 → 0.873 (instance norm) | same |
| E2 fold C: train KFall+FallAllD, test SisFall | F1 0.799 → 0.830 (instance norm) | same |
| E2 fold D: test UMAFall (accel-only, pocket) | F1 0.475 (reported once) | same |
| CORAL / DANN | hurt every fold / worse than none on all four | same |
| E3: KFall mean lead @ t=0.5 / 0.7 / 0.9 | 663 / 589 / 449 ms, detection 1.00 / 1.00 / 0.998, FA/hr 293 / 215 / 102 | `results_e3.csv`, `tables/table_III.md`, `e3_operating_points.csv` |
| E3 debounce: ≤1 FA/hr costs most lead time | e.g. (0.95, k=4): 0.44 FA/hr, 10.7% pre-impact | `tables/debounce_analysis.md` |
| Trial-level (Yu-protocol), deployed model | (0.7,k=4): sens 0.913, spec 0.969, lead 722 ms; (0.95,k=1): sens 0.958, spec 0.836, lead 539 ms | `final_model_card.md`, `final_operating_points.csv`, `trial_level_eval.csv` |
| Ablations | 2.0 s best window; 100 Hz best rate (+0.3 F1, not worth RAM); accel-only −1.4 F1; wrist ↓↓ | `results_e5.csv`, `tables/table_ablations.md` |

## Quantisation and model (all [KAGGLE-REF])

| Number | Value | Source |
|---|---|---|
| Parameters | 7,947 | `quantisation_report.json` |
| INT8 size / FP32 size | 22.54 KB (23,080 B) / 163.3 KB | same, `quantisation_comparison.json` |
| FP32 → INT8 macro-F1 delta (held-out subjects) | 0.7109 → 0.7138 (−0.288 pp), agreement 0.9944 | `quantisation_report.json` |
| Tensor arena desktop estimate | 8.0 KB (8,208 B) | same (`estimated_arena_bytes`) |
| In-graph vs pipeline instance norm | F1 0.668→0.359 in-graph vs 0.701→0.704 pipeline | `quantisation_comparison.json` |
| Recommended operating point | threshold 0.7, k=4 | `final_model_card.md` |

## This repository's own verification ([DESKTOP-VERIFY])

| Number | Value | Source |
|---|---|---|
| Final model on KFall S06T20R01 replay | p_fall 0.9961 on impact windows; +100 ms lead at (≤0.9, k=1); fires at all four sweep points | `results/replay_verification/*.json` (re-runnable) |
| Final-model ESP32 firmware compiles | both REPLAY_MODE 1 and 0, PlatformIO espressif32 | build log of `wokwi/final_model/` (not committed binaries) |
| Legacy model replay cross-check | fires on all 7 windows; behaviour matches the 2026-08 Wokwi demo | `legacy_synthetic_thr0.60_k1.json` |

## Physical evidence ([HW-VERIFIED], qualitative)

- I²C scan finds the MPU6050 at `0x68`; WHO_AM_I reads `0x68`; stationary
  acceleration magnitude ≈ 1 g (handoff §6).
- AI-free rule-based detector fires LED + buzzer on fall-like motion; serial
  commands work (handoff §7).
- Final INT8 firmware on the physical build prints class probabilities
  (e.g. `p_bkg=0.066 p_alert=0.781 p_fall=0.156`) and raises
  `FALL ALERT: CNN: alert/fall probability sustained` (handoff §8).
- No *quantitative* on-device numbers (latency, RAM, alarms/hour) exist yet.

## Simulation-only evidence ([SIM-ONLY])

- The 2026-08 Wokwi replay of KFall S06T20R01 ran the **legacy** model and
  fired buzzer/LED with p_fall ≈ 0.89–0.93 as observed then (handoff §9). The
  committed legacy-replay check reproduces the detection behaviour with
  somewhat lower peak probabilities — the legacy demo build is not fully
  recoverable; treat the final-model numbers above as the ones to cite.

## Pending ([PENDING])

| Measurement | Protocol | Record in |
|---|---|---|
| Inference latency (≥1,000 runs) | firmware `measure`, `MEASUREMENT.md` §2 | `experiments/hardware_measurement/hardware_measurements.json` |
| Tensor arena high-water mark | same | same |
| Sensor-to-buzzer end-to-end latency | preprocess+invoke (firmware) or I²C/GPIO probe | same |
| Current draw / battery life | MEASUREMENT.md §3 | same |
| Real false-alarm rate (1 hour ADL wear) | MEASUREMENT.md §4a | `on_device_trials.md` |
| Mattress drop trials CNN vs rule | MEASUREMENT.md §4b | same |
| Final-model Wokwi replay execution | `wokwi/final_model/` build + simulator | new entry when done |

**House rule:** any numbers table in `paper_support/` or a manuscript must
only use rows whose values exist in committed artifacts at commit time, or
explicit `PENDING` cells.
