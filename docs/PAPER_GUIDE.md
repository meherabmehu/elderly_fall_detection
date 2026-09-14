# Conference paper guide

How this repository maps onto the conference submission, and what is and is
not safe to claim today.

## Contributions and their evidence

| Contribution | Where proven | Status |
|---|---|---|
| **C1** — honest leave-one-dataset-out generalisation; instance-normalisation as the only effective adaptation | `results/tables/table_II.md` (`results_e2.csv`) | complete (KAGGLE-REF) |
| **C2** — pre-impact detection with lead-time distribution + honest false-alarm analysis | `results/tables/table_III.md`, `trial_level_eval.md`, `debounce_analysis.md` | complete (KAGGLE-REF) |
| **C3** — INT8 TinyML deployment verified to desktop level; on-device measurement programme defined | `models/final_int8/quantisation_report.json`, `firmware/MEASUREMENT.md`, this repo's firmware/Wokwi | **partial** — latency/RAM/battery PENDING |
| Baseline comparison within-dataset | `results/tables/table_I.md` + `table_Ib.md` | complete |
| CNN vs on-device rule baseline | `results/replay_verification/` + firmware `mode both` trial template | protocol ready, device drops pending |

## Paper-material locations

- Manuscript source (IEEEtran): `paper_support/manuscript/ieee/main.tex` with
  `refs.bib` and figures; build with `tectonic -X compile main.tex`.
- `# Table` Markdown versions: `results/tables/` (+ `paper_support/tables/`
  for derived tables: operating points, hardware resources, datasets).
- Figures: `paper_support/figures/` (lead time, sanity traces, alert
  intervals, system diagram).
- Number-audit practice: `paper_support/manuscript/VERIFICATION.md` documents
  the three errors caught and fixed in a full pass over the draft — the same
  discipline applies to any new number (a traceable artifact or `PENDING`).

## What is safe to claim now

- All E1/E2/E3/E5 dataset results (Kaggle reference runs; subjects/datasets/
  seeds logged per run in `experiment_log.csv`).
- Model size/params/quantisation deltas (committed artifacts).
- Hardware bring-up, AI-free baseline, and final INT8 firmware boot-and-run
  on the physical build (qualitative functional claims).
- Desktop replay verification of the final model on KFall S06T20R01
  (this repo, re-runnable) and compile-verified firmware.
- The deployment finding: in-graph instance normalisation collapses under
  INT8 (−31 F1 points); pipeline normalisation keeps the delta ≤ 0.29 pp.

## What is NOT safe to claim until measured

- Any millisecond of on-device latency, any byte of measured RAM, any mA of
  current, any hour of battery.
- "Works in real-world conditions" — needs the E6/E7 programme.
- Any accuracy on elderly populations specifically.

## Suggested venues / scope (from the project plan)

Regional IEEE conferences (ICCIT, ICECE, ICAEEE, STI, TENSYMP); if the
hardware programme completes strongly, MDPI *Sensors* / IEEE *Access* are
plausible journal targets (confirm APCs and fit with the supervisor).
