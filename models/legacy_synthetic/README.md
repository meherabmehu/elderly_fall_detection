# models/legacy_synthetic/ — superseded, do not deploy

The GEN-1 model: separable CNN with 8,219 parameters, 19,928 bytes INT8,
trained on **synthetic** mirror datasets produced by the legacy pipeline's own
generator (`legacy/src/synthetic.py`) and normalised with **frozen global
constants** (`frozen_normalization.json`).

Kept for provenance because it carried the first working end-to-end demo
(August 2026 Wokwi KFall replay). It is superseded by `models/final_int8/`
(real datasets, per-window instance normalisation) for all deployment and
paper claims.

Hard rules:
- Do not report this model's metrics as dataset results (its evaluation
  numbers live in `legacy/results/` and are synthetic-pipeline numbers).
- Do not mix this model with the instance-normalised firmware, or the final
  model with this frozen-normalisation firmware.
- Known latent bug in its `model_config.h`: header comment claims
  ±16 g = 16384 LSB/g (that is the ±2 g scale; ±16 g = 2048 LSB/g). The macro
  was unused by the legacy sketches, so it never executed, but copying that
  header forward would silently mis-scale accelerometer inputs by 8×.

Cross-check artifact: `results/replay_verification/legacy_synthetic_thr0.60_k1.json`.
