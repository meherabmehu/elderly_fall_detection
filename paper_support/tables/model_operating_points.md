# Deployed model — trial-level operating points

Source: `results/kaggle_reference/final_operating_points.csv` (nb07 final
model, trial-level evaluation on held-out subjects: 814 fall trials, 1,136 ADL
trials). Same protocol as `results/tables/trial_level_eval.md`.
`sens_trial` = fraction of fall trials with at least one alarm; `spec_trial` =
fraction of ADL trials with no alarm; lead time is the mean/alarm-trial
(negative would be post-impact; not present here).

| threshold | k | sensitivity | specificity | mean lead (ms) | pre-impact fraction | Youden's J | Use |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.70 | 4 | 0.9128 | 0.9692 | 722.0 | 0.184 | **0.8820** | model-card recommendation |
| 0.95 | 2 | 0.8857 | 0.9754 | 302.4 | 0.162 | 0.8611 | **firmware default (demo-robust)** |
| 0.90 | 3 | 0.8759 | 0.9815 | 362.4 | 0.118 | 0.8574 | high-specificity option |
| 0.90 | 2 | 0.9300 | 0.9261 | 539.1 | 0.366 | 0.8561 | balanced |
| 0.95 | 1 | 0.9582 | 0.8363 | 539.5 | 0.682 | 0.7945 | pre-impact optimal |
| 0.90 | 1 | 0.9865 | 0.6778 | 822.6 | 0.851 | 0.6643 | max sensitivity, alarm-heavy |
| 0.95 | 4 | 0.6744 | 0.9991 | 140.0 | 0.004 | 0.6735 | near-zero FA, largely post-impact |

Rows with pre-impact fraction < 0.05 are effectively post-impact detectors —
the debounce analysis' central finding (`results/tables/debounce_analysis.md`).
The full 20-row sweep is in the CSV.

**Caveat for the paper:** these figures are for the deployed (three-corpus)
model on its held-out subjects; the earlier KFall-only pooled-LOSO run in
`trial_level_eval.md` (2,346 fall trials, e.g. 0.9787/0.9941/229.2 ms at
0.95/k=2) scores easier because it is evaluated within one homogeneous
corpus — do not conflate the two (this exact mistake is item 2 in
`manuscript/VERIFICATION.md`).
