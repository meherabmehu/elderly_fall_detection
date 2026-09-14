# Deployment resources (paper Table IV, current state)

Sources: `models/final_int8/quantisation_report.json`,
`results/kaggle_reference/table_IV.md`. Pending cells follow
`experiments/hardware_measurement/hardware_measurements.json` (nulls).

| Metric | Value | Budget | Status |
|---|---|---:|---|
| Model parameters | 7,947 | — | measured (artifact) |
| INT8 model size (flash) | 22.54 KB (23,080 B) | ≤ 60 KB | measured (artifact) |
| FP32 model size | 163.3 KB | — | measured (artifact) |
| Tensor arena | ~8.0 KB estimated (24 KB allocated) | ≤ 120 KB | **PENDING-HW** high-water mark |
| FP32 → INT8 macro-F1 delta | −0.29 pp (0.7109 → 0.7138) | ≤ 2 pp | verified on desktop (held-out subjects) |
| FP32 ↔ INT8 output agreement | 99.44% | — | verified on desktop |
| Inference latency | PENDING-HW | < 50 ms | not yet measured on device |
| End-to-end latency (preprocess + invoke) | PENDING-HW | < 100 ms | firmware-instrumented, not yet measured |
| Mean current / battery life | PENDING-HW | — | not yet measured |

Note: −0.29 pp means the INT8 model scored *slightly higher* than FP32 on the
held-out subjects (noise in favour of int8); the budget is a two-point loss,
and the comparison that actually matters for deployment risk is the 99.44%
output agreement.

Recipe when the bench run exists: `firmware/MEASUREMENT.md` →
`tools/serial_capture.py` → `tools/parse_measure_log.py` →
`training/scripts/build_tables.py` regenerates `results/tables/table_IV.md`.
