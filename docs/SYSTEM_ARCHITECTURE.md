# System architecture

## 1. System context

```text
        training / evaluation                    deployment / verification
┌───────────────────────────────┐        ┌────────────────────────────────┐
│ 4 public datasets (never in   │        │ ESP32 DevKit V1                │
│ this repo; fetch scripts/doc) │        │  ├─ MPU6050 (waist, I2C 0x68)  │
│            │                  │        │  ├─ red LED (GPIO 2, 220 Ω)    │
│            v                  │        │  ├─ active buzzer (GPIO 4)     │
│ fdlib preprocessing (7 steps, │        │  ├─ TFLite Micro, INT8 CNN     │
│ signature 2574e6104c95)       │        │  └─ rule-based baseline        │
│            │                  │        └────────────┬───────────────────┘
│            v                  │                     │ same contract
│ window corpus (100x6 @ 50 Hz) │        ┌────────────v───────────────────┐
│            │                  │        │ Wokwi simulation (integration) │
│            v                  │        │  ├─ live MPU6050 part          │
│ Kaggle nb00–nb07              │        │  └─ KFall S06T20R01 replay     │
│ E1|E2|E3|E5 baselines,        │        └────────────────────────────────┘
│ pre-impact, ablations         │                     │
│            │                  │        ┌────────────v───────────────────┐
│            v                  │        │ tools/replay_desktop_check.py  │
│ nb06/nb07 export +            │        │ desktop replay verification  │
│ desktop INT8 verification     │        │ (committed numbers)          │
└────────────┬──────────────────┘        └────────────────────────────────┘
             │
             v
   models/final_int8/  (model.tflite 23,080 B, model.h, card, quantisation reports)
```

## 2. On-device dataflow (final firmware)

```text
┌──────────┐  I2C 400 kHz   ┌───────────────────────────────────────────────┐
│ MPU6050  │ ─────────────> │ 50 Hz hardware timer ISR -> sample flag       │
│ ±16 g    │   14 bytes     │ ring buffer int16[100][6]                     │
│ ±2000dps │                │ every 25 new samples:                         │
└──────────┘                │   1. raw counts -> g, dps                     │
                            │   2. per-window instance norm (float, eps 1e-6)│
                            │   3. quantise by input scale/zero-point       │
                            │   4. TFLite Micro Invoke() (INT8)             │
                            │   5. dequantise p_bkg/p_alert/p_fall          │
                            │   6. p_alarm = p_alert + p_fall               │
                            │   7. >= threshold for k consecutive windows?  │
                            │      -> LED + buzzer (3 s hold, 5 s cooldown) │
                            │ parallel: rule baseline (free-fall -> impact/ │
                            │ rotation) on the same samples; counters and   │
                            │ timers for the `measure` command              │
└─────────────────────────────────────────────────────────────────────────┘
```

## 3. Memory and compute budget (GEN-2)

| Resource | Budget | Expected | Status |
|---|---:|---:|---|
| Model flash | ≤ 60 KB | 22.54 KB (23,080 B) | verified (artifact size) |
| Tensor arena | ≤ 120 KB | ~8.0 KB desktop estimate; firmware allocates 24 KB | estimate; high-water mark PENDING |
| Inference latency | ≤ 50 ms | — | PENDING (firmware-instrumented) |
| End-to-end (preprocess + invoke) | ≤ 100 ms | — | PENDING |
| Decision rate | — | fixed 0.5 s stride | contract |

## 4. Wiring / pin map

Authoritative tables and diagrams live in `docs/HARDWARE.md` and `hardware/`
(same pins everywhere: SDA 21, SCL 22, buzzer 4, LED 2 via 220 Ω, AD0→GND,
3V3-only sensor power, common grounds; USB power for the verified build,
TP4056+LiPo as the documented next step).

## 5. Operating points documented for the final model

From the 814-fall-trial sweep (`models/final_int8/final_model_card.md`,
`final_operating_points.csv`):

| Use | threshold, k | sens | spec | mean lead |
|---|---|---:|---:|---:|
| model-card recommendation | 0.70, 4 | 0.913 | 0.969 | 722 ms |
| firmware default (demo-robust) | 0.95, 2 | 0.886 | 0.975 | 302 ms |
| pre-impact optimal | 0.95, 1 | 0.958 | 0.836 | 539 ms |

All switchable live over serial (`thr`, `k`) without reflashing.
