# Desktop replay verification — final INT8 model on KFall S06T20R01

**Status: VERIFIED (desktop, INT8 arithmetic identical to the device).**
**Method, not a hardware measurement.** These numbers are produced by
`tools/replay_desktop_check.py` from the committed artifacts
(`models/final_int8/model.tflite`, `wokwi/final_model/src/replay_data.h`)
running the *same* preprocessing the firmware runs — 100×6 window, per-window
instance normalisation in float, INT8 quantisation with the model's own scale
and zero point, then `Interpreter.invoke()`. They verify that the final model
and its preprocessing contract actually detect a real fall trial end-to-end.
They are **not** a substitute for the pending on-device measurements.

## Trial

KFall `S06T20R01` — forward fall, video-grounded labels from
`SA06_label.xlsx`:

| Label | Native frame (100 Hz) | 50 Hz working-rate sample |
|---|---|---|
| Fall onset | 130 | 65 |
| Fall impact | 208 | 104 |

268 samples at 50 Hz → 7 inference windows (stride 25 samples).

## Per-window model output (final INT8 model)

From `final_int8_thr0.95_k2.json` — identical for every operating point:

| Window (start) | Right edge (sample) | Time of edge | p_bkg | p_alert | p_fall |
|---:|---:|---|---:|---:|---:|
| 0   | 99  | 1.98 s | 0.0781 | **0.9219** | 0.0000 |
| 25  | 124 | 2.48 s | 0.0000 | 0.0000 | **0.9961** |
| 50  | 149 | 2.98 s | 0.0000 | 0.0000 | **0.9961** |
| 75  | 174 | 3.48 s | 0.0039 | 0.0000 | **0.9961** |
| 100 | 199 | 3.98 s | 0.1328 | 0.0000 | 0.8633 |
| 125 | 224 | 4.48 s | **0.9961** | 0.0039 | 0.0039 |
| 150 | 249 | 4.98 s | **0.9844** | 0.0156 | 0.0000 |

The model flags the pre-impact segment as ALERT at the first full window
(right edge 99, five samples — 100 ms — before the labelled impact), scores
the impact and post-impact segment 0.9961 FALL across three windows, and
returns decisively to background afterwards. No window outside the event is
above threshold at any operating point tested.

## Alarm behaviour as the operating point changes

Lead time = milliseconds between the **first alarm's** right edge and the
labelled impact at sample 104 (positive = before impact).

| Threshold | k | Alarms in trial | First alarm right edge | Lead time |
|---:|---:|---:|---:|---:|
| 0.70 | 1 | 5 | 99  | **+100 ms (pre-impact)** |
| 0.90 | 1 | 4 | 99  | **+100 ms (pre-impact)** |
| 0.70 | 4 † | 1 | 174 | −1400 ms |
| 0.90 | 3 | 1 | 149 | −900 ms |
| 0.95 | 1 | 3 | 124 | −400 ms |
| 0.95 | 2 ‡ | 1 | 149 | −900 ms |

† operating point recommended in `models/final_int8/final_model_card.md` for
trial-level balance (sensitivity 0.913 / specificity 0.969 / mean lead 722 ms
across 814 held-out fall trials).
‡ firmware default (`firmware/esp32_ai_final_live_deployment`).

On this single trial the alert window at edge 99 exceeds 0.9 but not 0.95, so
thresholds ≥ 0.95 cannot catch it pre-impact regardless of *k*; thresholds ≤
0.9 with *k* = 1 fire 100 ms before the labelled impact. This mirrors the
trial-level finding (E3 / debounce analysis): the threshold–*k* pair trades
lead time against false alarms, and one trial cannot select the operating
point — the 814-trial sweep does that. **This table is an integration check of
one trial, not an accuracy claim.**

## Legacy model cross-check

`legacy_synthetic_thr0.60_k1.json` replays the same trial through the
superseded synthetic-trained model (19,928 bytes, frozen global normalisation).
It fires on all 7 windows (first at edge 99) with `p_alert ≈ 0.5`,
`p_fall ≈ 0.5–0.61`. The Wokwi demo that shipped with the legacy project
reported `p_fall` up to ~0.93 on this trial; the replay through the committed
legacy artifact reproduces the *behaviour* (trial detected, alarm raised) but
not the exact probability values — the legacy demo build is not fully
recoverable, which is partly why the legacy model is superseded rather than
silently kept. The final model's outputs above are stronger and are the ones
to cite.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install tensorflow
.venv/bin/python tools/replay_desktop_check.py \
    --model models/final_int8/model.tflite \
    --replay wokwi/final_model/src/replay_data.h \
    --norm instance --threshold 0.95 --k 2 \
    --out results/replay_verification/final_int8_thr0.95_k2.json
```

## What remains pending

Running this exact replay **on the ESP32 / in Wokwi** (firmware in
`wokwi/final_model/`, which compiles for both REPLAY_MODE=1 and 0) is still a
simulation/bench execution step. Until that run is captured, treat the
on-device replay as *integration-ready, execution pending* — see
`docs/RESULTS_PROVENANCE.md`.
