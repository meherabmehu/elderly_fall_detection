# tools/

Small, dependency-light utilities for verification and measurement sessions.

| Script | What it does | Verification value |
|---|---|---|
| `replay_desktop_check.py` | Replays KFall `S06T20R01` (from `wokwi/final_model/src/replay_data.h`) through the committed INT8 models using the *exact* firmware preprocessing (window → per-window instance normalisation in float → INT8 quantise → invoke). Prints per-window `p_bkg / p_alert / p_fall`, alarm windows and lead time against the labelled impact. | Desktop verification that the model + preprocessing contract fires correctly on a real fall trial, using the same INT8 arithmetic the ESP32 runs. Runs anywhere Python + TensorFlow is available — no hardware needed. |
| `serial_capture.py` | Logs an ESP32 serial session (boot banner, `measure` blocks, `FALL ALERT` lines) to a timestamped file, with optional commands sent after connect. | Produces the provenance file for every bench measurement. |
| `parse_measure_log.py` | Extracts the last complete `MEASURE` block from a captured log and merges it into `experiments/hardware_measurement/hardware_measurements.json`. Fields it cannot read stay `null` (PENDING) — nothing is invented. | Turns raw serial output into the Table IV hardware numbers, without hand transcription. |

## Setup

```bash
python -m venv .venv && .venv/bin/pip install tensorflow pyserial
```

TensorFlow is only needed for `replay_desktop_check.py`; pyserial only for the
two hardware tools.

## Typical session

```bash
# 1. desktop-check the committed model on the replayed fall trial
.venv/bin/python tools/replay_desktop_check.py \
    --model models/final_int8/model.tflite \
    --replay wokwi/final_model/src/replay_data.h \
    --norm instance --threshold 0.95 --k 2

# 2. capture a bench session on the physical device (flashed with
#    firmware/esp32_ai_final_live_deployment)
.venv/bin/python tools/serial_capture.py --port COM5 --send measure --duration 600

# 3. fold the measurement into the pending-results template
.venv/bin/python tools/parse_measure_log.py logs/serial_YYYYMMDD_HHMMSS.log
```

Recorded outputs belong in `results/replay_verification/` (desktop) and
`experiments/hardware_measurement/` (device), so every reported number keeps a
chain back to the run that produced it.
