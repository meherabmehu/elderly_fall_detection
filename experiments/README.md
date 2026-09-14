# experiments/ — pending hardware measurement programme

This folder exists because the project deliberately does **not** invent
bench numbers. Everything here is a protocol plus an empty/null template that
the physical measurements will fill.

## hardware_measurement/

- `hardware_measurements.json` — the four Table IV numbers (inference
  latency, arena high-water mark, end-to-end latency, battery life, mean
  current) as `null` = **PENDING**. Field list matches
  `firmware/MEASUREMENT.md` §5.
- `on_device_trials.md` — blank tables for the two experiments that are worth
  more than Table IV: the 1-hour ADL false-alarm measurement (replaces the
  simulated 293 FA/hr figure with a real one) and the 20-drop mattress test
  (CNN vs rule on the same physical motion).

## Workflow when the bench session happens

```bash
# flash firmware/esp32_ai_final_live_deployment per docs/DEPLOYMENT.md
python tools/serial_capture.py --port COM5 --send measure --duration 600
python tools/parse_measure_log.py logs/serial_*.log \
    --measurements experiments/hardware_measurement/hardware_measurements.json
```

Until these files contain real values, every latency/RAM/current/battery cell
in every table stays `PENDING`.
