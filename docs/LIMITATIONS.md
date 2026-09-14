# Limitations and risks

Honest list, ordered by how much they matter for the claims.

## 1. On-device numbers are pending, not failed

Inference latency, tensor-arena high-water mark, sensor-to-buzzer latency,
current draw and battery life are **not yet measured on the physical device**.
They are firmware-instrumented and protocoled (`firmware/MEASUREMENT.md`),
and every table shows them as `PENDING`. Do not fill them from the desktop
estimates (~8 KB arena, "should be well under 50 ms"): estimates went in the
arena *sizing* decision, not in the results.

## 2. Wokwi replay is integration evidence, not validation

The simulation runs the exact firmware bytes and a real dataset trial, but
with an idealised sensor: no mounting noise, no skin/fabrics coupling, no
battery. It validates scheduling, preprocessing arithmetic and alarm logic —
nothing more.

## 3. Public datasets ≠ elderly falls

SisFall/KFall/FallAllD/UMAFall are mostly young/middle-aged adults performing
supervised falls onto mats; the SisFall mirror here has only 2 of the 15 older
subjects. Real geriatric falls differ in speed, surface and recovery.
Sensitivity numbers must not be quoted as clinical expectations.

## 4. The false-alarm problem is real and unsolved

Deployment converts every waking moment into 7,200 decisions per hour. At the
pre-impact-optimal point the system would raise ~293 alarms/hour of pure ADL
(E3). Suppressing to a wearable ~1/hour costs most of the lead time
(debounce_analysis: 10.7% pre-impact at 0.44 FA/hr). Any deployed version must
either accept lower lead, add context (e.g. rest/activity gating), or reduce
the decision rate — the stride, not the model, is what has to change.

## 5. Cross-dataset generalisation is limited

Within-dataset accuracy (0.95+) does not transfer: unseen FallAllD 0.34 F1.
Even with the best adaptation (instance norm) the gap remains large and
asymmetric; do not ship configuration trained on other labs' subjects without
local data. This is a *finding* of the work, and constrains the generalisation
claims.

## 6. Single-sensor, waist-only placement

Neck does slightly better, wrist much worse (E5). A waist belt is required;
clip-on or pocket use is a different (unmeasured) domain.

## 7. Detection, not prevention

Even ideal pre-impact detection (~0.5–0.7 s) is only enough to warn; it
cannot physically prevent a fall. Expected benefit is faster response /
alerting caregivers.

## 8. Aggregation risks in summary tables

- Window-level specificity looks impressive (0.99) but is near-meaningless at
  7,200 decisions/hour; always pair it with FA/hour or trial-level
  specificity.
- Lead-time means hide late detections; report medians and the
  pre-impact fraction alongside (they are, in the CSVs).

## 9. The legacy model's demo must not be cited as a result

`models/legacy_synthetic/` was trained on synthetic data. Its Wokwi demo
proves firmware integration only; its metrics are synthetic-pipeline numbers.

## 10. One replayed trial

The committed replay verification uses one KFall trial (S06T20R01). It
demonstrates correct end-to-end behaviour; it does not characterise
sensitivity.

## 11. Threats to validity acknowledged in the evaluation design

SisFall temporal labels are peak-acceleration proxies (the Enhanced
annotations proved unrecoverable); KFall labels are video-grounded. The
pre-impact claim rests on KFall alone — reported, not averaged away.
