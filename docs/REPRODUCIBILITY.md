# Reproducibility guide

Three levels, in increasing cost. Start at level 1.

---

## Level 1 — verify the committed claims locally (no hardware, no datasets)

```bash
git clone <this repo> && cd <repo>
python -m venv .venv && .venv/bin/pip install tensorflow

# 1a. Replay verification: the committed INT8 model on KFall S06T20R01.
.venv/bin/python tools/replay_desktop_check.py \
    --model models/final_int8/model.tflite \
    --replay wokwi/final_model/src/replay_data.h \
    --norm instance --threshold 0.95 --k 2
# expect: per-window probabilities matching results/replay_verification/

# 1b. Artifact identity: every copy of the model headers must match.
sha256sum models/final_int8/model.h \
          firmware/esp32_ai_final_live_deployment/model.h \
          wokwi/final_model/src/model.h
# expect: identical digests (see docs/ARTIFACT_INVENTORY.md)

# 1c. Firmware compiles (needs PlatformIO; downloads toolchains on first run)
cd wokwi/final_model && pio run && pio run -e esp32dev_live
```

## Level 2 — hardware verification (~1.5 hours + parts in `docs/HARDWARE.md`)

1. Assemble per `docs/HARDWARE.md`; flash `esp32_initial_hardware_test`
   (expect `0x68`, ≈1 g at rest).
2. Flash `esp32_usb_rule_based_fall_detector`, type `test`, confirm alarm.
3. Flash `esp32_ai_final_live_deployment` (with both headers), pass the
   stationary sanity gate.
4. Wokwi replay: `docs/WOKWI.md` (record the serial log).
5. Bench measurements per `firmware/MEASUREMENT.md`, capture with
   `tools/serial_capture.py`, fold into
   `experiments/hardware_measurement/hardware_measurements.json` via
   `tools/parse_measure_log.py`.

## Level 3 — full training/evaluation reproduction (Kaggle GPU)

The pipeline is designed so that **nothing is clicked in a browser** beyond
the one-time Kaggle account setup.

1. Fetch the four datasets and note your access dates (`datasets/README.md`):
   SisFall and UMAFall come from this repo's [`datasets-v1` release](https://github.com/meherabmehu/elderly_fall_detection/releases/tag/datasets-v1)
   via `python tools/download_datasets.py --extract` (sha256 against
   `datasets/SHA256SUMS.txt`); FallAllD from IEEE DataPort (free account);
   KFall only via official registration — it must not be redistributed.
2. Create the Kaggle datasets listed in `training/PROJECT_PLAN.md` (private is
   fine) and a Kaggle API token (`~/.kaggle/kaggle.json`).
3. Environment:

   ```bash
   cd training
   python -m venv ../.venv && ../.venv/bin/pip install -r requirements.txt
   ```

4. Publish the shared library once, then run the notebooks in order:

   ```bash
   make sync        # scripts/sync_fdlib.py -> Kaggle Dataset arifshekh/fdlib
                    # (create YOUR OWN fdlib dataset slug and edit scripts/)
   make preprocess  # nb01: windows + sanity gates + fig_sanity_traces
   make e1 e2 e3 e5 export
   # final model & pre-impact baselines:
   ../.venv/bin/python scripts/run_kernel.py nb07_final
   ```

   The runner enforces: one kernel at a time, halt on first failure, T4×2 on
   every kernel. Every run appends to `results/kaggle_reference/experiment_log.csv`
   when outputs are pulled back.

5. Rebuild the paper tables from the CSVs: `make tables`.

### Contract checklist (what must be equal on every run)

- preprocess signature `2574e6104c95` in every loaded corpus;
- 50 Hz, 100×6 window, stride 25, 6 channels, 3 classes;
- splits by subject/dataset; class weights (no oversampling);
- INT8 representative dataset from the **training split only**;
- desktop FP32↔INT8 agreement ≥ 0.99 before flashing anything;
- firmware normalisation = `fdlib.preprocess.instance_normalise`
  (per window, float, eps 1e-6).

## What is intentionally NOT reproducible from this repository

- Raw datasets (fetch from sources; see `datasets/README.md`).
- The physical measurements currently marked PENDING — they require the bench
  (that is the entire point of `experiments/hardware_measurement/`).
- The August-2026 Wokwi demo of the *legacy* model's exact console numbers
  (superseded; the committed final-model verification replaces it).
