"""config.py — single source of truth for the frozen protocol (plan sections 4-6).

Every experiment reads its parameters from here. If you change a number here,
experiment logs record the config hash so results stay traceable.
"""
import hashlib
import json
import os

# ---------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.abspath(__file__))
# FALL_RAW env var lets the Kaggle notebook point at /kaggle/input
RAW = os.environ.get("FALL_RAW") or os.path.join(ROOT, "data", "raw")
SYN = os.path.join(ROOT, "data", "synthetic")    # generated mirror (same formats)
CACHE = os.path.join(ROOT, "data", "cache")      # preprocessed .npz
HIRATE_DIR = os.path.join(CACHE, "hirate")       # canonical, native-decimated arrays
WIN_DIR = os.path.join(CACHE, "windows")         # windowed arrays (default config)
OUT = os.path.join(ROOT, "output")               # results.csv, tables, json
FIG = os.path.join(ROOT, "figures")
MODELS = os.path.join(ROOT, "models")
FW_DIR = os.path.join(ROOT, "firmware", "fall_detector")

for d in (RAW, SYN, CACHE, HIRATE_DIR, WIN_DIR, OUT, FIG, MODELS, FW_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------- datasets
# name: meta used by the synthetic generator AND the real-data parsers.
#   rotation : mounting rotation applied to the canonical frame when the
#              dataset was recorded (documented per plan 4.2). Preprocessing
#              applies the INVERSE to recover canonical axes.
DATASETS = {
    "SisFall": {
        "native_rate": 200.0,
        "subjects_young": 23, "subjects_older": 15,
        "label": "SA" if True else "SA",
        "falls_per_subject": 6, "adl_per_subject": 8,
        "fall_dur": (7.0, 12.0), "adl_dur": (6.0, 14.0),
        "rotation": {"name": "identity", "matrix": [1, 0, 0, 0, 1, 0, 0, 0, 1]},
        "units": "adc",           # raw ADC counts (9 columns: two accels + gyro)
        "placement": "waist",
        "accel_range_g": 16.0,
    },
    "KFall": {
        "native_rate": 100.0,
        "subjects_young": 12, "subjects_older": 20,
        "falls_per_subject": 6, "adl_per_subject": 8,
        "fall_dur": (6.0, 10.0), "adl_dur": (6.0, 12.0),
        "rotation": {"name": "rotz180", "matrix": [-1, 0, 0, 0, -1, 0, 0, 0, 1]},
        "units": "metric",        # already g / deg/s, low-back IMU
        "placement": "lowback",
    },
    "FallAllD": {
        "native_rate": 100.0,
        "subjects_young": 10, "subjects_older": 5,
        "falls_per_subject": 6, "adl_per_subject": 8,
        "fall_dur": (6.0, 10.0), "adl_dur": (6.0, 12.0),
        "rotation": {"name": "rotx90", "matrix": [1, 0, 0, 0, 0, -1, 0, 1, 0]},
        "units": "metric",
        "placement": "waist_and_wrist",
    },
    "UMAFall": {
        "native_rate": 200.0,
        "subjects_young": 14, "subjects_older": 3,
        "falls_per_subject": 6, "adl_per_subject": 8,
        "fall_dur": (6.0, 10.0), "adl_dur": (6.0, 12.0),
        "rotation": {"name": "axis_swap", "matrix": [0, 0, 1, 1, 0, 0, 0, 1, 0]},
        "units": "metric",
        "placement": "waist_and_wrist",
    },
}
ACTIVE_DATASETS = list(DATASETS.keys())

# ---------------------------------------------------------------- preprocessing (plan 4.1)
CHANNELS = 6                # ax ay az gx gy gz
LP_CUTOFF = 20.0            # Hz anti-alias low-pass before decimation
WORK_RATE = 50.0            # Hz
WIN_SEC = 2.0               # seconds -> 100 samples @ 50 Hz
STRIDE_SEC = 0.5            # 75 % overlap
NORM_EPS = 1e-6

# SisFall ADC conversion (simulated Readme constants; real SisFall uses
# ADXL345 4 mg/LSB and ITG3200 14.375 LSB/(deg/s) — same form, put real
# constants in data/raw/<name>/imus.json when using the real release)
SISFALL_IMU = {"accel": {"scale": 0.0039, "offset": 0.0},   # g / LSB
               "gyro": {"scale": 1.0 / 14.375, "offset": 0.0}}  # dps / LSB

# ---------------------------------------------------------------- model (plan 6)
MODEL = {
    "conv1": 24, "conv1_k": 7, "conv1_s": 2,
    "sep1": 48, "sep1_k": 5,
    "sep2": 64, "sep2_k": 3,
    "dense": 32, "dropout": 0.3,
    "use_instnorm": True,      # per-window instance normalisation (first layer)
}

# ---------------------------------------------------------------- training (plan 6)
TRAIN = {
    "batch_size": 128,
    "lr": 1e-3,
    "epochs": 25,              # QUICK default; FULL = 100
    "patience": 6,             # FULL = 15
    "optimizer": "adam",
    "decay": "cosine",
    "val_fraction": 0.12,      # subject-wise validation inside training data
    "loso_epochs": 25,
}
MODE_QUICK = {"epochs": 25, "patience": 6}
MODE_FULL = {"epochs": 100, "patience": 15}

# ---------------------------------------------------------------- experiments
E2_ADAPTERS = ["none", "instnorm", "coral", "dann"]
E2_FOLDS = [("A", ["SisFall", "KFall"], "FallAllD"),
            ("B", ["SisFall", "FallAllD"], "KFall"),
            ("C", ["KFall", "FallAllD"], "SisFall"),
            ("D", ["SisFall", "KFall", "FallAllD"], "UMAFall")]
E3_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
E3_TABLE_THRESHOLDS = [0.5, 0.7, 0.9]
CORAL_LAMBDA = 1.0
DANN_MAX_LAMBDA = 1.0

QUANT = {                     # plan section 9
    "rep_windows": 200,       # representative dataset size (training split only)
    "latency_runs": 1000,
}

# ---------------------------------------------------------------- demo (small) mode
# --small  => tiny but structurally identical datasets (fast end-to-end demo,
#             included in the simulation-ready zip). Full mode = the counts in
#             DATASETS above.
SMALL = {"subjects_young": 5, "subjects_older": 3,
         "falls_per_subject": 3, "adl_per_subject": 4}
SMALL_MODE = False

SEED = 42

# ---------------------------------------------------------------- misc
def config_hash() -> str:
    blob = {"datasets": DATASETS, "model": MODEL, "train": TRAIN,
            "win": WIN_SEC, "stride": STRIDE_SEC, "rate": WORK_RATE}
    return hashlib.md5(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:10]
