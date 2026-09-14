"""preprocessing.py — the SHARED preprocessing module (plan section 4.1).

Used by every experiment, and by the firmware code generator (tools/make_model_h.py)
so the ESP32 applies EXACTLY the same normalisation constants as training
(plan: "if preprocessing differs between training and firmware, the model scores
99% in Python and behaves randomly on the device").

Steps implemented here (plan 4.1):
  1 channel selection      -> parsers deliver [ax ay az gx gy gz] (waist/low-back)
  2 unit conversion        -> parsers (SisFall ADC -> g / deg/s, Readme constants)
  3 anti-alias LPF + decimate to target rate (scipy.signal.decimate, never slicing)
  4 windowing              2.0 s / 0.5 s stride / 75 % overlap, same at inference
  5 labelling              post-fall: window contains impact; pre-impact: window
                            right edge inside [onset, impact)
  6 normalisation          channel-wise mean/std from TRAINING subjects only,
                            frozen to JSON, applied verbatim to val/test/firmware
  7 caching                compressed .npz (raw windows + per-trial meta)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal as sig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.parsers import load_dataset

CH6 = np.array([0, 1, 2, 3, 4, 5])       # ax ay az gx gy gz


# ---------------------------------------------------------------- step 3
def decimate_trial(acc, gyr, fs, target, lp_cutoff=config.LP_CUTOFF):
    """Anti-alias low-pass, then decimate with scipy.signal.decimate."""
    if fs == target:
        sos = sig.butter(4, lp_cutoff, "low", fs=fs, output="sos")
        acc = sig.sosfiltfilt(sos, acc, axis=0)
        gyr = sig.sosfiltfilt(sos, gyr, axis=0)
        return acc, gyr
    q = int(round(fs / target))
    acc = sig.decimate(acc, q, axis=0, ftype="iir")   # filters internally
    gyr = sig.decimate(gyr, q, axis=0, ftype="iir")
    if fs / q != target:  # e.g. 100 -> 25 handled by q; non-integer ratios not expected
        acc = sig.resample_poly(acc, int(target), int(fs / q), axis=0)
        gyr = sig.resample_poly(gyr, int(target), int(fs / q), axis=0)
    return acc, gyr


NATIVE_DIR = os.path.join(config.CACHE, "native")   # sampling-rate ablations
os.makedirs(NATIVE_DIR, exist_ok=True)


# ---------------------------------------------------------------- step 7a
def cache_hirate(datasets=None, force=False):
    """Convert raw files -> canonical-axis, decimated-to-50 Hz arrays, cached per dataset."""
    datasets = datasets or config.ACTIVE_DATASETS
    os.makedirs(config.HIRATE_DIR, exist_ok=True)
    for name in datasets:
        npz = os.path.join(config.HIRATE_DIR, f"{name}.npz")
        if os.path.exists(npz) and not force:
            print(f"[pre] hirate {name}: cached, skip")
            continue
        trials = load_dataset(name, "any")
        Xs, tid = [], []
        meta = []
        for i, t in enumerate(trials):
            acc, gyr = decimate_trial(t.accel, t.gyro, t.rate, config.WORK_RATE) \
                if hasattr(t, "accel") else (None, None)
            X = np.hstack([acc, gyr]).astype(np.float32)
            Xs.append(X)
            tid.append(np.full(len(X), i, dtype=np.int64))
            meta.append({
                "id": i, "dataset": name, "subject": t.subject, "age": t.age,
                "kind": t.kind, "placement": t.placement,
                "rate": config.WORK_RATE,
                "onset_s": None if t.onset is None else t.onset / t.rate,
                "impact_s": None if t.impact is None else t.impact / t.rate,
                "n": len(acc),
            })
        X = np.concatenate(Xs)
        tid = np.concatenate(tid)
        np.savez_compressed(npz, X=X, trial=tid)
        json.dump(meta, open(os.path.join(config.HIRATE_DIR, f"{name}_meta.json"), "w"))
        print(f"[pre] hirate {name}: {X.shape[0]} samples, {len(meta)} trials -> {npz}")


# ---------------------------------------------------------------- sampling-rate ablation cache
def cache_native(datasets=None, force=False):
    """Native-rate (200/100 Hz) canonical arrays — source for the 25/50/100 Hz ablation."""
    for name in (datasets or config.ACTIVE_DATASETS):
        if not os.path.exists(os.path.join(config.SYN, name, "dataset_info.json")) \
                and not os.path.exists(os.path.join(config.RAW, name)):
            print(f"[pre] native {name}: missing, skip")
            continue
        rate = config.DATASETS[name]["native_rate"]
        npz = os.path.join(NATIVE_DIR, f"{name}.npz")
        if os.path.exists(npz) and not force:
            continue
        trials = load_dataset(name, "any")
        Xs, tid, meta = [], [], []
        for i, t in enumerate(trials):
            X = np.hstack([t.accel, t.gyro]).astype(np.float32)
            Xs.append(X)
            tid.append(np.full(len(X), i, dtype=np.int64))
            meta.append({"id": i, "dataset": name, "subject": t.subject,
                         "age": t.age, "kind": t.kind, "placement": t.placement,
                         "rate": t.rate,
                         "onset_s": None if t.onset is None else t.onset / t.rate,
                         "impact_s": None if t.impact is None else t.impact / t.rate,
                         "n": len(X)})
        X = np.concatenate(Xs)
        np.savez_compressed(npz, X=X, trial=np.concatenate(tid))
        json.dump(meta, open(os.path.join(NATIVE_DIR, f"{name}_meta.json"), "w"))
        print(f"[pre] native {name}: {X.shape[0]} samples @ {rate} Hz")


# ---------------------------------------------------------------- step 7b
def build_windows(name, rate=config.WORK_RATE, win=config.WIN_SEC,
                  stride=config.STRIDE_SEC, placement=None, hirate_dir=None):
    """Slide the window across each trial (window 2.0 s, stride 0.5 s by default).

    `rate`: WORK_RATE -> cached decimated arrays; other rates -> native cache
    (only datasets whose native rate >= `rate` are valid, e.g. SisFall for 100 Hz).
    Returns dict with X (n, samples, 6), y3, y_post, y_pre, subject, age,
    kind, dataset, trial_id, t0/t1 (s), onset_s/impact_s (s).
    """
    if rate == config.WORK_RATE and hirate_dir is None:
        hirate_dir = config.HIRATE_DIR
    elif hirate_dir is None:
        hirate_dir = NATIVE_DIR
    npz = np.load(os.path.join(hirate_dir, f"{name}.npz"))
    X_all, tid_all = npz["X"], npz["trial"]
    meta = json.load(open(os.path.join(hirate_dir, f"{name}_meta.json")))
    meta_by_id = {m["id"]: m for m in meta}
    if placement:
        want = {"waist", "lowback"} if placement == "waist" else {placement}
        keep = {m["id"] for m in meta if m["placement"] in want}
    else:
        keep = None
    win_n = int(round(win * rate))
    str_n = int(round(stride * rate))
    rows = []
    for m in meta:
        if keep is not None and m["id"] not in keep:
            continue
        mask = tid_all == m["id"]
        X = X_all[mask]
        fs = m.get("rate", rate)
        if fs != rate:                      # decimate within the cached native rate
            if fs < rate:
                raise ValueError(f"{name}: native {fs} Hz < requested {rate} Hz")
            a, g = decimate_trial(X[:, :3], X[:, 3:], fs, rate)
            X = np.hstack([a, g]).astype(np.float32)
        n = len(X)
        if n < win_n:
            continue
        starts = np.arange(0, n - win_n + 1, str_n)
        for st in starts:
            en = st + win_n
            t0, t1 = st / rate, en / rate
            imp = m["impact_s"]
            ons = m["onset_s"]
            # plan 4.1 step 5 labels (times in seconds, rate-independent)
            y_post = 1 if (imp is not None and t0 <= imp < t1) else 0
            y_pre = 1 if (imp is not None and ons is not None
                          and ons <= t1 < imp) else 0
            if y_post:
                y3 = 2
            elif y_pre:
                y3 = 1
            else:
                y3 = 0
            rows.append((X[st:en], y3, y_post, y_pre, m["subject"], m["age"],
                         m["kind"], name, m["id"], t0, t1,
                         -1.0 if ons is None else ons,
                         -1.0 if imp is None else imp))
    X = np.stack([r[0] for r in rows]).astype(np.float32)
    return {
        "X": X,
        "y3": np.array([r[1] for r in rows], dtype=np.int8),
        "y_post": np.array([r[2] for r in rows], dtype=np.int8),
        "y_pre": np.array([r[3] for r in rows], dtype=np.int8),
        "subject": np.array([r[4] for r in rows]),
        "age": np.array([r[5] for r in rows]),
        "kind": np.array([r[6] for r in rows]),
        "dataset": np.array([r[7] for r in rows]),
        "trial": np.array([r[8] for r in rows]),
        "t0": np.array([r[9] for r in rows], dtype=np.float32),
        "t1": np.array([r[10] for r in rows], dtype=np.float32),
        "onset_s": np.array([r[11] for r in rows], dtype=np.float32),
        "impact_s": np.array([r[12] for r in rows], dtype=np.float32),
    }


def cache_windows(force=False):
    """Cached default-config windows per dataset, WAIST placement only (plan step 7)."""
    os.makedirs(config.WIN_DIR, exist_ok=True)
    for name in config.ACTIVE_DATASETS:
        if not os.path.exists(os.path.join(config.HIRATE_DIR, f"{name}.npz")):
            print(f"[pre] windows {name}: no hirate cache, skip")
            continue
        p = os.path.join(config.WIN_DIR, f"{name}.npz")
        if os.path.exists(p) and not force:
            continue
        w = build_windows(name, placement="waist")
        np.savez_compressed(p, **{k: v for k, v in w.items()})
        print(f"[pre] windows {name}: {len(w['X'])} windows")


def load_windows(datasets=None):
    """Concatenate cached windows for the given datasets (returns the dict)."""
    datasets = datasets or config.ACTIVE_DATASETS
    parts = [build_windows(d) for d in datasets]
    out = {}
    keys = list(parts[0].keys())
    for k in keys:
        if k == "X":
            out[k] = np.concatenate([p[k] for p in parts])
        else:
            out[k] = np.concatenate([p[k] for p in parts])
    return out


# ---------------------------------------------------------------- steps 6/7c
def compute_norm(X, subjects):
    """Channel-wise mean/std from TRAINING subjects only -> (mean(6), std(6))."""
    mask = np.isin(subjects, subjects)  # caller pre-filters; kept explicit
    xs = X[mask]
    mean = xs.mean(axis=(0, 1))
    std = xs.std(axis=(0, 1)) + config.NORM_EPS
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


def event_labels(w):
    """Binary 'fall-event' label consistent with the 3-class merge rule
    (alert ∪ fall -> positive, plan §6). Datasets without temporal labels
    fall back to the impact-containment label (y_post).
    """
    if (w["y3"] == 1).any():
        return (w["y3"] > 0).astype(np.int8)
    return w["y_post"].astype(np.int8)


def save_norm(path, mean, std, subjects, source):
    """Frozen JSON — the 12 numbers firmware uses (plan 4.1 step 6)."""
    json.dump({"mean": mean.tolist(), "std": std.tolist(),
               "channels": ["ax", "ay", "az", "gx", "gy", "gz"],
               "subjects": sorted(set(subjects.tolist())), "source": source},
              open(path, "w"), indent=2)


def load_norm(path):
    d = json.load(open(path))
    return np.array(d["mean"], np.float32), np.array(d["std"], np.float32)


# ---------------------------------------------------------------- sanity checks (plan 4.2)
def sanity_checks(datasets=None, out_fig=None):
    """4.2 checks: resting magnitude 1.0 g, window class counts, axis convention."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    datasets = datasets or config.ACTIVE_DATASETS
    report = {}
    if out_fig:
        fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 3 * len(datasets)),
                                 sharex=False)
    for i, name in enumerate(datasets):
        w = build_windows(name, placement="waist")
        # "resting" = windows of pure-ADL trials with near-constant acceleration
        fall_trials = set(np.unique(w["trial"][w["y3"] > 0]).tolist())
        pure = np.array([t not in fall_trials for t in w["trial"]])
        Xa = w["X"][pure]
        stdw = np.linalg.norm(Xa[:, :, :3], axis=-1).std(axis=1)
        resting = Xa[stdw < 0.06]
        rest_mag = np.linalg.norm(resting[:, :, :3], axis=-1).mean() \
            if len(resting) else np.nan
        counts = {"non-fall": int((w["y3"] == 0).sum()), "alert": int((w["y3"] == 1).sum()),
                  "fall": int((w["y3"] == 2).sum())}
        report[name] = {"windows": int(len(w["X"])),
                        "mean_resting_g": float(rest_mag),
                        "class_counts": counts,
                        "global_abs_max_g": float(np.abs(w["X"][:, :, :3]).max()),
                        "window_shape": list(w["X"].shape)}
        if out_fig:
            # one fall + one ADL trial (canonical axes after rotation check)
            ax = axes[i] if len(datasets) > 1 else axes
            fall_trials = np.unique(w["trial"][w["y3"] == 2])
            trial = fall_trials[0] if len(fall_trials) else np.unique(w["trial"])[0]
            m = w["trial"] == trial
            ax.plot(w["t0"][m], np.linalg.norm(w["X"][m][:, :, :3], axis=-1).mean(axis=1),
                    lw=0.8, label="fall trial |a| (window mean)")
            m2 = w["y3"] == 0
            adl_X = w["X"][m2][:50]
            ax.plot(np.arange(len(adl_X)) * config.STRIDE_SEC,
                    np.linalg.norm(adl_X[:, :, :3], axis=-1).mean(axis=1),
                    lw=0.8, alpha=0.7, label="ADL windows")
            ax.set_title(f"{name} — sanity (axis-convention check)")
            ax.legend(fontsize=7)
    if out_fig:
        plt.tight_layout()
        plt.savefig(out_fig, dpi=110)
        plt.close()
    return report


if __name__ == "__main__":
    cache_hirate(force=True)
    cache_windows(force=True)
    print(json.dumps(sanity_checks(), indent=2))
