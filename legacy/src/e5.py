"""e5.py — E5 ablations (plan section 5): window length, sampling rate,
accelerometer-only, sensor placement. All 5-fold subject-grouped CV, macro-F1.
"""
import json
import os
import numpy as np
from sklearn.model_selection import GroupKFold

import config
from src.preprocessing import build_windows, compute_norm, apply_norm, event_labels
from src.utils import (set_seed, record_fold, log_line, binary_metrics,
                       subject_group_val_split, Timer)
from src.model import build_model, train_model

TAG = "E5"


def _cv_binary(X, y, g, epochs, patience, norm=True, label="", n_classes=2):
    """5-fold subject-grouped CV, returns list of fold F1s and mean macro-F1."""
    gkf = GroupKFold(n_splits=5)
    f1s = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, g)):
        if norm:
            mu, sd = compute_norm(X[tr], g[tr])
            Xtr, Xte = apply_norm(X[tr], mu, sd), apply_norm(X[te], mu, sd)
        else:
            Xtr, Xte = X[tr], X[te]
        trmask, vamask = subject_group_val_split(g[tr], config.TRAIN["val_fraction"])
        mdl = build_model(n_classes=n_classes, input_len=X.shape[1],
                          channels=X.shape[2])
        mdl, _ = train_model(mdl, Xtr[trmask], y[tr][trmask],
                             Xtr[vamask], y[tr][vamask],
                             epochs=epochs, patience=patience, verbose=0)
        p = mdl.predict(Xte, verbose=0, batch_size=512)
        prob = p[:, 1] if p.shape[1] > 1 else p
        m = binary_metrics(y[te], prob)
        f1s.append(m["f1"])
        record_fold(TAG, f"{label}-f{fold}", "Proposed", m, extra=label)
    return float(np.mean(f1s)), float(np.std(f1s)), f1s


def run(epochs=None, patience=None, verbose=1):
    set_seed()
    t = Timer(TAG)
    res = {}

    # ---------------- ablation 1: window length (SisFall, 50 Hz, stride 0.5 s)
    for win in [1.0, 1.5, 2.0]:
        w = build_windows("SisFall", win=win, stride=config.STRIDE_SEC)
        f1, sd, _ = _cv_binary(w["X"], event_labels(w), w["subject"],
                               epochs, patience, label=f"win{win}")
        res[f"win_{win}s"] = {"f1": f1, "std": sd}
        if verbose:
            print(f"[E5] window {win}s: F1 {f1:.4f}", flush=True)

    # ---------------- ablation 2: sampling rate (SisFall native 200 Hz)
    for rate in [25.0, 50.0, 100.0]:
        w = build_windows("SisFall", rate=rate, win=2.0, stride=0.5)
        f1, sd, _ = _cv_binary(w["X"], event_labels(w), w["subject"],
                               epochs, patience, label=f"rate{int(rate)}")
        res[f"rate_{int(rate)}Hz"] = {"f1": f1, "std": sd}
        if verbose:
            print(f"[E5] rate {int(rate)}Hz: F1 {f1:.4f}", flush=True)

    # ---------------- ablation 3: accel-only vs accel+gyro
    w = build_windows("SisFall", win=2.0, stride=0.5)
    f1_3, sd3, _ = _cv_binary(w["X"][:, :, :3], event_labels(w), w["subject"],
                              epochs, patience, label="accel-only")
    res["channels_accel3"] = {"f1": f1_3, "std": sd3}
    f1_6, sd6, _ = _cv_binary(w["X"], event_labels(w), w["subject"],
                              epochs, patience, label="accel+gyro")
    res["channels_accel6"] = {"f1": f1_6, "std": sd6}
    if verbose:
        print(f"[E5] accel-only: F1 {f1_3:.4f} | accel+gyro: F1 {f1_6:.4f}")

    # ---------------- ablation 4: sensor placement (FallAllD waist vs wrist)
    # Same subjects wear both IMUs: train on one placement, test on the other.
    ww = build_windows("FallAllD", placement="waist")
    wwr = build_windows("FallAllD", placement="wrist")
    assert set(ww["subject"]) == set(wwr["subject"]), "placements must share subjects"
    gkf = GroupKFold(n_splits=5)
    buckets = {"waist": [], "wrist": []}
    for tr, te in gkf.split(ww["X"], ww["y_post"], ww["subject"]):
        Xtr, ytr = ww["X"][tr], event_labels(ww)[tr]
        gtr = ww["subject"][tr]
        mu, sd = compute_norm(Xtr, gtr)
        Xtr_n = apply_norm(Xtr, mu, sd)
        trmask, vamask = subject_group_val_split(gtr, config.TRAIN["val_fraction"])
        for te_name, te_w in (("waist", ww), ("wrist", wwr)):
            Xte, yte = te_w["X"][te], event_labels(te_w)[te]
            mdl = build_model(n_classes=2)
            mdl, _ = train_model(mdl, Xtr_n[trmask], ytr[trmask],
                                 Xtr_n[vamask], ytr[vamask],
                                 epochs=epochs, patience=patience, verbose=0)
            p = mdl.predict(apply_norm(Xte, mu, sd), verbose=0, batch_size=512)
            m = binary_metrics(yte, p[:, 1])
            buckets[te_name].append(m["f1"])
            record_fold(TAG, f"placement-{te_name}-f{len(buckets[te_name])-1}",
                        "Proposed", m, extra="train waist / test " + te_name)
    res["placement_train_waist_test_waist"] = {"f1": float(np.mean(buckets["waist"])),
                                               "std": float(np.std(buckets["waist"]))}
    res["placement_train_waist_test_wrist"] = {"f1": float(np.mean(buckets["wrist"])),
                                               "std": float(np.std(buckets["wrist"]))}
    if verbose:
        print(f"[E5] placement waist->waist F1 {np.mean(buckets['waist']):.4f} | "
              f"waist->wrist F1 {np.mean(buckets['wrist']):.4f}")

    json.dump(res, open(os.path.join(config.OUT, "e5_summary.json"), "w"), indent=2)
    log_line(TAG, "done")
    t.done()
    return res
