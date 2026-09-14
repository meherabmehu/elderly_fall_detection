"""e3.py — E3 pre-impact detection, lead time and false-alarm economics (C2).

Protocol (plan 5, E3):
   primary source      : KFall   (purpose-built temporal labels)
   corroborating       : SisFall Enhanced (Musci annotations)
   primary metric      : lead time ms between the model firing and labelled impact
   secondary           : sensitivity at a fixed false-alarm budget and false
                         alarms per hour of ADL data
   operating point     : threshold sweep -> lead-time vs false-alarm curve
   cross-source note   : both label sources reported separately

Firing rule (same as the firmware): fire when p(alert)+p(fall) >= theta, with a
5 s cooldown; fire time is the right edge of the first firing window.
"""
import json
import os
import numpy as np
from sklearn.model_selection import GroupKFold

import config
from src.preprocessing import build_windows, compute_norm, apply_norm
from src.utils import (set_seed, record_fold, log_line, subject_group_val_split,
                       Timer)
from src.model import build_model, train_model

TAG = "E3"
COOLDOWN_SEC = 5.0


# ------------------------------------------------------------ firing simulator
def fire_times(probs, t1s, impact_t, theta, cooldown=COOLDOWN_SEC, max_lead=3.0):
    """Simulate the online detector on one trial.

    probs : p(alert)+p(fall) per window;  t1s : window right edges (s).
    Returns (fire_t or None, fire_after_impact: bool).
    """
    firings = []
    last = -np.inf
    for p, t1 in zip(probs, t1s):
        if p >= theta and t1 - last >= cooldown:
            firings.append(t1)
            last = t1
    if not firings:
        return None, False
    first = firings[0]
    return first, first > impact_t


def evaluate_fold(model, w_test, theta, subject_filter=None):
    """Run the detector over all test-subject trials; returns metrics.

    Fire time = right edge of the first window with p(alert)+p(fall) >= theta
    (5 s cooldown, same rule as the firmware). Lead time = impact - fire.
    """
    preds = model.predict(w_test["X"], verbose=0, batch_size=512)
    fire_p = preds[:, 1] + preds[:, 2] if preds.shape[1] == 3 else preds[:, 1]
    leads, miss, late = [], 0, 0
    fa_events = 0
    adl_seconds = 0.0
    trids = np.unique(w_test["trial"])
    for t in trids:
        m = w_test["trial"] == t
        subj = w_test["subject"][m][0]
        if subject_filter is not None and subj not in subject_filter:
            continue
        impact_t = float(w_test["impact_s"][m][0])
        if impact_t > 0:                     # fall trial with temporal label
            ft, after = fire_times(fire_p[m], w_test["t1"][m], impact_t, theta)
            if ft is None:
                miss += 1
            elif after:
                late += 1
            else:
                leads.append((impact_t - ft) * 1000.0)
        else:                                # ADL trial -> false-alarm economics
            dur = float(w_test["t1"][m][-1] - w_test["t0"][m][0]) if len(m) else 0.0
            adl_seconds += max(dur, 0.0)
            ft, _ = fire_times(fire_p[m], w_test["t1"][m], np.inf, theta)
            if ft is not None:
                fa_events += 1
    det = len(leads)
    falls_total = det + miss + late
    h = adl_seconds / 3600.0 if adl_seconds > 0 else 1e-9
    return {
        "theta": theta,
        "detected": det, "missed": miss, "late": late,
        "total_falls": falls_total,
        "detection_rate": det / falls_total if falls_total else np.nan,
        "mean_lead_ms": float(np.mean(leads)) if leads else np.nan,
        "std_lead_ms": float(np.std(leads)) if leads else np.nan,
        "median_lead_ms": float(np.median(leads)) if leads else np.nan,
        "n_leads": len(leads),
        "fa_per_hour": fa_events / h,
        "adl_hours": h,
    }


def run(dataset="KFall", splits="grouped5", epochs=None, patience=None,
        verbose=1, thresholds=None):
    """Train a pre-impact (3-class) model on `dataset`, evaluate lead times."""
    set_seed()
    t = Timer(TAG + "-" + dataset)
    thresholds = thresholds or config.E3_THRESHOLDS
    w = build_windows(dataset, placement="waist")
    X, y3, g = w["X"], w["y3"], w["subject"]
    print(f"[E3] {dataset}: {len(X)} windows, {len(np.unique(g))} subjects")

    # global registry of trial meta (filled per fold below)
    subjects = np.unique(g)
    if splits == "loso":
        fold_iter = [(s, np.where(g == s)[0], np.where(g != s)[0])
                     for s in subjects]
    else:
        gkf = GroupKFold(n_splits=5)
        fold_iter = [(f"s{i}", te, tr) for i, (tr, te) in
                     enumerate(gkf.split(X, y3, g))]

    metrics = {th: [] for th in thresholds}
    all_rows = []
    for name, te, tr in fold_iter:
        mu, sd = compute_norm(X[tr], g[tr])
        Xtr, Xte = apply_norm(X[tr], mu, sd), apply_norm(X[te], mu, sd)
        trmask, vamask = subject_group_val_split(g[tr], config.TRAIN["val_fraction"])
        mdl = build_model(n_classes=3)
        mdl, _ = train_model(mdl, Xtr[trmask], y3[tr][trmask],
                             Xtr[vamask], y3[tr][vamask],
                             epochs=epochs, patience=patience, verbose=0)
        subj_filter = set(g[te])
        w_test = {k: v[te] for k, v in w.items()}
        for th in thresholds:
            r = evaluate_fold(mdl, w_test, th, subject_filter=subj_filter)
            metrics[th].append(r)
        record_fold(TAG, f"{dataset}-{name}", "proposed-3class", {
            "sens": np.nan, "spec": np.nan, "f1": np.nan, "auc": np.nan,
            "params": mdl.count_params()}, extra=f"started {name}")
        if verbose:
            print(f"  fold {name}: n_test_subjects={len(subj_filter)}", flush=True)

    summary = {}
    for th in thresholds:
        rs = metrics[th]
        summary[str(th)] = {
            "mean_lead_ms": float(np.nanmean([r["mean_lead_ms"] for r in rs])),
            "std_lead_ms": float(np.nanmean([r["std_lead_ms"] for r in rs])),
            "median_lead_ms": float(np.nanmean([r["median_lead_ms"] for r in rs])),
            "detection_rate": float(np.nanmean([r["detection_rate"] for r in rs])),
            "fa_per_hour": float(np.nanmean([r["fa_per_hour"] for r in rs])),
            "adl_hours": float(np.nansum([r["adl_hours"] for r in rs])),
        }
    json.dump({"dataset": dataset, "splits": splits, "summary": summary},
              open(os.path.join(config.OUT, f"e3_{dataset}_summary.json"), "w"),
              indent=2)
    log_line(TAG, f"done {dataset} {splits}")
    t.done()
    return summary
