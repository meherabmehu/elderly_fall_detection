"""e1.py — E1 within-dataset baseline on SisFall Enhanced (plan section 5).

Protocol:
  comparators (SMV, SVM, RF, 1D-CNN, CNN-LSTM, proposed): 5-fold subject-grouped CV
  proposed model additionally: full 38-fold LOSO (the headline protocol)
Metric convention: binary post-fall classification, macro-averaged where multi-class.
"""
import json
import os
import numpy as np
from sklearn.model_selection import GroupKFold

import config
from src.preprocessing import build_windows, event_labels
from src.utils import (set_seed, record_fold, log_line, binary_metrics,
                       macro_f1, subject_group_val_split, Timer)
from src.model import build_model, build_cnn, build_cnn_lstm, train_model
import src.baselines as bl

TAG = "E1"


def _prep():
    w = build_windows("SisFall", placement="waist")
    # merged binary labels: the 3-class model is trained on y3; for the E2-style
    # comparators we use the binary post-fall label (plan: 3-class -> merge 1|2)
    return w


def run(epochs=None, patience=None, do_loso=False, verbose=1):
    set_seed()
    t = Timer(TAG)
    w = _prep()
    X, y3, g = w["X"], w["y3"], w["subject"]
    # binary label = fall EVENT (alert ∪ fall), consistent with the 3-class
    # model's merged decision — plan §6 (a pre-impact firing is a detection,
    # not a false alarm; E3 reports the timing separately)
    y = event_labels(w)
    print(f"[E1] windows={len(X)} subjects={len(np.unique(g))} "
          f"events={int(y.sum())} adl={int((1-y).sum())}")

    # ---- frozen stats for the whole experiment (training-fold statistics are
    #      recomputed per fold; here stats are per-fold inside the loop)
    results = {name: [] for name in
               ["SMV", "SVM", "RF", "1D-CNN", "CNN-LSTM", "Proposed"]}

    gkf = GroupKFold(n_splits=5)
    for fold, (tr, te) in enumerate(gkf.split(X, y, g)):
        tr_s, te_s = set(g[tr]), set(g[te])
        assert not (tr_s & te_s), "leak: subject in both train and test!"
        # per-fold normalisation constants (training subjects only, plan 4.1.6)
        from src.preprocessing import compute_norm, apply_norm, save_norm
        mu, sd = compute_norm(X[tr], g[tr])
        Xtr = apply_norm(X[tr], mu, sd)
        Xte = apply_norm(X[te], mu, sd)

        # ---------- SMV threshold
        pred, thr = bl.smv_predict(Xtr, y[tr], Xte)
        m = binary_metrics(y[te], pred)
        m["params"] = 0
        results["SMV"].append(m); record_fold(TAG, f"f{fold}", "SMV", m,
                                              extra=f"thr={thr:.2f}")

        # ---------- SVM
        clf = bl.fit_svm(Xtr, y[tr])
        p = bl.svm_predict(clf, Xte)
        m = binary_metrics(y[te], p)
        m["params"] = "~1k"
        results["SVM"].append(m); record_fold(TAG, f"f{fold}", "SVM", m)

        # ---------- Random Forest
        clf = bl.fit_rf(Xtr, y[tr])
        p = bl.rf_predict(clf, Xte)
        m = binary_metrics(y[te], p)
        m["params"] = "~250 trees"
        results["RF"].append(m); record_fold(TAG, f"f{fold}", "RF", m)

        # ---------- neural comparators (binary head)
        for name, build, ncls in (("1D-CNN", build_cnn, 2),
                                  ("CNN-LSTM", build_cnn_lstm, 2)):
            mdl = build(n_classes=ncls)
            trmask, vamask = subject_group_val_split(g[tr], config.TRAIN["val_fraction"])
            mdl, hist = train_model(mdl, Xtr[trmask], y[tr][trmask],
                                    Xtr[vamask], y[tr][vamask],
                                    epochs=epochs, patience=patience, verbose=0)
            p = mdl.predict(Xte, verbose=0, batch_size=512)
            m = binary_metrics(y[te], p[:, 1] if p.ndim > 1 else p)
            m["params"] = mdl.count_params()
            results[name].append(m)
            record_fold(TAG, f"f{fold}", name, m)
            if verbose:
                print(f"[E1] fold {fold} {name}: F1 {m['f1']:.3f} "
                      f"({mdl.count_params()} params)", flush=True)

        # ---------- proposed (3-class head -> merged binary, plan section 6)
        mdl = build_model(n_classes=3)
        trmask, vamask = subject_group_val_split(g[tr], config.TRAIN["val_fraction"])
        mdl, hist = train_model(mdl, Xtr[trmask], y3[tr][trmask],
                                Xtr[vamask], y3[tr][vamask],
                                epochs=epochs, patience=patience, verbose=0)
        p = mdl.predict(Xte, verbose=0, batch_size=512)
        merged = p[:, 1] + p[:, 2]          # p(alert) + p(fall)
        m = binary_metrics(y[te], merged)
        m["params"] = mdl.count_params()
        results["Proposed"].append(m)
        record_fold(TAG, f"f{fold}", "Proposed", m, extra="3-class merged")
        if verbose:
            print(f"[E1] fold {fold} Proposed: F1 {m['f1']:.3f} "
                  f"({mdl.count_params()} params)", flush=True)

    out = {}
    for name, rs in results.items():
        out[name] = {k: float(np.mean([r[k] for r in rs])) if all(
            isinstance(r[k], float) or np.isfinite(r[k]) for r in rs) else np.nan
            for k in ["sens", "spec", "f1", "auc"]}
        out[name]["std_f1"] = float(np.std([r["f1"] for r in rs]))
        out[name]["params"] = rs[0]["params"]

    # ---- full 38-fold LOSO for the proposed model (headline result, plan 5.1)
    loso = None
    if do_loso:
        loso_f1, loso_rows = [], []
        for i, subj in enumerate(sorted(np.unique(g))):
            tr = np.where(g != subj)[0]
            te = np.where(g == subj)[0]
            mu, sd = compute_norm(X[tr], g[tr])
            Xtr, Xte = apply_norm(X[tr], mu, sd), apply_norm(X[te], mu, sd)
            mdl = build_model(n_classes=3)
            trmask, vamask = subject_group_val_split(g[tr], config.TRAIN["val_fraction"])
            mdl, _ = train_model(mdl, Xtr[trmask], y3[tr][trmask],
                                 Xtr[vamask], y3[tr][vamask],
                                 epochs=epochs, patience=patience, verbose=0)
            p = mdl.predict(Xte, verbose=0, batch_size=512)
            merged = p[:, 1] + p[:, 2]
            m = binary_metrics(y[te], merged)
            m["params"] = mdl.count_params()
            loso_rows.append(m)
            record_fold(TAG, f"loso-{subj}", "Proposed", m, extra="LOSO")
            if verbose and (i + 1) % 5 == 0:
                print(f"[E1] LOSO {i+1}/{len(np.unique(g))} ({subj}): F1 {m['f1']:.3f}",
                      flush=True)
        loso_f1 = [r["f1"] for r in loso_rows]
        loso = {"f1": float(np.mean(loso_f1)), "std": float(np.std(loso_f1)),
                "n_folds": len(loso_rows),
                "sens": float(np.mean([r["sens"] for r in loso_rows])),
                "spec": float(np.mean([r["spec"] for r in loso_rows])),
                "auc": float(np.mean([r["auc"] for r in loso_rows]))}

    json.dump({"grouped5": out, "loso": loso},
              open(os.path.join(config.OUT, "e1_summary.json"), "w"), indent=2)
    log_line(TAG, "done", f"loso={bool(loso)}")
    tm = t.done()
    print(json.dumps({"grouped5": {k: round(v["f1"], 4) for k, v in out.items()},
                      "loso": loso}, indent=2))
    return {"grouped5": out, "loso": loso, "seconds": tm}
