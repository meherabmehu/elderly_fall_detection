"""e2.py — E2 leave-one-dataset-out generalisation (plan section 5, C1).

Folds:  A train [SisFall+KFall]  -> test FallAllD
        B train [SisFall+FallAllD] -> test KFall
        C train [KFall+FallAllD] -> test SisFall
        D train [all three]      -> test UMAFall (held out, reported once)

Adapter protocol (documented, subject-honest):
   - 'none'     : frozen global training statistics only
   - 'instnorm' : per-window normalisation instead of global stats
   - 'coral'    : unlabelled target windows from HALF the target subjects
                  (never the evaluation subjects), second-order alignment
   - 'dann'     : adversarial head on the same unlabelled target split
No target labels, no fine-tuning of the classifier on labelled target data.
"""
import json
import os
import numpy as np

import config
from src.preprocessing import build_windows, compute_norm, apply_norm, event_labels
from src.utils import (set_seed, record_fold, log_line, binary_metrics,
                       subject_group_val_split, Timer)
from src.model import build_model, train_model
from src.adaptation import window_normalise, train_coral, train_dann

TAG = "E2"


def _load(name):
    w = build_windows(name, placement="waist")
    return w


def _fold_data(train_names, test_name):
    w = {n: _load(n) for n in train_names + [test_name]}
    X_tr = np.concatenate([w[n]["X"] for n in train_names])
    y_tr = np.concatenate([event_labels(w[n]) for n in train_names])
    g_tr = np.concatenate([np.array([f"{n}::{s}" for s in w[n]["subject"]])
                           for n in train_names])
    X_te, y_te, g_te = (w[test_name]["X"], event_labels(w[test_name]),
                        w[test_name]["subject"])
    return X_tr, y_tr, g_tr, X_te, y_te, g_te, w[test_name]


def run(adapters=None, epochs=None, patience=None, verbose=1):
    set_seed()
    t = Timer(TAG)
    adapters = adapters or ["none", "instnorm", "coral"]
    rows = []
    for fid, train_names, test_name in config.E2_FOLDS:
        X_tr, y_tr, g_tr, X_te, y_te, g_te, wte = _fold_data(train_names, test_name)
        print(f"[E2] fold {fid}: train {train_names} ({len(X_tr)} windows) "
              f"-> test {test_name} ({len(X_te)} windows)")

        # subject-honest split of target: 50% subjects for unlabelled adaptation,
        # 50% for evaluation (never touched, even unlabelled)
        u = np.array(sorted(set(g_te)))
        rng = np.random.default_rng(config.SEED)
        rng.shuffle(u)
        half = u[:len(u) // 2]
        adapt_mask = np.isin(g_te, half)
        ev_mask = ~adapt_mask
        X_te_adapt = X_te[adapt_mask]
        X_te_ev, y_te_ev = X_te[ev_mask], y_te[ev_mask]

        # frozen stats from training subjects only
        mu, sd = compute_norm(X_tr, g_tr)
        Xtr_g = apply_norm(X_tr, mu, sd)
        Xte_ev_g = apply_norm(X_te_ev, mu, sd)
        Xte_adapt_g = apply_norm(X_te_adapt, mu, sd)

        # HARD GUARD: never train/predict on an empty target evaluation set.
        # This prevents the Keras batch_outputs crash seen when a parser returns 0 windows.
        if len(X_te_ev) == 0 or len(y_te_ev) == 0:
            print(f"[E2] SKIP fold {fid}: {test_name} has 0 evaluation windows. Fix its parser before rerunning.", flush=True)
            continue

        fold_row = {"fold": fid, "test": test_name}
        for adapter in adapters:
            if verbose:
                print(f"[E2]   adapter = {adapter}")
            # subject-wise val inside training data
            trmask, vamask = subject_group_val_split(g_tr, config.TRAIN["val_fraction"])

            if adapter == "none":
                mdl = build_model(n_classes=2)
                mdl, _ = train_model(mdl, Xtr_g[trmask], y_tr[trmask],
                                     Xtr_g[vamask], y_tr[vamask],
                                     epochs=epochs, patience=patience, verbose=0)
                p = mdl.predict(Xte_ev_g, verbose=0, batch_size=512)[:, 1]
            elif adapter == "instnorm":
                Xtr_n = window_normalise(Xtr_g[trmask])
                Xva_n = window_normalise(Xtr_g[vamask])
                mdl = build_model(n_classes=2)
                mdl, _ = train_model(mdl, Xtr_n, y_tr[trmask], Xva_n,
                                     y_tr[vamask], epochs=epochs,
                                     patience=patience, verbose=0)
                p = mdl.predict(window_normalise(Xte_ev_g), verbose=0,
                                batch_size=512)[:, 1]
            elif adapter == "coral":
                mdl, _ = train_coral(Xtr_g[trmask], y_tr[trmask],
                                     Xte_adapt_g, Xtr_g[vamask], y_tr[vamask],
                                     epochs=epochs, patience=patience, verbose=0)
                p = mdl.predict(Xte_ev_g, verbose=0, batch_size=512)[:, 1]
            elif adapter == "dann":
                mdl, _ = train_dann(Xtr_g[trmask], y_tr[trmask],
                                    Xte_adapt_g, Xtr_g[vamask], y_tr[vamask],
                                    epochs=epochs, patience=patience, verbose=0)
                p = mdl.predict(Xte_ev_g, verbose=0, batch_size=512)[0][:, 1]
            else:
                raise ValueError(adapter)

            m = binary_metrics(y_te_ev, p)
            m["params"] = mdl.count_params()
            record_fold(TAG, f"{fid}-{adapter}", f"LODO-{test_name}", m)
            fold_row[adapter] = round(m["f1"], 4)
            if verbose:
                print(f"  {fid}/{adapter}: F1 {m['f1']:.4f} sens {m['sens']:.3f} "
                      f"spec {m['spec']:.3f}", flush=True)
        rows.append(fold_row)

    # per-fold on the evaluation subject half
    json.dump(rows, open(os.path.join(config.OUT, "e2_summary.json"), "w"), indent=2)
    log_line(TAG, "done")
    t.done()
    return rows
