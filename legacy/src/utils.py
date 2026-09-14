"""utils.py — seeds, logging, metrics, fold recording, macro-F1 early stopping."""
import numpy as np
import os
import time
import csv

import config

RESULTS_CSV = os.path.join(config.OUT, "results.csv")
LOG_CSV = os.path.join(config.OUT, "experiments_log.csv")


def set_seed(seed=config.SEED):
    import random
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def log_line(stage, note, extra=""):
    import datetime
    row = [datetime.datetime.now().isoformat(timespec="seconds"), stage, note,
           config.config_hash(), extra]
    new = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "stage", "note", "config_hash", "extra"])
        w.writerow(row)
    print(f"[log] {stage}: {note}", flush=True)


def record_fold(stage, fold, model, metrics, extra=""):
    """Plan rule: write results after EVERY fold, not at the end."""
    new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["stage", "fold", "model", "f1", "sensitivity",
                        "specificity", "auc", "params", "config_hash", "extra"])
        w.writerow([stage, fold, model,
                    f"{metrics.get('f1', float('nan')):.4f}",
                    f"{metrics.get('sens', float('nan')):.4f}",
                    f"{metrics.get('spec', float('nan')):.4f}",
                    f"{metrics.get('auc', float('nan')):.4f}",
                    metrics.get("params", ""),
                    config.config_hash(), extra])


def binary_metrics(y_true, y_prob, threshold=0.5):
    from sklearn.metrics import (roc_auc_score, precision_recall_fscore_support,
                                 confusion_matrix)
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = precision_recall_fscore_support(y, pred, average="binary",
                                         zero_division=0)[2]
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")
    return {"sens": sens, "spec": spec, "f1": float(f1), "auc": float(auc)}


def macro_f1(y_true, y_prob_multi, threshold=None):
    from sklearn.metrics import f1_score
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob_multi)
    if p.ndim == 2:
        pred = p.argmax(axis=1)
    else:
        pred = (p >= (threshold or 0.5)).astype(int)
    return float(f1_score(y, pred, average="macro", zero_division=0))


def class_sample_weights(y, mu=1.0):
    """Plan rule: class weighting, never oversampling (window overlap leakage)."""
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=int(y.max()) + 1).astype(float)
    counts[counts == 0] = 1.0
    w = (len(y) / (len(counts) * counts)) ** mu
    return w[y]


def subject_group_val_split(subjects, val_fraction):
    """Split subjects (not windows) into train/val for early stopping."""
    u = np.array(sorted(set(subjects)))
    rng = np.random.default_rng(config.SEED)
    rng.shuffle(u)
    n_val = max(1, int(round(len(u) * val_fraction)))
    val_s = set(u[:n_val].tolist())
    tr_mask = np.array([s not in val_s for s in subjects])
    va_mask = ~tr_mask
    return tr_mask, va_mask


class MacroF1EarlyStopping:
    """Keras-version-agnostic early stopping on validation macro-F1.

    Drop-in replacement for metrics + EarlyStopping (works with Keras 2 and 3):
    after every epoch it predicts on the validation split (numpy), computes
    macro-F1, restores the best weights and stops when patience expires.
    """
    def __init__(self, patience=6, min_delta=1e-4, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best = -1
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, model, X_val, y_val, history):
        import tensorflow as tf
        p = model.predict(X_val, verbose=0, batch_size=512)
        if p.ndim == 2 and p.shape[1] > 1:
            score = macro_f1(y_val, p)
        else:
            score = macro_f1(y_val, p, threshold=0.5)
        history.append(score)
        if score > self.best + self.min_delta:
            self.best = score
            self.wait = 0
            if self.restore_best:
                self.best_weights = [w.numpy() if hasattr(w, "numpy") else w
                                     for w in model.get_weights()]
        else:
            self.wait += 1
        return self.wait < self.patience


class Timer:
    def __init__(self, label):
        self.label = label
        self.t0 = time.time()

    def done(self, extra=""):
        dt = time.time() - self.t0
        print(f"[time] {self.label}: {dt:.1f} s {extra}", flush=True)
        return dt
