"""baselines.py — E1 comparators from the plan: SMV threshold, SVM, Random Forest.

All operate on the same 2.0 s windows, same subject-grouped splits.
"""
import numpy as np

from src.utils import binary_metrics


def handcrafted_features(X):
    """~48 features: per-channel time-domain stats + SMV statistics."""
    n = len(X)
    feats = []
    for c in range(X.shape[2]):
        ch = X[:, :, c]
        feats.append(np.concatenate([
            ch.mean(axis=1, keepdims=True),
            ch.std(axis=1, keepdims=True),
            ch.min(axis=1, keepdims=True),
            ch.max(axis=1, keepdims=True),
            np.percentile(ch, 95, axis=1, keepdims=True),
            np.abs(ch).mean(axis=1, keepdims=True),
        ], axis=1))
    a = np.linalg.norm(X[:, :, :3], axis=-1)
    g = np.linalg.norm(X[:, :, 3:], axis=-1)
    smv = np.concatenate([a.max(axis=1, keepdims=True), a.mean(axis=1, keepdims=True),
                          a.std(axis=1, keepdims=True),
                          g.max(axis=1, keepdims=True), g.mean(axis=1, keepdims=True)],
                         axis=1)
    return np.hstack(feats + [smv]).astype(np.float32)


def smv_predict_threshold(X, thr):
    """Signal magnitude vector rule: window is a fall if peak |a| > thr."""
    a = np.linalg.norm(X[:, :, :3], axis=-1)
    return (a.max(axis=1) > thr).astype(int)


def fit_smv(X_tr, y_tr, grid=None):
    """Threshold chosen on the TRAINING fold only (subject-honest)."""
    grid = grid if grid is not None else np.arange(1.2, 3.6, 0.05)
    best_thr, best_f1 = grid[0], -1
    a_tr = np.linalg.norm(X_tr[:, :, :3], axis=-1).max(axis=1)
    for thr in grid:
        pred = (a_tr > thr).astype(int)
        m = binary_metrics(y_tr, pred)
        if m["f1"] > best_f1:
            best_f1, best_thr = m["f1"], thr
    return float(best_thr)


def smv_predict(X_tr, y_tr, X_te):
    thr = fit_smv(X_tr, y_tr)
    pred = smv_predict_threshold(X_te, thr)
    return pred, thr


def fit_svm(X_tr, y_tr):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(StandardScaler(),
                        SVC(C=10.0, gamma="scale", class_weight="balanced",
                            probability=True, random_state=0))
    clf.fit(handcrafted_features(X_tr), y_tr)
    return clf


def svm_predict(clf, X_te):
    p = clf.predict_proba(handcrafted_features(X_te))
    return p[:, 1] if p.shape[1] == 2 else p[:, 0]


def fit_rf(X_tr, y_tr):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=250, min_samples_leaf=3,
                                 class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=0)
    clf.fit(handcrafted_features(X_tr), y_tr)
    return clf


def rf_predict(clf, X_te):
    p = clf.predict_proba(handcrafted_features(X_te))
    return p[:, 1] if p.shape[1] == 2 else p[:, 0]
