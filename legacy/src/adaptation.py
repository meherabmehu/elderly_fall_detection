"""adaptation.py — E2 gap-closing steps (plan 5.1, ascending cost):

  1. instnorm : per-window instance normalisation instead of frozen global stats
                (near-zero MCU cost — the model's first layer does the same thing)
  2. coral    : deep CORAL — aligns second-order feature statistics between source
                and unlabelled target windows (no target labels required)
  3. dann     : adversarial domain adaptation (gradient-reversal), only if 1-2
                are insufficient and time permits (per the plan)
"""
import numpy as np
import tensorflow as tf

import config
from src.model import build_model, InstanceNorm
from src.utils import macro_f1


# ---------------------------------------------------------------- 1. instnorm
def window_normalise(X):
    """Per-window, per-channel stats (mean/std over time, per window)."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-6
    return ((X - mean) / std).astype(np.float32)


# ---------------------------------------------------------------- 2. CORAL
def coral_loss(f_src, f_tgt):
    """Deep CORAL loss on feature vectors (d = feature dim)."""
    d = tf.cast(tf.shape(f_src)[1], tf.float32)
    cs = _cov(f_src)
    ct = _cov(f_tgt)
    return tf.reduce_sum((cs - ct) ** 2) / (4.0 * d * d)


def _cov(f):
    f = f - tf.reduce_mean(f, axis=0, keepdims=True)
    n = tf.cast(tf.shape(f)[0], tf.float32)
    return tf.matmul(tf.transpose(f), f) / (n - 1.0 + 1e-6)


def _make_reversal(lam_var):
    """Custom op: forward identity, backward -lambda * grad (DANN).
    `lam_var` is a tf.Variable so the factor can be scheduled per epoch."""
    @tf.custom_gradient
    def rev(x):
        def grad(dy):
            return -lam_var * dy, None
        return tf.identity(x), grad
    return rev


def train_coral(X_src, y_src, X_tgt, X_va, y_va,
                epochs=None, patience=None, verbose=1, lamb=config.CORAL_LAMBDA):
    """GradientTape loop: source labelled + target unlabelled, shared encoder."""
    from src.utils import class_sample_weights
    epochs = epochs or config.TRAIN["epochs"]
    patience = patience or config.TRAIN["patience"]
    batch = config.TRAIN["batch_size"]

    model = build_model(n_classes=2, name="coral")
    # feature extractor = everything up to (not including) the final dense
    feat_model = tf.keras.Model(model.input,
                                model.get_layer("probs").input)
    opt = tf.keras.optimizers.Adam(config.TRAIN["lr"])
    sw = class_sample_weights(y_src)
    best_f1, best_w, wait = -1, None, 0
    hist = []
    n = len(X_src)
    for ep in range(epochs):
        idx = np.random.permutation(n)
        xb = X_src[idx[:min(batch, n)]]
        yb = y_src[idx[:min(batch, n)]].astype(np.int32)
        wb = sw[idx[:min(batch, n)]].astype(np.float32)
        tb = X_tgt[np.random.randint(len(X_tgt), size=min(batch, len(X_tgt)))]
        with tf.GradientTape() as tape:
            fs = feat_model(xb, training=True)
            ft = feat_model(tb, training=True)
            logits = model(xb, training=True)
            ce = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(yb, logits) * wb)
            loss = ce + lamb * coral_loss(fs, ft)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        pred = model.predict(X_va, verbose=0, batch_size=512)
        f1 = macro_f1(y_va, pred)
        hist.append(f1)
        if f1 > best_f1 + 1e-4:
            best_f1, wait = f1, 0
            best_w = [w for w in model.get_weights()]
        else:
            wait += 1
        if verbose:
            print(f"    coral epoch {ep+1:3d}/{epochs} val F1 {f1:.4f} (best {best_f1:.4f})",
                  flush=True)
        if wait >= patience:
            break
    if best_w is not None:
        model.set_weights(best_w)
    return model, hist


# ---------------------------------------------------------------- 3. DANN
def train_dann(X_src, y_src, X_tgt, X_va, y_va, epochs=None, patience=None, verbose=1):
    """DANN with a gradient-reversal domain head (target labels never used)."""
    from src.utils import class_sample_weights
    epochs = epochs or config.TRAIN["epochs"]
    patience = patience or config.TRAIN["patience"]
    batch = config.TRAIN["batch_size"]
    lam_max = config.DANN_MAX_LAMBDA
    n_dom = 2

    lam_var = tf.Variable(1.0, trainable=False, dtype=tf.float32)
    rev = _make_reversal(lam_var)

    inp = tf.keras.Input(shape=(int(config.WIN_SEC * config.WORK_RATE), config.CHANNELS))
    m = config.MODEL
    x = InstanceNorm()(inp)
    x = tf.keras.layers.Conv1D(m["conv1"], m["conv1_k"], strides=m["conv1_s"],
                               padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.SeparableConv1D(m["sep1"], m["sep1_k"], padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.SeparableConv1D(m["sep2"], m["sep2_k"], padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x); x = tf.keras.layers.ReLU()(x)
    feat = tf.keras.layers.GlobalAveragePooling1D()(x)
    class_out = tf.keras.layers.Dense(2, activation="softmax", name="class_head")(feat)
    dom_out = tf.keras.layers.Dense(n_dom, activation="softmax", name="dom_head")(rev(feat))
    dann = tf.keras.Model(inp, [class_out, dom_out], name="dann")

    opt = tf.keras.optimizers.Adam(config.TRAIN["lr"])
    sw = class_sample_weights(y_src)
    best_f1, best_w, wait = -1, None, 0
    hist = []
    for ep in range(epochs):
        p = ep / max(1, epochs - 1)
        lam_var.assign(lam_max * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0))  # 0 -> 1
        idx = np.random.permutation(len(X_src))
        for bi in range(0, len(X_src), batch):
            b = idx[bi:bi + batch]
            xb, yb, wb = X_src[b], y_src[b].astype(np.int32), sw[b].astype(np.float32)
            tb = X_tgt[np.random.randint(len(X_tgt), size=len(b))]
            dom_y = np.concatenate([np.zeros(len(b)), np.ones(len(tb))]).astype(np.int64)
            with tf.GradientTape() as tape:
                logits, dom_logits = dann(tf.concat([xb, tb], 0), training=True)
                ce = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(yb, logits[:len(b)]) * wb)
                dom_ce = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(dom_y, dom_logits))
                loss = ce + dom_ce          # reversal handles the -lambda scaling
            grads = tape.gradient(loss, dann.trainable_variables)
            opt.apply_gradients(zip(grads, dann.trainable_variables))
        pred = dann.predict(X_va, verbose=0, batch_size=512)[0]
        f1 = macro_f1(y_va, pred)
        hist.append(f1)
        if f1 > best_f1 + 1e-4:
            best_f1, wait = f1, 0
            best_w = [w for w in dann.get_weights()]
        else:
            wait += 1
        if verbose:
            print(f"    dann epoch {ep+1:3d}/{epochs} val F1 {f1:.4f} (best {best_f1:.4f})",
                  flush=True)
        if wait >= patience:
            break
    if best_w is not None:
        dann.set_weights(best_w)
    return dann, hist
