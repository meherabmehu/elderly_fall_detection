"""model.py — the proposed compact separable-CNN (plan section 6) + train helper.

Architecture (frozen, from the plan):
  Input 100 x 6
    -> instance normalisation (per-window, per-channel)
    -> Conv1D(24, k=7, s=2) + BN + ReLU           50 x 24
    -> SeparableConv1D(48, k=5) + BN + ReLU       50 x 48
    -> MaxPool1D(2)                               25 x 48
    -> SeparableConv1D(64, k=3) + BN + ReLU       25 x 64
    -> GlobalAveragePooling1D                     64
    -> Dense(32) + ReLU + Dropout(0.3)
    -> Dense(N) + Softmax
"""
import numpy as np
import tensorflow as tf

import config


class InstanceNorm(tf.keras.layers.Layer):
    """Per-window, per-channel normalisation (the model's first layer)."""

    def __init__(self, eps=1e-5, **kw):
        super().__init__(**kw)
        self.eps = eps

    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1], keepdims=True)
        return (x - mean) / tf.sqrt(var + self.eps)

    def get_config(self):
        c = super().get_config()
        c["eps"] = self.eps
        return c


def build_model(input_len=None, channels=None, n_classes=3,
                dropout=None, seed=config.SEED, name="proposed",
                norm_mode="window"):
    """Build the frozen architecture. input_len is samples (100 @ 50 Hz / 2.0 s).

    norm_mode:
      "window" -> per-window InstanceNorm as first layer (plan §6 architecture;
                  note: this layer breaks clean INT8 conversion — see README,
                  "why the deployed model uses norm_mode='none'")
      "none"   -> no first-layer normalisation (inputs are already normalised by
                  the frozen global statistics in preprocessing/firmware)
    """
    input_len = input_len or int(round(config.WIN_SEC * config.WORK_RATE))
    channels = channels or config.CHANNELS
    dropout = config.MODEL["dropout"] if dropout is None else dropout
    m = config.MODEL
    tf.random.set_seed(seed)
    inp = tf.keras.Input(shape=(input_len, channels), name="input")
    x = InstanceNorm()(inp) if norm_mode == "window" else inp
    x = tf.keras.layers.Conv1D(m["conv1"], m["conv1_k"], strides=m["conv1_s"],
                               padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.SeparableConv1D(m["sep1"], m["sep1_k"], padding="same",
                                        use_bias=False, depthwise_initializer="he_normal",
                                        pointwise_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.SeparableConv1D(m["sep2"], m["sep2_k"], padding="same",
                                        use_bias=False, depthwise_initializer="he_normal",
                                        pointwise_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(m["dense"], activation="relu",
                              kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax", name="probs")(x)
    model = tf.keras.Model(inp, out, name=name)
    return model


def build_cnn_lstm(input_len=None, channels=None, n_classes=2):
    """E1 comparator CNN-LSTM (similar budget, non-separable)."""
    input_len = input_len or int(round(config.WIN_SEC * config.WORK_RATE))
    channels = channels or config.CHANNELS
    inp = tf.keras.Input(shape=(input_len, channels))
    x = tf.keras.layers.Conv1D(24, 7, strides=2, padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.LSTM(32, dropout=0.2)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, x, name="cnn_lstm")


def build_cnn(input_len=None, channels=None, n_classes=2):
    """E1 comparator plain 1D-CNN (non-separable)."""
    input_len = input_len or int(round(config.WIN_SEC * config.WORK_RATE))
    channels = channels or config.CHANNELS
    inp = tf.keras.Input(shape=(input_len, channels))
    x = tf.keras.layers.Conv1D(32, 7, strides=2, padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, x, name="cnn")


def tf_frozen_func(model):
    """Raw-TF forward pass of the trained model (BatchNorm folded into the
    convolutions, dropout removed). Contains NO variables, so the TFLite INT8
    converter never emits ReadVariable nodes — the cleanest conversion path."""
    import tensorflow as tf
    bns = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    convs = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv1D)]
    seps = [l for l in model.layers if isinstance(l, tf.keras.layers.SeparableConv1D)]
    denses = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]

    def fold(conv, bn):
        g, b, m, v = bn.get_weights()
        scale = (g / np.sqrt(v + bn.epsilon)).astype(np.float32)
        off = (b - m * scale).astype(np.float32)
        return scale, off

    s1, o1 = fold(convs[0], bns[0])
    k1 = (convs[0].get_weights()[0] * s1.reshape(1, 1, -1)).astype(np.float32)
    b1 = o1
    s2, o2 = fold(seps[0], bns[1])
    dw1, pw1 = seps[0].get_weights()[:2]
    dw1 = dw1[:, None].astype(np.float32)      # (5,24,1) -> (5,1,24,1) for conv2d
    pw1 = (pw1 * s2.reshape(1, 1, -1)).astype(np.float32)
    b2 = o2
    s3, o3 = fold(seps[1], bns[2])
    dw2, pw2 = seps[1].get_weights()[:2]
    dw2 = dw2[:, None].astype(np.float32)
    pw2 = (pw2 * s3.reshape(1, 1, -1)).astype(np.float32)
    b3 = o3
    w4, b4 = [w.astype(np.float32) for w in denses[0].get_weights()]
    w5, b5 = [w.astype(np.float32) for w in denses[1].get_weights()]

    has_instnorm = any(isinstance(l, InstanceNorm) for l in model.layers)

    @tf.function
    def serve(x):
        if has_instnorm:
            mean, var = tf.nn.moments(x, axes=[1], keepdims=True)
            y = (x - mean) / tf.sqrt(var + 1e-5)
        else:
            y = x
        y = tf.nn.conv1d(y, k1, stride=2, padding="SAME") + b1
        y = tf.nn.relu(y)
        y = tf.nn.depthwise_conv2d(tf.expand_dims(y, -2), dw1,
                                   strides=[1, 1, 1, 1], padding="SAME")
        y = tf.squeeze(y, -2)
        y = tf.nn.conv2d(tf.expand_dims(y, -2), pw1[None],
                         strides=[1, 1, 1, 1], padding="SAME")
        y = tf.squeeze(y, -2) + b2
        y = tf.nn.relu(y)
        y = tf.nn.max_pool1d(y, ksize=2, strides=2, padding="VALID")
        y = tf.nn.depthwise_conv2d(tf.expand_dims(y, -2), dw2,
                                   strides=[1, 1, 1, 1], padding="SAME")
        y = tf.squeeze(y, -2)
        y = tf.nn.conv2d(tf.expand_dims(y, -2), pw2[None],
                         strides=[1, 1, 1, 1], padding="SAME")
        y = tf.squeeze(y, -2) + b3
        y = tf.nn.relu(y)
        y = tf.reduce_mean(y, axis=1)
        y = tf.nn.relu(tf.matmul(y, w4) + b4)
        return {"probs": tf.nn.softmax(tf.matmul(y, w5) + b5)}

    return serve


def freeze_bn(model):
    """Return an inference-only clone with all BatchNorm layers folded into the
    preceding convolution kernels/biases. Needed for clean TFLite conversion
    (the resource-variable BatchNorm ops break the INT8 interpreter path)."""
    import tensorflow as tf

    def _fold(conv, bn):
        g, b, m, v = [bn.get_weights()[i] for i in range(4)]
        eps = bn.epsilon
        scale = g / np.sqrt(v + eps)
        off = b - m * scale
        return scale, off

    src = model.layers                    # frozen plan-6 architecture order
    bns = [l for l in src if isinstance(l, tf.keras.layers.BatchNormalization)]
    convs = [l for l in src if isinstance(l, tf.keras.layers.Conv1D)]
    seps = [l for l in src if isinstance(l, tf.keras.layers.SeparableConv1D)]
    denses = [l for l in src if isinstance(l, tf.keras.layers.Dense)]
    inp = tf.keras.Input(shape=model.input.shape[1:])
    x = InstanceNorm()(inp)

    def make_conv1d(layer, kscale, koff, x):
        w = layer.get_weights()[0]
        w = (w * kscale.reshape((1, 1, -1))).astype(np.float32)
        l = tf.keras.layers.Conv1D(layer.filters, layer.kernel_size[0],
                                   strides=layer.strides[0], padding=layer.padding,
                                   use_bias=True)
        l.build(x.shape)
        l.set_weights([w, koff.astype(np.float32)])
        return l

    def make_sep(layer, kscale, koff, x):
        dw, pw = layer.get_weights()[:2]
        pw = (pw * kscale.reshape((1, 1, -1))).astype(np.float32)
        l = tf.keras.layers.SeparableConv1D(layer.filters, layer.kernel_size[0],
                                            padding=layer.padding, use_bias=True)
        l.build(x.shape)
        l.set_weights([dw, pw, koff.astype(np.float32)])
        return l

    # conv1 + bn, sep1 + bn, sep2 + bn, dense, dense (plan-6 order)
    s, o = _fold(convs[0], bns[0])
    x = tf.keras.layers.ReLU()(make_conv1d(convs[0], s, o, x)(x))
    s, o = _fold(seps[0], bns[1])
    x = tf.keras.layers.ReLU()(make_sep(seps[0], s, o, x)(x))
    x = tf.keras.layers.MaxPool1D(2)(x)
    s, o = _fold(seps[1], bns[2])
    x = tf.keras.layers.ReLU()(make_sep(seps[1], s, o, x)(x))
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    d1 = tf.keras.layers.Dense(denses[0].units, activation="relu")
    d1.build(x.shape)
    d1.set_weights([w.astype(np.float32) for w in denses[0].get_weights()])
    x = d1(x)
    d2 = tf.keras.layers.Dense(denses[1].units, activation="softmax")
    d2.build(x.shape)
    d2.set_weights([w.astype(np.float32) for w in denses[1].get_weights()])
    out = d2(x)
    return tf.keras.Model(inp, out)


def train_model(model, X_tr, y_tr, X_va, y_va, epochs=None, patience=None,
                batch_size=None, class_weight=True, verbose=1, restoration=None):
    """Train with the plan's configuration:
       Adam 1e-3, cosine decay, early stopping on validation macro-F1,
       class weighting (never oversampling). Returns (model, history).
    """
    from src.utils import MacroF1EarlyStopping
    epochs = epochs or config.TRAIN["epochs"]
    patience = patience or config.TRAIN["patience"]
    batch_size = batch_size or config.TRAIN["batch_size"]

    sw = None
    if class_weight:
        from src.utils import class_sample_weights
        sw = class_sample_weights(y_tr)
    steps = max(1, int(np.ceil(len(X_tr) / batch_size)))
    lr = tf.keras.optimizers.schedules.CosineDecay(
        config.TRAIN["lr"], decay_steps=epochs * steps)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    es = MacroF1EarlyStopping(patience=patience)
    hist = []
    best_weights = None
    for ep in range(epochs):
        model.fit(X_tr, y_tr, batch_size=batch_size, epochs=1, verbose=0,
                  sample_weight=sw, shuffle=True)
        keep = es.on_epoch_end(model, X_va, y_va, hist)
        if verbose:
            print(f"    epoch {ep+1:3d}/{epochs} val macro-F1 {hist[-1]:.4f}"
                  f" (best {es.best:.4f})", flush=True)
        if not keep:
            break
    if es.best_weights is not None:
        model.set_weights(es.best_weights)
    return model, hist
