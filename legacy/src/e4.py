"""e4.py — E4 deployment: INT8 quantisation, verification, measurement (C3).

Step 1  convert to full-integer TFLite (representative dataset = TRAINING windows,
        plan rule: never from test)
Step 2  verify in the Python TFLite interpreter against FP32 on the held-out set
Step 3  export C array (model_data.h) + frozen normalisation constants + quant
        params (model_config.h) for the firmware, all from the SAME training run
Step 4  measure .tflite size, host inference latency (1,000 runs), tensor bytes
ESP32-measured numbers (arena high-water, battery) come from flashing the
provided firmware — the table marks them as measured-on-hardware.
"""
import json
import os
import time
import numpy as np
import tensorflow as tf

import config
from src.preprocessing import (build_windows, compute_norm, apply_norm,
                               save_norm, load_norm)
from src.utils import (set_seed, record_fold, log_line, binary_metrics,
                       subject_group_val_split, Timer)
from src.model import build_model, train_model, tf_frozen_func

TAG = "E4"


def _quant_input(x, scale, zp, dtype=np.int8):
    q = np.round(x / scale) + zp
    return np.clip(q, -128, 127).astype(dtype)


def run(epochs=None, patience=None, verbose=1, pretrained=False):
    """pretrained=True: skip training entirely — load models/proposed_fp32.keras
    and output/frozen_normalization.json (e.g. a model trained on Kaggle), then
    quantise, verify and re-export firmware files."""
    set_seed()
    t = Timer(TAG)
    w = build_windows("SisFall", placement="waist")
    X, y3, g = w["X"], w["y3"], w["subject"]
    subs = np.array(sorted(set(g)))
    rng = np.random.default_rng(config.SEED)
    rng.shuffle(subs)
    n_te = max(1, int(round(len(subs) * 0.2)))
    te_s = set(subs[:n_te].tolist())
    tr = np.array([s not in te_s for s in g])
    te = ~tr

    fp32_path = os.path.join(config.MODELS, "proposed_fp32.keras")
    if pretrained:
        assert os.path.exists(fp32_path), \
            "no pretrained model found — run without --pretrained, or drop your " \
            "Kaggle-trained files into models/ and output/"
        model = tf.keras.models.load_model(fp32_path)
        mu, sd = load_norm(os.path.join(config.OUT, "frozen_normalization.json"))
        print(f"[E4] pretrained model loaded ({model.count_params()} params), "
              f"frozen norm loaded — no training performed")
    else:
        mu, sd = compute_norm(X[tr], g[tr])
        save_norm(os.path.join(config.OUT, "frozen_normalization.json"),
                  mu, sd, g[tr], f"SisFall training subjects ({len(set(g[tr]))})")

        # ------------------------------------------------------------ FP32 model
        # Deployed model uses norm_mode="none": the frozen global statistics are
        # applied by preprocessing (and by the firmware, model_config.h), because
        # the plan's per-window InstanceNorm first layer is an int8 killer — its
        # moments-based ops collapse under full-integer quantisation (measured:
        # INT8 F1 drops 0.98 -> 0.55). In-domain accuracy is the same either way.
        trmask, vamask = subject_group_val_split(g[tr], config.TRAIN["val_fraction"])
        model = build_model(n_classes=3, norm_mode="none")
        model, _ = train_model(model, apply_norm(X[tr], mu, sd)[trmask],
                               y3[tr][trmask],
                               apply_norm(X[tr], mu, sd)[vamask], y3[tr][vamask],
                               epochs=epochs, patience=patience, verbose=0)
        model.save(fp32_path)

    Xte = apply_norm(X[te], mu, sd)
    Xtr = apply_norm(X[tr], mu, sd)      # needed for the representative dataset
    p32 = model.predict(Xte, verbose=0, batch_size=512)
    merge32 = p32[:, 1] + p32[:, 2]
    m32 = binary_metrics(y3[te] > 0, merge32)
    m32["params"] = model.count_params()
    record_fold(TAG, "host-fp32", "Proposed-3class", m32)

    # ---------------------------------------------------------------- INT8 quant
    # Raw-TF forward pass (BatchNorm folded, no variables) converted via a
    # concrete function — the INT8 interpreter path then runs cleanly.
    serve = tf_frozen_func(model)
    dmax = np.abs(serve(Xte[:64])["probs"].numpy() -
                  model.predict(Xte[:64], verbose=0)).max()
    print(f"[E4] raw-TF equivalence: max diff {dmax:.2e}")
    assert dmax < 1e-4, "tf_frozen_func diverges from the trained model!"

    def rep_gen():
        idx = np.random.RandomState(config.SEED).choice(len(Xtr),
                                                        size=config.QUANT["rep_windows"],
                                                        replace=False)
        for i in idx:
            yield [Xtr[i][None].astype(np.float32)]  # list = one array per input

    _il = int(config.WIN_SEC * config.WORK_RATE)     # 100 samples
    concrete = serve.get_concrete_function(
        tf.TensorSpec([1, _il, config.CHANNELS], tf.float32))
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    tflite_path = os.path.join(config.MODELS, "model.tflite")
    open(tflite_path, "wb").write(tflite_model)
    size_kb = os.path.getsize(tflite_path) / 1024.0

    # ---------------------------------------------------------------- verify
    interp = tf.lite.Interpreter(model_path=tflite_path, num_threads=4)
    interp.allocate_tensors()
    i_d = interp.get_input_details()[0]
    o_d = interp.get_output_details()[0]
    scale, zp = i_d["quantization"]
    print(f"[E4] input quant: scale={scale:.6f} zero_point={zp} "
          f"in_shape={i_d['shape']} out_shape={o_d['shape']}")

    def predict_int8(X):
        out = np.zeros((len(X), 3), dtype=np.float32)
        for i, x in enumerate(X):
            interp.set_tensor(i_d["index"], _quant_input(x[None], scale, zp))
            interp.invoke()
            q = interp.get_tensor(o_d["index"])[0]
            out[i] = (q.astype(np.float32) - o_d["quantization"][1]) * o_d["quantization"][0]
        return out

    p8 = predict_int8(Xte)
    merge8 = p8[:, 1] + p8[:, 2]
    m8 = binary_metrics(y3[te] > 0, merge8)
    m8["params"] = model.count_params()

    # ---------------------------------------------------------------- latency
    times = []
    x0 = np.zeros(i_d["shape"].tolist(), dtype=np.int8)
    for _ in range(20):
        interp.set_tensor(i_d["index"], x0); interp.invoke()
    for _ in range(config.QUANT["latency_runs"]):
        t0 = time.perf_counter()
        interp.set_tensor(i_d["index"], x0)
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000.0)
    lat_ms = float(np.median(times))

    tensor_bytes = int(sum(int(np.prod(d["shape"]) * np.dtype(d["dtype"]).itemsize)
                           for d in interp.get_tensor_details()))
    fp32_size_kb = os.path.getsize(fp32_path) / 1024.0

    record_fold(TAG, "host-int8", "Proposed-3class", m8,
                extra=f"size_kb={size_kb:.1f} lat_ms={lat_ms:.2f}")

    # ---------------------------------------------------------------- firmware exports
    import tools.make_model_h as mh
    mh.export(tflite_path,
              norm_path=os.path.join(config.OUT, "frozen_normalization.json"),
              out_dir=config.FW_DIR, scale=float(scale), zp=int(zp),
              input_len=int(i_d["shape"][1]), channels=int(i_d["shape"][2]))

    out = {"fp32_size_kb": fp32_size_kb, "tflite_size_kb": size_kb,
           "tensor_bytes_est": tensor_bytes,
           "latency_ms_host_int8": lat_ms,
           "f1_fp32": m32["f1"], "f1_int8": m8["f1"],
           "f1_delta_pp": abs(m32["f1"] - m8["f1"]) * 100.0,
           "input_scale": scale, "input_zero_point": zp,
           "test_windows": int(len(Xte)), "params": int(model.count_params())}
    json.dump(out, open(os.path.join(config.OUT, "e4_summary.json"), "w"), indent=2)
    log_line(TAG, "done", str({k: round(v, 3) if isinstance(v, float) else v
                               for k, v in out.items() if k != "tensor_bytes_est"}))
    t.done()
    return out
