#!/usr/bin/env python3
"""Desktop replay verification: replay KFall S06T20R01 through the INT8 models.

Replicates, bit-for-bit in float, the deployed firmware pipeline:
  ring buffer (100x6 @ 50 Hz) -> per-window instance normalisation (eps 1e-6)
  -> INT8 quantise (model's input scale / zero point) -> TFLite invoke
  -> p_alarm = p_alert + p_fall -> threshold + k-consecutive confirmation.

Ground truth for the replayed trial (KFall label file SA06_label.xlsx):
  onset frame 130, impact frame 208 at native 100 Hz
  -> at 50 Hz: onset sample 65, impact sample 104 (resample_index rounding).

Outputs a JSON provenance record. Meant to be run against committed artifacts only;
it fabricates nothing -- every probability printed comes from the bundled model.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def load_replay(path: Path) -> np.ndarray:
    """Parse replay_data.h -> (N, 6) float32 in g / deg/s, already at 50 Hz."""
    txt = path.read_text()
    m = re.search(r"const float replay_data\[\]?\[6\] = \{(.*?)\};", txt, re.S)
    if not m:
        m = re.search(r"REPLAY_SAMPLES\]\[6\] = \{(.*?)\};", txt, re.S)
    rows = re.findall(r"\{([^}]*)\}", m.group(1))
    data = np.array([[float(v.replace("f", "")) for v in r.split(",")] for r in rows], np.float32)
    return data


def instance_normalise(w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = w.mean(axis=0, keepdims=True)
    s = w.std(axis=0, keepdims=True) + eps
    return ((w - m) / s).astype(np.float32)


class TfliteRunner:
    def __init__(self, model_path: Path):
        import tensorflow as tf
        self.interp = tf.lite.Interpreter(model_path=str(model_path))
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.in_scale, self.in_zp = self.inp["quantization"]
        self.out_scale, self.out_zp = self.out["quantization"]

    def predict(self, window: np.ndarray) -> np.ndarray:
        q = np.round(window / self.in_scale + self.in_zp).clip(-128, 127).astype(np.int8)
        self.interp.set_tensor(self.inp["index"], q[None, ...])
        self.interp.invoke()
        o = self.interp.get_tensor(self.out["index"])[0].astype(np.float32)
        return (o - self.out_zp) * self.out_scale


def run_replay(signal, runner, window_len=100, stride=25, norm="instance",
               norm_mean=None, norm_std=None, threshold=0.95, k=2):
    """Slide the window exactly as the firmware does; return per-window records."""
    n = len(signal)
    records = []
    consecutive = 0
    for start in range(0, n - window_len + 1, stride):
        w = signal[start:start + window_len]
        if norm == "instance":
            x = instance_normalise(w)
        else:  # global frozen constants (legacy firmware)
            x = ((w - np.asarray(norm_mean, np.float32)) / np.asarray(norm_std, np.float32)).astype(np.float32)
        p = runner.predict(x)
        p_alarm = float(p[1] + p[2])
        consecutive = consecutive + 1 if p_alarm >= threshold else 0
        fired = consecutive >= k
        if fired:
            consecutive = 0
        records.append({
            "window_start": start,
            "right_edge": start + window_len - 1,
            "p_bkg": round(float(p[0]), 4),
            "p_alert": round(float(p[1]), 4),
            "p_fall": round(float(p[2]), 4),
            "p_alarm": round(p_alarm, 4),
            "alarm": bool(fired),
        })
    return records


def summarise(records, impact_idx, threshold, k, hz=50):
    fired = [r for r in records if r["alarm"]]
    first = fired[0] if fired else None
    lead_ms = None
    if first is not None:
        lead_ms = round((impact_idx - first["right_edge"]) * 1000.0 / hz, 1)
    return {
        "threshold": threshold, "k": k,
        "n_windows": len(records), "n_alarms": len(fired),
        "first_alarm_right_edge": None if first is None else int(first["right_edge"]),
        "impact_sample_50hz": int(impact_idx),
        "lead_time_ms": lead_ms,
        "max_p_fall": max(r["p_fall"] for r in records),
        "max_p_alarm": max(r["p_alarm"] for r in records),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--replay", required=True, type=Path)
    ap.add_argument("--norm", choices=["instance", "global"], default="instance")
    ap.add_argument("--norm-mean", type=float, nargs=6)
    ap.add_argument("--norm-std", type=float, nargs=6)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--impact", type=int, default=104, help="impact sample index at 50 Hz")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    signal = load_replay(args.replay)
    runner = TfliteRunner(args.model)
    recs = run_replay(signal, runner, norm=args.norm,
                      norm_mean=args.norm_mean, norm_std=args.norm_std,
                      threshold=args.threshold, k=args.k)
    summary = summarise(recs, args.impact, args.threshold, args.k)
    summary.update({
        "model": str(args.model), "model_bytes": args.model.stat().st_size,
        "input_scale": float(runner.in_scale), "input_zero_point": int(runner.in_zp),
        "output_scale": float(runner.out_scale), "output_zero_point": int(runner.out_zp),
        "norm": args.norm,
    })
    out = {"summary": summary, "records": recs}
    text = json.dumps(out, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(json.dumps(summary, indent=2))
    for r in recs:
        print(f"  w{r['window_start']:3d} edge={r['right_edge']:3d} "
              f"p_bkg={r['p_bkg']:.4f} p_alert={r['p_alert']:.4f} p_fall={r['p_fall']:.4f}"
              f"{'  ALARM' if r['alarm'] else ''}")


if __name__ == "__main__":
    main()
