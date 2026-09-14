"""tables.py — assemble Tables I-IV (plan section 10) into markdown + CSV."""
import csv
import json
import os
import numpy as np

import config
from src.utils import macro_f1


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _avg_fold(results, stage, model):
    """Mean/SD across the recorded folds of one stage+model combo."""
    import pandas as pd
    df = pd.read_csv(config.RESULTS_CSV) if os.path.exists(config.RESULTS_CSV) else None
    if df is None or len(df) == 0:
        return {}
    sub = df[(df["stage"] == stage) & (df["model"] == model)]
    if "f1" not in sub or len(sub) == 0:
        return {}
    return {"f1": float(sub["f1"].mean()), "sd": float(sub["f1"].std(ddof=0)),
            "sens": float(sub["sensitivity"].mean()),
            "spec": float(sub["specificity"].mean()),
            "auc": float(sub["auc"].mean()),
            "params": sub["params"].iloc[0] if len(sub) else "?",
            "n": len(sub)}


def table1():
    import pandas as pd
    from src.utils import RESULTS_CSV
    e1 = _load(os.path.join(config.OUT, "e1_summary.json")) or {}
    g5 = e1.get("grouped5", {})
    df = pd.read_csv(RESULTS_CSV) if os.path.exists(RESULTS_CSV) else None
    rows = []
    for name, p in [("SMV", "5-fold grouped"), ("SVM", "5-fold grouped"),
                    ("RF", "5-fold grouped"), ("1D-CNN", "5-fold grouped"),
                    ("CNN-LSTM", "5-fold grouped"), ("Proposed", "38-fold LOSO*")]:
        entry = g5.get(name)
        if not entry:
            continue
        if name == "Proposed" and df is not None:
            # headline: full 38-fold LOSO, means over all folds from results.csv
            loso = df[(df["stage"] == "E1") & (df["model"] == "Proposed")
                      & (df["fold"].astype(str).str.startswith("loso"))]
            if len(loso):
                pv = loso["params"].iloc[0]
                ps = str(g5.get("Proposed", {}).get("params", "?")) \
                    if pd.isna(pv) else str(int(float(pv)))
                rows.append([name, p, ps,
                             f"{loso['sensitivity'].mean():.3f}",
                             f"{loso['specificity'].mean():.3f}",
                             f"{loso['f1'].mean():.3f} ± {loso['f1'].std(ddof=0):.3f}",
                             f"{loso['auc'].mean():.3f}"])
                continue
        rows.append([name, p, entry.get("params", "?"),
                     f"{entry['sens']:.3f}", f"{entry['spec']:.3f}",
                     f"{entry['f1']:.3f} ± {entry.get('std_f1', 0):.3f}",
                     f"{entry['auc']:.3f}"])
    return rows


def table2():
    e2 = _load(os.path.join(config.OUT, "e2_summary.json")) or []
    e1 = _load(os.path.join(config.OUT, "e1_summary.json")) or {}
    # within-dataset reference for the drop column: LOSO headline number
    ref = e1.get("loso", {}).get("f1", float("nan"))
    train_by_fold = {fid: " + ".join(tr) for fid, tr, _ in config.E2_FOLDS}
    rows = []
    for r in e2:
        ad = r.get("coral")
        if ad is None:
            ad = r.get("dann")
        na = r.get("none", float("nan"))
        iw = r.get("instnorm", float("nan"))
        best = max([v for v in (na, iw, ad) if v is not None], default=float("nan"))
        rows.append([r["fold"], train_by_fold.get(r["fold"], "?"),
                     r["test"], f"{na:.3f}", f"{iw:.3f}",
                     f"{ad:.3f}" if ad is not None else "—",
                     f"{(best - ref) * 100:+.1f} pp"])
    return rows


def table3():
    out = []
    for ds in ("KFall", "SisFall"):
        d = _load(os.path.join(config.OUT, f"e3_{ds}_summary.json"))
        if not d:
            continue
        s = d["summary"]
        for th in config.E3_TABLE_THRESHOLDS:
            k = str(th)
            if k not in s:
                continue
            v = s[k]
            out.append([ds, th, f"{v['mean_lead_ms']:.0f}", f"{v['std_lead_ms']:.0f}",
                        f"{v['detection_rate'] * 100:.1f}%",
                        f"{v['fa_per_hour']:.3f}"])
    return out


def table4():
    e4 = _load(os.path.join(config.OUT, "e4_summary.json"))
    if not e4:
        return []
    return [["Model size (KB)", f"{e4['fp32_size_kb']:.1f}", f"{e4['tflite_size_kb']:.1f}",
             "measure on device (Flash size) → < 60 KB target"],
            ["Peak RAM (KB)", "—", f"{e4['tensor_bytes_est'] / 1024:.1f} (tensor bytes)",
             "arena high-water from firmware → < 120 KB target"],
            ["Inference latency (ms)", "—", f"{e4['latency_ms_host_int8']:.2f} (host CPU)",
             "esp_timer_get_time() → < 50 ms target"],
            ["Macro-F1 (held-out set)", f"{e4['f1_fp32']:.4f}", f"{e4['f1_int8']:.4f}",
             "same model on device (≈ INT8 desktop)"],
            ["Battery life (hours)", "—", "—",
             "run-to-shutdown, 1000 mAh cell → measure & report"],
            ["Accuracy delta INT8 vs FP32", "—", f"{e4['f1_delta_pp']:.2f} pp",
             "< 2 pp target" + (" ✓ met" if e4["f1_delta_pp"] < 2 else " ✗ exceeded")]]


def write_tables():
    import pandas as pd
    os.makedirs(config.OUT, exist_ok=True)

    # Table I
    t1 = table1()
    md = ["# Table I — Within-dataset baseline (SisFall Enhanced)",
          "", "| Model | Protocol | Params | Sensitivity | Specificity | Macro-F1 | AUC |",
          "|---|---|---|---|---|---|---|"]
    for r in t1:
        md.append("| " + " | ".join(str(x) for x in r) + " |")
    md.append("")
    md.append("*Proposed model: full 38-fold leave-one-subject-out; comparators: "
              "5-fold subject-grouped CV (difference stated per plan §5.1). "
              "Positive class = window belongs to a fall event (alert ∪ impact; "
              "the same merge the 3-class head produces — a pre-impact firing is "
              "a detection, its timing is reported in Table III). Synthetic-mirror "
              "dataset in this simulation — replace with real SisFall Enhanced "
              "downloads and re-run for paper numbers.")
    md.append("")
    for r in t1:
        md.append(f"`{r[0]}`: F1 {r[5]} (AUC {r[6]})")
    table1_md = "\n".join(md)
    open(os.path.join(config.OUT, "table_I.md"), "w").write(table1_md)

    # Table II
    t2 = table2()
    md = ["# Table II — Cross-dataset generalisation (headline table)",
          "", "| Fold | Train | Test (unseen) | F1 (none) | F1 (inst.norm) | "
              "F1 (CORAL) | Δ vs Table I (pp) |",
          "|---|---|---|---|---|---|---|"]
    for r in t2:
        md.append("| " + " | ".join(str(x) for x in r) + " |")
    md.append("")
    md.append("Protocol: no target labels, no fine-tuning; CORAL/inst-norm use "
              "unlabelled windows from half of the target subjects (the other "
              "half is the evaluation set). Fold D's UMAFall is reported once — "
              "never used in training in any fold. Last column = best-adapter F1 "
              "minus the Table I LOSO reference (+ = no drop). On the "
              "synthetic-mirror domains the binary task transfers well — the "
              "honest conclusion is that the gap concentrates in the pre-impact "
              "task (Table III) and in label/measurement protocol differences; "
              "the same code on the real corpora gives the publishable numbers.")
    open(os.path.join(config.OUT, "table_II.md"), "w").write("\n".join(md))

    # Table III
    t3 = table3()
    md = ["# Table III — Pre-impact performance, both label sources",
          "", "| Label source | Threshold | Mean lead time (ms) | Std (ms) | "
              "Detection rate | False alarms / hour ADL |",
          "|---|---|---|---|---|---|"]
    for r in t3:
        md.append("| " + " | ".join(str(x) for x in r) + " |")
    md.append("")
    md.append("KFall is the primary source (purpose-built temporal labels); "
              "SisFall Enhanced corroborates. Lead time = impact − fire time, "
              "fire at first window right edge with p(alert)+p(fall) ≥ θ, 5 s cooldown.")
    open(os.path.join(config.OUT, "table_III.md"), "w").write("\n".join(md))

    # Table IV
    t4 = table4()
    md = ["# Table IV — On-device deployment",
          "", "| Metric | FP32 (desktop) | INT8 (desktop, TFLite interpreter) | "
              "INT8 (ESP32, measured on hardware) |",
          "|---|---|---|---|"]
    for r in t4:
        md.append("| " + " | ".join(str(x) for x in r) + " |")
    md.append("")
    md.append("Desktop numbers measured in this simulation on the host CPU; "
              "ESP32 column is filled from the firmware's serial output "
              "(see firmware/README.md and README §6-7: flash, mattress-drop test, "
              "record latency/arena/battery).")
    open(os.path.join(config.OUT, "table_IV.md"), "w").write("\n".join(md))

    # CSV copies
    for name, rows in (("table_I", t1), ("table_II", t2),
                       ("table_III", t3), ("table_IV", t4)):
        with open(os.path.join(config.OUT, f"{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerows(rows)
    print("[tables] wrote table_I..IV.md + .csv ->", config.OUT)
    return t1, t2, t3, t4
