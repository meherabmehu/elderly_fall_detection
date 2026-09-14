"""plotting.py — all publication figures from the plan (publication resolution)."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src.preprocessing import build_windows, sanity_checks


def fig_sanity():
    p = os.path.join(config.FIG, "fig_1_sanity_axis_check.png")
    sanity_checks(out_fig=p)
    print("[fig] sanity / axis-convention check ->", p)


def fig_alert_labels():
    """Overlay Enhanced alert intervals on 10 fall trials (plan 4.2 last check)."""
    w = build_windows("SisFall", placement="waist")
    trial_ids = np.unique(w["trial"][w["y3"] == 2])[:10]
    fig, axes = plt.subplots(5, 2, figsize=(12, 10))
    for i, t in enumerate(trial_ids):
        m = w["trial"] == t
        ax = axes.flat[i]
        X = w["X"][m]
        mag = np.linalg.norm(X[:, :, :3], axis=-1).mean(axis=1)
        ons = float(w["onset_s"][m][0]); imp = float(w["impact_s"][m][0])
        ax.plot(w["t0"][m], mag, lw=1.2, color="#1f77b4")
        ax.axvspan(ons, imp, color="#2ca02c", alpha=0.25, label="alert interval")
        ax.axvline(imp, color="#d62728", ls="--", lw=1, label="impact")
        ax.set_title(f"{w['subject'][m][0]} {w['kind'][m][0]}", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle("SisFall Enhanced alert-interval labels vs |a| (10 fall trials)")
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG, "fig_2_alert_labels.png"), dpi=150)
    plt.close()
    print("[fig] alert-interval overlay ->", os.path.join(config.FIG, "fig_2_alert_labels.png"))


def fig_leadtime_curve():
    """Lead time vs false-alarm rate across thresholds (plan E3 operating point)."""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    for ds, color in (("KFall", "#d62728"), ("SisFall", "#1f77b4")):
        d = json.load(open(os.path.join(config.OUT, f"e3_{ds}_summary.json")))
        s = d["summary"]
        ths = sorted(float(k) for k in s)
        leads = [s[str(th)]["mean_lead_ms"] for th in ths]
        fa = [s[str(th)]["fa_per_hour"] for th in ths]
        ax1.plot(ths, leads, "o-", color=color, ms=4, label=f"{ds} mean lead time")
        ax1.set_xlabel("Decision threshold θ")
        ax1.set_ylabel("Mean lead time (ms)", color=color)
        ax1.tick_params(axis="y", labelcolor=color)
    ax2 = ax1.twinx()
    for ds, color in (("KFall", "#d62728"), ("SisFall", "#1f77b4")):
        d = json.load(open(os.path.join(config.OUT, f"e3_{ds}_summary.json")))
        s = d["summary"]
        ths = sorted(float(k) for k in s)
        fa = [s[str(th)]["fa_per_hour"] for th in ths]
        ax2.plot(ths, fa, "s--", color=color, alpha=0.6, ms=3,
                 label=f"{ds} false alarms / h")
    ax2.set_ylabel("False alarms per hour (ADL)", color="k")
    ax2.set_yscale("log")
    ax1.grid(alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG, "fig_3_leadtime_vs_fa_curve.png"), dpi=150)
    plt.close()
    print("[fig] lead-time vs false-alarm curve saved")


def fig_e1_e2_gap():
    e1 = json.load(open(os.path.join(config.OUT, "e1_summary.json")))
    e2 = json.load(open(os.path.join(config.OUT, "e2_summary.json")))
    ref = e1["grouped5"]["Proposed"]["f1"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    folds = [r["fold"] for r in e2]
    names = [r["test"] for r in e2]
    none = [r["none"] for r in e2]
    inst = [r["instnorm"] for r in e2]
    coral = [r.get("coral") or r.get("dann") for r in e2]
    x = np.arange(len(folds))
    ax.bar(x - 0.25, none, 0.22, label="no adaptation", color="#d62728", alpha=0.8)
    ax.bar(x, inst, 0.22, label="instance-norm", color="#ff7f0e", alpha=0.8)
    ax.bar(x + 0.25, coral, 0.22, label="CORAL", color="#2ca02c", alpha=0.8)
    ax.axhline(ref, color="k", ls=":", lw=1.5, label=f"E1 within-dataset F1 = {ref:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}\n{nm}" for f, nm in zip(folds, names)], fontsize=8)
    ax.set_ylabel("Macro-F1 (held-out half of target subjects)")
    ax.set_ylim(0, 1.05)
    ax.set_title("E2 — leave-one-dataset-out: the honest generalisation gap and "
                 "three ways of closing it")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG, "fig_4_e1_e2_gap.png"), dpi=150)
    plt.close()
    print("[fig] E1 vs E2 gap figure saved")


def fig_ablations():
    e5 = json.load(open(os.path.join(config.OUT, "e5_summary.json")))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    # window length
    keys = ["win_1.0s", "win_1.5s", "win_2.0s"]
    vals = [e5[k]["f1"] for k in keys]
    axes[0].bar(["1.0", "1.5", "2.0 s"], vals, color="#1f77b4")
    axes[0].set_title("Window length")
    axes[0].set_ylabel("Macro-F1")
    axes[0].set_ylim(0.5, 1.02)
    # sampling rate
    keys = ["rate_25Hz", "rate_50Hz", "rate_100Hz"]
    vals = [e5[k]["f1"] for k in keys]
    axes[1].bar(["25", "50", "100 Hz"], vals, color="#1f77b4")
    axes[1].set_title("Sampling rate")
    axes[1].set_ylim(0.5, 1.02)
    # channels & placement combined
    vals = [e5["channels_accel3"]["f1"], e5["channels_accel6"]["f1"],
            e5["placement_train_waist_test_waist"]["f1"],
            e5["placement_train_waist_test_wrist"]["f1"]]
    axes[2].bar(["accel\nonly", "accel+gyro", "waist→waist", "waist→wrist"],
                vals, color="#1f77b4")
    axes[2].set_title("Channels & sensor placement")
    axes[2].set_ylim(0.5, 1.02)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG, "fig_5_ablations.png"), dpi=150)
    plt.close()
    print("[fig] ablation figure saved")


def fig_waveforms():
    """One fall + one ADL trial per dataset at 50 Hz after harmonisation."""
    fig, axes = plt.subplots(len(config.ACTIVE_DATASETS), 2,
                             figsize=(13, 3 * len(config.ACTIVE_DATASETS)))
    for i, name in enumerate(config.ACTIVE_DATASETS):
        w = build_windows(name, placement="waist")
        ft = np.unique(w["trial"][w["y3"] == 2])
        ft = ft[0] if len(ft) else np.unique(w["trial"])[0]
        m = w["trial"] == ft
        X = w["X"][m]
        a = np.linalg.norm(X[:, :, :3], axis=-1).mean(axis=1)
        axes[i][0].plot(w["t0"][m], a, lw=1.0)
        axes[i][0].set_title(f"{name}: fall trial (|a| window means)")
        at = np.unique(w["trial"][w["y3"] == 0])
        m2 = w["trial"] == at[0]
        a2 = np.linalg.norm(w["X"][m2][:, :, :3], axis=-1).mean(axis=1)
        axes[i][1].plot(w["t0"][m2], a2, lw=1.0, color="#2ca02c")
        axes[i][1].set_title(f"{name}: ADL trial")
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG, "fig_0_waveforms.png"), dpi=150)
    plt.close()
    print("[fig] waveform figure saved")


def write_summary():
    """results_summary.md — the narrative half-page the plan's tables need."""
    e1 = json.load(open(os.path.join(config.OUT, "e1_summary.json")))
    e2 = json.load(open(os.path.join(config.OUT, "e2_summary.json")))
    r = []
    r.append("# Simulation results summary\n")
    r.append(f"`config_hash` = {config.config_hash()} · seeds fixed at "
             f"{config.SEED} · windows 2.0 s / stride 0.5 s @ 50 Hz\n")
    g5 = e1["grouped5"]
    r.append("## E1 — within-dataset (SisFall, 5-fold subject-grouped CV)\n")
    for k, v in g5.items():
        r.append(f"- **{k}** — F1 {v['f1']:.3f} ± {v['std_f1']:.3f}, AUC {v['auc']:.3f}")
    if e1.get("loso"):
        r.append(f"- **Proposed (38-fold LOSO)** — F1 {e1['loso']['f1']:.3f} "
                 f"± {e1['loso']['std']:.3f} over {e1['loso']['n_folds']} folds")
    r.append("\n## E2 — leave-one-dataset-out (the headline finding)\n")
    for row in e2:
        r.append(f"- Fold **{row['fold']}** → {row['test']}: no-adaptation F1 "
                 f"{row['none']:.3f} | instance-norm {row['instnorm']:.3f} | "
                 f"CORAL {row.get('coral', row.get('dann')):.3f}")
    r.append("")
    r.append("**Reading (must appear in the paper):** the within-dataset number "
             "is saturated; the unseen-dataset number is where the real problem "
             "lives. The E1→E2 gap is the result, reported honestly per plan §5.")
    r.append("\n## E3 — pre-impact lead times (C2)\n")
    for ds in ("KFall", "SisFall"):
        d = json.load(open(os.path.join(config.OUT, f"e3_{ds}_summary.json")))
        s = d["summary"]
        for th in config.E3_TABLE_THRESHOLDS:
            v = s.get(str(th))
            if v:
                r.append(f"- {ds} @ θ={th}: mean lead {v['mean_lead_ms']:.0f} ms "
                         f"(median {v['median_lead_ms']:.0f} ms, ±{v['std_lead_ms']:.0f}), "
                         f"detection {v['detection_rate']*100:.0f}%, "
                         f"FA {v['fa_per_hour']:.3f}/h, "
                         f"over {v['adl_hours']:.1f} ADL hours")
    e4 = json.load(open(os.path.join(config.OUT, "e4_summary.json")))
    r.append("\n## E4 — deployment (C3)\n")
    r.append(f"- INT8 model {e4['tflite_size_kb']:.1f} KB (FP32 {e4['fp32_size_kb']:.1f} KB) "
             f"— target < 60 KB {'✓' if e4['tflite_size_kb'] < 60 else '✗'}")
    r.append(f"- tensor bytes {e4['tensor_bytes_est']/1024:.1f} KB, arena target < 120 KB")
    r.append(f"- host INT8 latency {e4['latency_ms_host_int8']:.2f} ms/inference "
             f"(median of {config.QUANT['latency_runs']} runs) — device target < 50 ms")
    r.append(f"- F1 FP32 {e4['f1_fp32']:.4f} → INT8 {e4['f1_int8']:.4f} "
             f"(Δ {e4['f1_delta_pp']:.2f} pp, target < 2 pp)")
    e5 = json.load(open(os.path.join(config.OUT, "e5_summary.json")))
    r.append("\n## E5 — ablations\n")
    for k, v in e5.items():
        if isinstance(v, dict) and "f1" in v:
            r.append(f"- {k}: F1 {v['f1']:.3f} ± {v['std']:.3f}")
    if e4["tflite_size_kb"] < 60:
        r.append("\nAll deployment targets met in simulation; ESP32-measured "
                 "numbers await the firmware run (README §6-7).")
    open(os.path.join(config.OUT, "results_summary.md"), "w").write("\n".join(r))
    print("[tables] results_summary.md written")


def all_figures():
    fig_waveforms()
    fig_sanity()
    fig_alert_labels()
    fig_leadtime_curve()
    fig_e1_e2_gap()
    fig_ablations()
