"""synthetic.py — generates the four dataset formats with physically plausible IMU signals.

Why this exists: the real corpora (SisFall Enhanced, KFall, FallAllD, UMAFall) are
1-10 GB and not part of this workspace. These generators write EXACTLY the same
layouts/columns the real parsers accept, so the rest of the pipeline (parsers →
preprocessing → experiments) is the same code that will run on the real data;
drop the real files into data/raw/<name> and re-run the pipeline.

Canonical frame used internally:  z up (gravity +1 g on az when standing),
x forward, y right. Angles in radians, gyro in deg/s, accel in g.
"""

import json
import os
import sys

import numpy as np
from scipy import signal as sig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

rng = np.random.default_rng(config.SEED)

CANON2DATASET = {k: np.array(v["rotation"]["matrix"], dtype=float).reshape(3, 3)
                 for k, v in config.DATASETS.items()}

# ---------------------------------------------------------------- domain profiles
# Each REAL dataset was recorded with different hardware and protocol (different
# low-pass filtering of the MEMS, noise, gain, population, fall strength).
# These profiles reproduce that so the LODO experiment has a genuine
# cross-domain gap to expose (the whole point of C1). SisFall is the reference
# (profile = identity) so E1/E4 numbers are untouched by this table.
DEFAULT_ADL_MENU = ["walk", "jog", "standstill", "sitdown", "standup",
                    "liedown", "rise", "stairs", "bend"]
DOMAIN_PROFILES = {
    "SisFall":  {"noise": 1.00, "gait": 1.00, "strength": 1.00, "gain": 1.00,
                 "lp": None, "adl_menu": DEFAULT_ADL_MENU},
    "KFall":    {"noise": 1.45, "gait": 0.90, "strength": 0.80, "gain": 0.95,
                 "lp": 45.0,          # sensor DLPF heavier, older population
                 "adl_menu": ["walk", "standstill", "sitdown", "standup",
                              "liedown", "rise", "stairs", "bend", "jog"]},
    "FallAllD": {"noise": 0.80, "gait": 1.06, "strength": 1.18, "gain": 1.06,
                 "lp": None,          # raw accelerometer, hard laboratory falls
                 "adl_menu": ["walk", "jog", "standstill", "sitdown", "standup",
                              "liedown", "rise", "quicksit", "kneel", "bend"]},
    "UMAFall":  {"noise": 2.30, "gait": 0.94, "strength": 0.85, "gain": 0.90,
                 "lp": 12.0,          # smartphone stream: heavy smoothing
                 "adl_menu": ["walk", "standstill", "sitdown", "standup",
                              "liedown", "rise", "quicksit", "kneel", "jog", "bend"]},
}


# ============================================================ core synthesis
def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _lowpass(x, fs, cutoff, order=4, axis=0):
    sos = sig.butter(order, cutoff, "low", fs=fs, output="sos")
    return sig.sosfiltfilt(sos, x, axis=axis)


def _rotate(v, R):
    return np.einsum("ij,tj->ti", R, v)


def _apply_impact(acc, gyro, fs, idx, A, e_imp, G, e_gyr, tau=0.018, f_osc=18.0):
    """Add a damped-oscillation impact spike at sample idx (plan: 7-16 g peaks)."""
    n = len(acc)
    t = np.arange(n) / fs
    tt = t - t[idx]
    m = tt >= 0
    env = np.exp(-tt / tau)
    osc = np.cos(2 * np.pi * f_osc * tt)
    amp = A * env * osc * m
    gamp = G * np.exp(-tt / (tau * 1.6)) * m
    acc += np.outer(amp, e_imp)
    gyro += np.outer(gamp, e_gyr)


def _make_stance(kind, fs, dur, sp):
    """Return accel (n,3) g, gyro (n,3) dps for a pre-fall activity beat."""
    n = int(round(dur * fs))
    t = np.arange(n) / fs
    a = np.zeros((n, 3))
    g = np.zeros((n, 3))
    f = sp["gait_f"]
    ph = rng.uniform(0, 2 * np.pi, 6)
    if kind in ("walk", "jog"):
        amp = 1.0 if kind == "walk" else 2.6
        a[:, 2] = 1.0 + amp * 0.09 * np.sin(2 * np.pi * f * t + ph[0]) \
            + amp * 0.03 * np.sin(4 * np.pi * f * t + ph[1])
        a[:, 0] = amp * 0.09 * np.sin(np.pi * f * t + ph[2])
        a[:, 1] = amp * 0.05 * np.sin(np.pi * f * t + ph[3])
        g[:, 0] = amp * 9 * np.sin(np.pi * f * t + ph[4])
        g[:, 1] = amp * 6 * np.sin(np.pi * f * t + ph[5])
        g[:, 2] = amp * 5 * np.sin(np.pi * f * t * 0.5 + ph[0])
    elif kind == "stairs":
        a[:, 2] = 1.0 + 0.22 * np.sin(2 * np.pi * f * 0.8 * t + ph[0]) + 0.04
        a[:, 0] = 0.11 * np.sin(np.pi * f * 0.8 * t + ph[1])
        g[:, 0] = 13 * np.sin(np.pi * f * 0.8 * t + ph[2])
        g[:, 1] = 9 * np.sin(np.pi * f * 0.8 * t + ph[3])
        a[:, 2] += 0.06 * np.sin(2 * np.pi * 1.3 * t + ph[4])
    elif kind == "bend":
        th = 0.95 * _smoothstep(np.clip(t / (dur * 0.5), 0, 1))
        th *= _smoothstep(np.clip((dur - t) / (dur * 0.5), 0, 1))
        a[:, 0] = 9.81 / 9.81 * np.sin(th) * 1.0
        a[:, 2] = np.cos(th)
        g[:, 1] = np.gradient(th, fs) * 180 / np.pi
    elif kind == "sitdown" or kind == "standup":
        # stand -> seated (az ~0.9, small lean) or reverse
        th = _smoothstep(np.clip(t / dur, 0, 1)) * (0.12 if kind == "sitdown" else 0.12)
        if kind == "standup":
            th = (1 - _smoothstep(np.clip(t / dur, 0, 1))) * 0.12
        a[:, 0] = np.sin(th * 0.4)
        a[:, 2] = np.cos(th * 0.4) * 0.92 + 0.08
        g[:, 1] = np.gradient(th, fs) * 180 / np.pi * 4
        a[:, 2] += 0.5 * _smoothstep(np.clip((t - dur * 0.55) / 0.25, 0, 1)) \
            * (1 - _smoothstep(np.clip((t - dur * 0.8) / 0.2, 0, 1)))
    elif kind == "liedown" or kind == "rise":
        th = _smoothstep(np.clip(t / dur, 0, 1)) * np.pi / 2  # to lying, ax=1, az=0
        if kind == "rise":
            th = (1 - _smoothstep(np.clip(t / dur, 0, 1))) * np.pi / 2
        a[:, 0] = np.sin(th)
        a[:, 2] = np.cos(th)
        g[:, 0] = (np.gradient(th, fs) * 180 / np.pi) * 0.8
        g[:, 1] = (np.gradient(th, fs) * 180 / np.pi) * 0.4
        if kind == "rise":
            a[:, 2] += 0.9 * _smoothstep(np.clip((t - dur * 0.7) / (dur * 0.25), 0, 1)) \
                * (1 - _smoothstep(np.clip((t - dur * 0.95) / (dur * 0.15), 0, 1)))
    elif kind == "seated":
        a[:, 2] = 0.92 + 0.02 * np.sin(2 * np.pi * 0.25 * t + ph[0])
        a[:, 1] = 0.03 * np.sin(2 * np.pi * 0.13 * t + ph[1])
        g[:, 2] = 2 * np.sin(2 * np.pi * 0.2 * t + ph[2])
    elif kind == "standstill":
        a[:, 2] = 1.0 + 0.008 * np.sin(2 * np.pi * 0.3 * t + ph[0])
        a[:, 1] = 0.02 * np.sin(2 * np.pi * 0.11 * t + ph[1])
        g[:, 2] = 1.5 * np.sin(2 * np.pi * 0.1 * t + ph[2])
        a[:, 2] += 0.01 * np.sin(2 * np.pi * 1.9 * t + ph[3])   # breathing
    elif kind == "quicksit":
        # abrupt sit-down: fast rotation + a small but real 2-4 g deceleration
        # "impact" — the classic fall-detection false-positive, dataset-specific
        a[:, 2] = 1.0
        th = _smoothstep(np.clip(t / (dur * 0.28), 0, 1)) * 0.45
        a[:, 0] += np.sin(th)
        a[:, 2] += np.cos(th) - 1.0
        g[:, 1] += np.gradient(th, fs) * 180 / np.pi
        i_b = int(0.30 * dur * fs)
        bump = 3.2 * np.array([np.exp(-((t[i_b:] - t[i_b]) / 0.03))])[0] * \
            np.cos(2 * np.pi * 16 * (t[i_b:] - t[i_b]))
        a[i_b:, 2] += bump
        a[:, 2] += 0.05 * np.sin(2 * np.pi * 1.6 * t)           # post-sit wobble
    elif kind == "kneel":
        # kneeling: forward flexion with a small knee-contact bump (2-4 g),
        # then hold the low posture — a realistic fall-detection false positive
        th = 0.9 * _smoothstep(np.clip(t / (dur * 0.4), 0, 1))
        a[:, 0] = np.sin(th); a[:, 2] = np.cos(th)
        g[:, 1] += np.gradient(th, fs) * 180 / np.pi * 0.9
        i_b = int(0.42 * dur * fs)
        bump = 2.6 * np.exp(-((t[i_b:] - t[i_b]) / 0.035)) * \
            np.cos(2 * np.pi * 14 * (t[i_b:] - t[i_b]))
        a[i_b:, 0] += bump
    return a, g


FALL_BY_TYPE = {
    "fall_forward": dict(e_imp=[0.55, 0.10, -0.83], e_gyr=[0.35, 0.90, 0.15], axis=1),
    "fall_back":    dict(e_imp=[-0.50, 0.15, -0.85], e_gyr=[0.30, -0.90, 0.20], axis=1),
    "fall_side":    dict(e_imp=[0.15, 0.60, -0.78], e_gyr=[0.90, 0.25, 0.20], axis=0),
    "fall_stumble": dict(e_imp=[0.45, 0.25, -0.86], e_gyr=[0.40, 0.75, 0.40], axis=1),
    "fall_walk":    dict(e_imp=[0.60, 0.05, -0.80], e_gyr=[0.30, 0.92, 0.18], axis=1),
    "fall_chair":   dict(e_imp=[0.42, 0.30, -0.86], e_gyr=[0.55, -0.75, 0.25], axis=0),
}
ADL_BY_TYPE = {  # (pre-activity, activity-duration fraction)
    "walk": ("walk", 1.0), "jog": ("jog", 1.0), "standstill": ("standstill", 1.0),
    "sitdown": ("standstill", 0.35), "standup": ("seated", 0.35),
    "liedown": ("standstill", 0.30), "rise": ("liedown", 0.30),
    "stairs": ("stairs", 1.0), "bend": ("walk", 0.35),
}


def make_fall(kind, fs, sp, stance_kind="walk"):
    """Full fall trial: stance -> alert (label 1) -> descent -> impact -> lying.

    Returns accel (n,3) g, gyro (n,3) dps, labels (n,) 0/1/2, onset idx, impact idx.
    """
    t_stand = rng.uniform(1.5, 3.2)
    t_alert = rng.uniform(0.70, 1.40)
    t_desc = rng.uniform(0.30, 0.70)
    t_post = rng.uniform(2.0, 4.0)
    dur = t_stand + t_alert + t_desc + t_post + 0.35
    n = int(round(dur * fs))
    t = np.arange(n) / fs

    a, g = _make_stance(stance_kind, fs, t_stand, sp)
    pad = lambda x, m: np.vstack([x, np.zeros((m - len(x), 3))])
    a = pad(a, n); g = pad(g, n)

    spec = FALL_BY_TYPE[kind]
    theta1 = np.deg2rad(rng.uniform(16, 28))
    theta2 = np.deg2rad(rng.uniform(82, 96))

    t_on = t_stand
    t_im = t_on + t_alert + t_desc
    i_on, i_im = int(round(t_on * fs)), int(round(t_im * fs))
    labels = np.zeros(n, dtype=np.int8)
    labels[i_on:i_im] = 1
    labels[i_im:] = 2

    # --- alert: fast lean to theta1, then an unmistakable stumble/trip signature
    #     (loss-of-balance: double dip + lateral sway + tremor + gyro bursts)
    seg = slice(i_on, i_im)
    ts = t[seg] - t_on
    th = theta1 * _smoothstep(np.clip(ts / (t_alert * 0.3), 0, 1))
    a[seg, 0] += np.sin(th); a[seg, 2] += np.cos(th) - 1.0
    g[seg, spec["axis"]] += np.gradient(th, fs) * 180 / np.pi
    # stumble: two quick centre-of-mass dips + sway + tremor
    d1 = 0.75 * np.exp(-((ts - t_alert * 0.40) / 0.10) ** 2)
    d2 = 0.55 * np.exp(-((ts - t_alert * 0.62) / 0.08) ** 2)
    a[seg, 2] -= (d1 + d2)
    a[seg, 1] += 0.22 * np.sin(2 * np.pi * 4.5 * ts) * np.exp(-((ts - t_alert * 0.5) / 0.25) ** 2)
    tremor = 0.09 * np.sin(2 * np.pi * 10.0 * ts + rng.uniform(0, 6.28))
    a[seg, :] += tremor[:, None] * 0.35
    g[seg, 0] += 95 * np.exp(-((ts - t_alert * 0.5) / 0.12) ** 2) * np.sign(rng.normal())
    g[seg, 2] += 70 * np.exp(-((ts - t_alert * 0.45) / 0.14) ** 2) * np.sign(rng.normal())

    # --- descent: accelerate rotation, slight free-fall
    seg = slice(i_im - int(t_desc * fs), i_im)
    ts = t[seg] - (t_im - t_desc)
    th = theta1 + (theta2 - theta1) * _smoothstep(np.clip(ts / t_desc, 0, 1)) ** 1.15
    a[seg, 0] += np.sin(th) - np.sin(theta1)
    a[seg, 2] += np.cos(th) - np.cos(theta1)
    g[seg, spec["axis"]] += np.gradient(th, fs) * 180 / np.pi
    freefall = 0.55 * np.exp(-((ts - t_desc * 0.85) / 0.18) ** 2)
    a[seg, 2] -= freefall

    # --- post-impact: lying + decaying tremor (first, so the spike rides on top)
    seg = slice(i_im, n)
    ts = t[seg] - t_im
    lying_a = np.array([0.95, 0.05, 0.10]) if spec["axis"] == 1 else np.array([0.10, 0.95, 0.10])
    a[seg] = np.outer(np.ones(len(ts)), lying_a)
    tremor = 0.16 * np.exp(-ts / 0.9) * np.sin(2 * np.pi * rng.uniform(7, 13) * ts)
    a[seg, 0] += tremor
    a[seg, 2] += 0.5 * tremor
    g[seg] *= np.exp(-ts / 0.5)[:, None]

    # --- impact: damped 3.5-11 g spike + gyro burst (realistic waist impacts;
    #     peak |a| ends up ~0.9-1.1 x A, so it overlaps the 2-5 g ADL bumps)
    A = rng.uniform(3.5, 11.0) * sp["strength"]
    G = rng.uniform(250, 750)
    e_imp = np.array(spec["e_imp"]); e_gyr = np.array(spec["e_gyr"])
    _apply_impact(a, g, fs, i_im, A, e_imp, G, e_gyr,
                  tau=rng.uniform(0.012, 0.030), f_osc=rng.uniform(14, 24))

    # --- sensor noise & bias
    a += sp["bias_a"] + rng.normal(0, sp["noise_a"], a.shape)
    g += sp["bias_g"] + rng.normal(0, sp["noise_g"], g.shape)
    return a, g, labels, i_on, i_im


def make_adl(kind, fs, sp):
    """ADL trial → (accel, gyro, labels=0, onset=None, impact=None)."""
    dur = rng.uniform(*config.DATASETS["SisFall"]["adl_dur"])
    if kind in ("walk", "jog", "stairs", "bend", "standstill", "seated",
                "quicksit", "kneel"):
        # single-phase ADL: the template IS the whole trial
        a, g = _make_stance(kind, fs, dur, sp)
    else:                            # composite: transition + steady state
        base, frac = ADL_BY_TYPE[kind]
        a1, g1 = _make_stance(base, fs, dur * frac, sp)
        rest_dur = dur * (1 - frac)
        steady = {"sitdown": "seated", "standup": "standstill",
                  "liedown": "liedown", "rise": "standstill"}[kind]
        a2, g2 = _make_stance(steady, fs, rest_dur, sp)
        a = np.vstack([a1, a2]); g = np.vstack([g1, g2])
    a += sp["bias_a"] + rng.normal(0, sp["noise_a"], a.shape)
    g += sp["bias_g"] + rng.normal(0, sp["noise_g"], g.shape)
    return a, g, np.zeros(len(a), dtype=np.int8), None, None


def wrist_view(a, g, fs, sp, kind_is_walk):
    """Second placement: attenuated, lagged, with arm swing on the walk-like phases."""
    lag = int(round(0.05 * fs))
    a2 = np.vstack([np.zeros((lag, 3)), a[:-lag]]) if lag else a.copy()
    g2 = np.vstack([np.zeros((lag, 3)), g[:-lag]]) if lag else g.copy()
    a2 = a2 * rng.uniform(0.65, 0.8)
    g2 = g2 * rng.uniform(0.75, 0.9)
    n = len(a2)
    t = np.arange(n) / fs
    if kind_is_walk:
        f = sp["gait_f"]
        ph = rng.uniform(0, 2 * np.pi, 3)
        a2[:, 0] += 0.45 * np.sin(2 * np.pi * f * t + ph[0])
        a2[:, 2] += 0.12 * np.sin(4 * np.pi * f * t + ph[1])
        g2[:, 1] += 90 * np.sin(2 * np.pi * f * t + ph[2])
    a2 += rng.normal(0, sp["noise_a"] * 1.5, a2.shape)
    g2 += rng.normal(0, sp["noise_g"] * 1.5, g2.shape)
    return a2, g2


# ============================================================ emitters
def _subject_params(older, rng):
    return dict(
        gait_f=rng.uniform(1.25, 1.65) if older else rng.uniform(1.5, 2.05),
        noise_a=rng.uniform(0.008, 0.022),
        noise_g=rng.uniform(0.6, 2.4),
        bias_a=rng.normal(0, 0.028, 3),
        bias_g=rng.normal(0, 2.0, 3),
        strength=0.8 if older else rng.uniform(0.9, 1.25),
    )


def _trial_set(name, rng):
    """Deterministic per-dataset trial list: (subject_id, age, kind, is_fall)."""
    meta = config.DATASETS[name]
    if getattr(config, "SMALL_MODE", False):
        ny, no, nf, na = (config.SMALL["subjects_young"], config.SMALL["subjects_older"],
                          config.SMALL["falls_per_subject"], config.SMALL["adl_per_subject"])
    else:
        ny, no, nf, na = (meta["subjects_young"], meta["subjects_older"],
                          meta["falls_per_subject"], meta["adl_per_subject"])
    subs = []
    for i in range(ny):
        subs.append((f"A{i+1:02d}", "young"))
    for i in range(no):
        subs.append((f"B{i+1:02d}", "older"))
    trials = []
    for sid, age in subs:
        for i in range(nf):
            trials.append((sid, age, f"F{i+1:02d}", True))
        for i in range(na):
            trials.append((sid, age, f"D{i+1:02d}", False))
    return trials


def _write_csv(path, header, rows):
    import pandas as pd
    pd.DataFrame(rows, columns=header).to_csv(path, index=False, float_format="%.5f")


def generate_dataset(name, force=False):
    global rng
    # deterministic per-dataset stream: regenerating one dataset never perturbs
    # the others (seed independent of generation order)
    rng = np.random.default_rng(1000 + sum(ord(c) for c in name) * 7)
    meta = config.DATASETS[name]
    out = os.path.join(config.SYN, name)
    if os.path.exists(os.path.join(out, "dataset_info.json")) and not force:
        print(f"[gen] {name}: already present, skip")
        return
    os.makedirs(out, exist_ok=True)
    R = CANON2DATASET[name]
    trials = _trial_set(name, rng)
    rows = {"SisFall": [], "KFall": [], "FallAllD": [], "UMAFall": []}[name]
    print(f"[gen] {name}: {len(trials)} trials ...", flush=True)

    prof = DOMAIN_PROFILES[name]
    for sid, age, label_kind, is_fall in trials:
        older = age == "older"
        sp = _subject_params(older, rng)
        # domain profile: hardware / protocol characteristics (must be applied
        # BEFORE synthesis so noise and dynamics inherit them)
        sp["noise_a"] *= prof["noise"]
        sp["noise_g"] *= prof["noise"]
        sp["gait_f"] *= prof["gait"]
        sp["strength"] *= prof["strength"]
        stance = "walk" if not older else "standstill"
        kind = label_kind                      # activity name (fall template / ADL name)
        if is_fall:
            kinds = list(FALL_BY_TYPE.keys())
            a, g, labels, i_on, i_im = make_fall(kinds[int(label_kind[1:]) % len(kinds)],
                                                 meta["native_rate"], sp, stance)
        else:
            menu = prof.get("adl_menu", DEFAULT_ADL_MENU)
            k = menu[int(label_kind[1:]) % len(menu)]
            a, g, labels, i_on, i_im = make_adl(k, meta["native_rate"], sp)
            kind = k

        # ---- domain profile: gain + sensor low-pass (applied after synthesis) ----
        a = a * prof["gain"]
        g = g * prof["gain"]
        if prof["lp"]:
            sos = sig.butter(4, prof["lp"], "low", fs=meta["native_rate"], output="sos")
            a = sig.sosfiltfilt(sos, a, axis=0)
            g = sig.sosfiltfilt(sos, g, axis=0)

        # mounting rotation + per-subject extra misrotation (±4°)
        mis = rng.normal(0, np.deg2rad(4), 3)
        Rm = R @ _rot_matrix(mis)
        a_r = _rotate(a, Rm); g_r = _rotate(g, Rm)
        n = len(a)

        if name == "SisFall":
            fn = f"{label_kind[0]}{int(label_kind[1:]):02d}_S{sid}_R01.txt"
            adc = np.column_stack([
                np.round(a_r[:, 0] / 0.0039).astype(np.int64),
                np.round(a_r[:, 1] / 0.0039).astype(np.int64),
                np.round(a_r[:, 2] / 0.0039).astype(np.int64),
                np.round((a_r[:, 0] + rng.normal(0, 0.0008, n)) / 0.0039).astype(np.int64),
                np.round((a_r[:, 1] + rng.normal(0, 0.0008, n)) / 0.0039).astype(np.int64),
                np.round((a_r[:, 2] + rng.normal(0, 0.0008, n)) / 0.0039).astype(np.int64),
                np.round(g_r[:, 0] * 14.375).astype(np.int64),
                np.round(g_r[:, 1] * 14.375).astype(np.int64),
                np.round(g_r[:, 2] * 14.375).astype(np.int64),
            ])
            np.savetxt(os.path.join(out, fn), adc, fmt="%d", delimiter=",")
            np.savetxt(os.path.join(out, fn.replace(".txt", ".labels.csv")),
                       labels, fmt="%d")
            rows.append([fn, sid, age, kind, i_on if i_on else 0, i_im if i_im else 0])
        elif name == "KFall":
            t_no = int(label_kind[1:])
            if label_kind.startswith("D"):  # ADLs T07..T14, falls T01..T06
                t_no = 6 + t_no
            fn = f"S{sid}T{t_no:02d}R01.csv"
            _write_csv(os.path.join(out, fn), ["time", "ax", "ay", "az", "gx", "gy", "gz"],
                       np.column_stack([np.arange(n) / meta["native_rate"], a_r, g_r]))
            rows.append([fn, sid, age, kind, i_on if i_on else 0, i_im if i_im else 0])
        elif name == "FallAllD":
            d = os.path.join(out, f"Subject{sid}", kind)
            os.makedirs(d, exist_ok=True)
            aw, gw = a_r, g_r
            ar, gr = wrist_view(a_r, g_r, meta["native_rate"], sp,
                                kind in ("walk", "jog", "stairs"))
            _write_csv(os.path.join(d, "acc_0.csv"), ["ax", "ay", "az"], aw)
            _write_csv(os.path.join(d, "gyr_0.csv"), ["gx", "gy", "gz"], gw)
            _write_csv(os.path.join(d, "acc_1.csv"), ["ax", "ay", "az"], ar)
            _write_csv(os.path.join(d, "gyr_1.csv"), ["gx", "gy", "gz"], gr)
            rows.append([f"Subject{sid}", kind, sid, age,
                         i_on if i_on else 0, i_im if i_im else 0])
        else:  # UMAFall
            aw, gw = a_r, g_r
            ar, gr = wrist_view(a_r, g_r, meta["native_rate"], sp,
                                kind in ("walk", "jog", "stairs"))
            fn = f"UMA_{sid}_{kind}.csv"
            _write_csv(os.path.join(out, fn),
                       ["time", "w_ax", "w_ay", "w_az", "w_gx", "w_gy", "w_gz",
                        "r_ax", "r_ay", "r_az", "r_gx", "r_gy", "r_gz"],
                       np.column_stack([np.arange(n) / meta["native_rate"],
                                        aw, gw, ar, gr]))
            rows.append([fn, sid, age, kind, i_on if i_on else 0, i_im if i_im else 0])

    # meta files (identical names for real-data drop-in)
    if name in ("SisFall", "KFall"):
        _write_csv(os.path.join(out, "trials_meta.csv"),
                   ["file", "subject", "age", "kind", "onset_frame", "impact_frame"], rows)
    elif name == "FallAllD":
        _write_csv(os.path.join(out, "subjects.csv"),
                   ["subject_dir", "kind", "subject", "age", "onset_frame", "impact_frame"], rows)
    else:
        _write_csv(os.path.join(out, "trials_meta.csv"),
                   ["file", "subject", "age", "kind", "onset_frame", "impact_frame"], rows)

    if name == "SisFall":
        json.dump({"accel": {"scale": 0.0039, "offset": 0.0},
                   "gyro": {"scale": 1.0 / 14.375, "offset": 0.0},
                   "note": "simulated Readme constants; replace with real SisFall Readme.txt values"},
                  open(os.path.join(out, "imus.json"), "w"), indent=2)
    json.dump({"name": name, "version": "SIMULATED-1.0",
               "native_rate": meta["native_rate"],
               "rotation": meta["rotation"],
               "placement": meta["placement"],
               "source": "replace with real dataset URL once downloaded",
               "download_date": "2026-08-22"},
              open(os.path.join(out, "dataset_info.json"), "w"), indent=2)
    print(f"[gen] {name}: done, {len(trials)} trials -> {out}")


def _rot_matrix(e):
    rx, ry, rz = e
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def generate_all(force=False, only=None):
    for name in config.ACTIVE_DATASETS:
        if only and name not in only:
            continue
        generate_dataset(name, force)


if __name__ == "__main__":
    import sys
    generate_all(force="--force" in sys.argv, only=sys.argv[1:] or None)
