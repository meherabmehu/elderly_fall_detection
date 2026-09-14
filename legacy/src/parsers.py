"""parsers.py — read each dataset's on-disk format into a common Trial structure.

Single entry point: load_dataset(name, placement) -> list[Trial].
Works identically on the synthetic mirror (data/synthetic/<name>) and on the
real releases once dropped into data/raw/<name> (see README "real data" section).

Trial fields (SI units after parsing: accel g, gyro deg/s, canonical axes):
    dataset, subject, age, kind, placement, rate,
    accel (n,3), gyro (n,3),
    labels (n,) 0=non-fall 1=alert 2=fall   (None when the dataset has none)
    onset, impact   (None or sample index at `rate`)
NOTE: preprocessing applies the documented inverse mounting rotation — parsers
deliver data in the dataset's native axes.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Trial:
    __slots__ = ["dataset", "subject", "age", "kind", "placement", "rate",
                 "accel", "gyro", "labels", "onset", "impact"]

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __repr__(self):
        return (f"<Trial {self.dataset}/{self.subject}/{self.kind} "
                f"({self.placement}) {self.accel.shape}>")


def _root(name):
    """Prefer real data (data/raw/<name>) when present, else the synthetic mirror."""
    p = os.path.join(config.RAW, name)
    if os.path.isdir(p) and any(os.scandir(p)):
        return p
    return os.path.join(config.SYN, name)


# ------------------------------------------------ tolerant column picking
def _pick(df, stem, axis):
    """Find the column for (stem, axis), tolerant of _ / casing, e.g. ('acc','x')."""
    candidates = [f"{stem}{axis}", f"{stem}_{axis}",
                  f"{stem.upper()}{axis.upper()}", f"{stem.upper()}_{axis.upper()}"]
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in low:
            return low[c.lower()]
    return None


def _take(df, stems, axis):
    for s in stems:
        c = _pick(df, s, axis)
        if c:
            return df[c].to_numpy(dtype=float)
    raise ValueError(f"no column for axis '{axis}' (stems {stems}); "
                     f"available: {list(df.columns)[:14]}")


def _as_si(df, stems):
    """(n,3) from axis columns x,y,z using the first stem that matches."""
    return np.column_stack([_take(df, stems, a) for a in "xyz"])


def _apply_rotation(t: Trial, name):
    """Preprocessing step 4.2: invert the documented mounting rotation."""
    R = np.array(config.DATASETS[name]["rotation"]["matrix"], float).reshape(3, 3)
    t.accel = np.einsum("ji,tj->ti", R, t.accel)
    t.gyro = np.einsum("ji,tj->ti", R, t.gyro)
    return t


# ------------------------------------------------ SisFall (+ Enhanced labels)
def parse_sisfall(name="SisFall"):
    root = _root(name)
    info = json.load(open(os.path.join(root, "dataset_info.json")))
    rate = float(info.get("native_rate", config.DATASETS[name]["native_rate"]))
    imu = config.SISFALL_IMU
    imu_path = os.path.join(root, "imus.json")
    if os.path.exists(imu_path):
        imu = json.load(open(imu_path))
    meta_path = os.path.join(root, "trials_meta.csv")
    meta = pd.read_csv(meta_path) if os.path.exists(meta_path) else None
    trials = []
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".txt"):
            continue
        arr = np.loadtxt(os.path.join(root, fn), delimiter=",", dtype=float)
        arr = arr.reshape(-1, 9)
        accel = np.column_stack([arr[:, 0], arr[:, 1], arr[:, 2]])  # first accel
        gyro = np.column_stack([arr[:, 6], arr[:, 7], arr[:, 8]])
        accel = accel * imu["accel"]["scale"] + imu["accel"]["offset"]
        gyro = gyro * imu["gyro"]["scale"] + imu["gyro"]["offset"]
        labels_path = os.path.join(root, fn.replace(".txt", ".labels.csv"))
        labels = np.loadtxt(labels_path, dtype=np.int8) \
            if os.path.exists(labels_path) else None
        m = re.match(r"(F|D)(\d+)_([A-Z]+)(\d+)_R(\d+)", fn)
        subject, age = ("?", "young")
        if m:
            subject = f"{m.group(3)}{int(m.group(4)):02d}"
            age = "older" if m.group(3) == "SE" else "young"
        onset = impact = None
        if meta is not None and m:
            row = meta[meta["file"] == fn]
            if len(row):
                onset = int(row.iloc[0]["onset_frame"]) or None
                impact = int(row.iloc[0]["impact_frame"]) or None
                if "age" in meta.columns:
                    age = str(row.iloc[0]["age"])
        t = Trial(dataset=name, subject=subject, age=age,
                  kind=fn.split("_")[0], placement="waist", rate=rate,
                  accel=accel, gyro=gyro, labels=labels, onset=onset, impact=impact)
        _apply_rotation(t, name)
        trials.append(t)
    return trials


# ------------------------------------------------ KFall
def parse_kfall(name="KFall"):
    root = _root(name)
    info = json.load(open(os.path.join(root, "dataset_info.json")))
    rate = float(info.get("native_rate", config.DATASETS[name]["native_rate"]))
    meta = None
    for mp in ("labels.csv", "trials_meta.csv"):
        if os.path.exists(os.path.join(root, mp)):
            meta = pd.read_csv(os.path.join(root, mp))
            break
    trials = []
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".csv") or fn.startswith(("label", "trial")):
            continue
        df = pd.read_csv(os.path.join(root, fn))
        acc = _as_si(df, ["acc", "a"])
        gyr = _as_si(df, ["gyr", "g"])
        m = re.match(r"S([A-Z0-9]+)T(\d+)R(\d+)", fn)
        subject, age, kind = "?", "young", "?"
        onset = impact = None
        if meta is not None and m:
            row = meta[meta["file"] == fn]
            if m:
                subject = m.group(1)
                kind = f"T{m.group(2)}"
            if len(row):
                age = str(row.iloc[0]["age"])
                onset = int(row.iloc[0]["onset_frame"]) or None
                impact = int(row.iloc[0]["impact_frame"]) or None
        t = Trial(dataset=name, subject=subject, age=age, kind=kind,
                  placement="lowback", rate=rate, accel=acc, gyro=gyr,
                  labels=None, onset=onset, impact=impact)
        _apply_rotation(t, name)
        trials.append(t)
    return trials


# ------------------------------------------------ FallAllD
def parse_fallalld(name="FallAllD"):
    root = _root(name)
    rate = float(config.DATASETS[name]["native_rate"])
    meta_path = os.path.join(root, "subjects.csv")
    meta = pd.read_csv(meta_path) if os.path.exists(meta_path) else None
    trials = []
    for subj_dir in sorted(os.listdir(root)):
        sp = os.path.join(root, subj_dir)
        if not os.path.isdir(sp) or subj_dir.startswith((".")):
            continue
        for trial_dir in sorted(os.listdir(sp)):
            td = os.path.join(sp, trial_dir)
            if not os.path.isdir(td):
                continue
            acc0 = gyr0 = acc1 = gyr1 = None
            for f in sorted(os.listdir(td)):
                if not f.endswith(".csv"):
                    continue
                df = pd.read_csv(os.path.join(td, f))
                low = f.lower()
                if re.search(r"acc.*_?0", low) or re.search(r"_?0.*acc", low):
                    acc0 = _as_si(df, ["acc", "a"])
                elif re.search(r"acc.*_?1", low) or re.search(r"_?1.*acc", low):
                    acc1 = _as_si(df, ["acc", "a"])
                elif re.search(r"gyr.*_?0", low) or re.search(r"_?0.*gyr", low):
                    gyr0 = _as_si(df, ["gyr", "g"])
                elif re.search(r"gyr.*_?1", low) or re.search(r"_?1.*gyr", low):
                    gyr1 = _as_si(df, ["gyr", "g"])
            if acc0 is None or gyr0 is None:
                continue
            m = re.match(r"Subject(\w+)", subj_dir)
            subject = m.group(1) if m else subj_dir
            age, onset, impact = "young", None, None
            is_fall = bool(re.match(r"F\d+", trial_dir))
            if meta is not None:
                mrows = meta[meta["subject_dir"] == subj_dir]
                if len(mrows):
                    rr = mrows[mrows["kind"] == trial_dir]
                    row = rr.iloc[0] if len(rr) else mrows.iloc[0]
                    if "age" in meta.columns:
                        age = str(row["age"])
                    if "onset_frame" in meta.columns and "impact_frame" in meta.columns:
                        o, i_ = row["onset_frame"], row["impact_frame"]
                        if is_fall and (int(o or 0) > 0 and int(i_ or 0) > 0):
                            onset, impact = int(o), int(i_)
            if is_fall and (onset is None or impact is None):
                # FallAllD has no per-frame annotations; derive the impact from
                # the |a| peak of the trial (used for POST-fall labelling only;
                # C2 pre-impact claims rest on KFall + SisFall Enhanced).
                mag = np.linalg.norm(acc0, axis=1)
                k = max(15, int(0.2 * rate))
                sm = np.convolve(mag, np.ones(k) / k, mode="same")
                impact = int(np.argmax(sm))
                onset = None
            for placement, acc, gyr in (("waist", acc0, gyr0), ("wrist", acc1, gyr1)):
                if acc is None or gyr is None:
                    continue
                t = Trial(dataset=name, subject=subject, age=age,
                          kind=trial_dir.split("_")[0], placement=placement,
                          rate=rate, accel=acc.copy(), gyro=gyr.copy(),
                          labels=None, onset=onset, impact=impact)
                _apply_rotation(t, name)
                trials.append(t)
    return trials


# ------------------------------------------------ UMAFall
def parse_umafall(name="UMAFall"):
    root = _root(name)
    rate = float(config.DATASETS[name]["native_rate"])
    meta_path = os.path.join(root, "trials_meta.csv")
    meta = pd.read_csv(meta_path) if os.path.exists(meta_path) else None
    trials = []
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".csv") or fn.startswith(("label", "trial", "meta")):
            continue
        df = pd.read_csv(os.path.join(root, fn))
        if not any(str(c).startswith("w_") for c in df.columns):
            continue                        # not a UMAFall trace file
        acc = _as_si(df, ["w_acc", "w_a", "waist_acc", "acc"])
        gyr = _as_si(df, ["w_gyr", "w_g", "waist_gyr", "gyr"])
        if _pick(df, "r_acc", "x") or _pick(df, "r_a", "x"):
            accr = _as_si(df, ["r_acc", "r_a", "wrist_acc"])
            gyrr = _as_si(df, ["r_gyr", "r_g", "wrist_gyr"])
        else:
            accr, gyrr = acc, gyr
        m = re.match(r"UMA_(\w+)_", fn)
        subject = m.group(1) if m else fn
        age, onset, impact = "young", None, None
        if meta is not None:
            row = meta[meta["file"] == fn]
            if len(row):
                onset = int(row.iloc[0]["onset_frame"]) or None
                impact = int(row.iloc[0]["impact_frame"]) or None
                if "age" in meta.columns:
                    age = str(row.iloc[0]["age"])
        for placement, a_, g_ in (("waist", acc, gyr), ("wrist", accr, gyrr)):
            t = Trial(dataset=name, subject=subject, age=age,
                      kind=fn.split("_")[-1].replace(".csv", ""),
                      placement=placement, rate=rate,
                      accel=a_.astype(float), gyro=g_.astype(float),
                      labels=None, onset=onset, impact=impact)
            _apply_rotation(t, name)
            trials.append(t)
    return trials


PARSERS = {"SisFall": parse_sisfall, "KFall": parse_kfall,
           "FallAllD": parse_fallalld, "UMAFall": parse_umafall}


def load_dataset(name, placement="waist"):
    """Load trials of one dataset, optionally filtered by placement."""
    trials = PARSERS[name]()
    if placement != "any":
        trials = [t for t in trials if t.placement == placement]
    return trials


if __name__ == "__main__":
    for n in config.ACTIVE_DATASETS:
        tr = load_dataset(n, "any")
        fall = [t for t in tr if t.impact is not None]
        print(f"{n}: {len(tr)} trials, {len(fall)} with temporal labels;",
              "e.g.", tr[0])
