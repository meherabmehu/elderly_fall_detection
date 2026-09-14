#!/usr/bin/env python3
"""Fetch the datasets this project uses and verify them before extraction.

What lives where (see datasets/README.md for why):

  UMAFall   -> downloadable here from the repo's GitHub release 'datasets-v1'
               (CC BY 4.0, redistribution with attribution allowed).
               Falls back to the original figshare link.
  SisFall   -> downloadable here from the same GitHub release
               (public research dataset; mirror credited to the original
               authors — cite Sucerquia et al. 2017).
  FallAllD  -> NOT auto-downloadable: IEEE DataPort requires an account.
               The script prints where to log in and where the file goes.
  KFall     -> NEVER downloadable here: its terms forbid transferring the
               dataset to third parties. Register at the official site.

Usage:
    python tools/download_datasets.py                 # fetch what is fetchable
    python tools/download_datasets.py --check         # verify local copies only
    python tools/download_datasets.py --into datasets --extract
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = "meherabmehu/elderly_fall_detection"
RELEASE_TAG = "datasets-v1"

# (asset, sha256 placeholder-filled by SHA256SUMS.txt, upstream fallback, dest dir)
DATASETS = {
    "UMAFall": {
        "asset": "UMAFall_Dataset.zip",
        "upstream": "https://ndownloader.figshare.com/files/43076140",
        "dest": "umafall",
        "license": "CC BY 4.0 — cite Casilari et al. 2017",
    },
    "SisFall": {
        "asset": "SisFall.zip",
        "upstream": None,  # original UdeA link is dead; release holds the mirror
        "dest": "sisfall",
        "license": "public research dataset — cite Sucerquia et al. 2017",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sums(root: Path) -> dict[str, str]:
    sums = {}
    f = root / "datasets" / "SHA256SUMS.txt"
    if f.exists():
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                sums[parts[1]] = parts[0]
    return sums


def download(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)


def fetch_one(name: str, info: dict, root: Path, sums: dict, extract: bool) -> bool:
    zpath = root / "datasets" / info["dest"] / info["asset"]
    want = sums.get(info["asset"])
    if zpath.exists() and want and sha256(zpath) == want:
        print(f"[{name}] already present, sha256 OK")
    else:
        url = info["upstream"]
        if url is None:
            url = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{info['asset']}"
        download(url, zpath)
        got = sha256(zpath)
        if want and got != want:
            zpath.unlink(missing_ok=True)
            print(f"[{name}] CHECKSUM MISMATCH: {got} != {want} (deleted)")
            return False
        print(f"[{name}] downloaded, sha256 {got[:16]}… OK")
    if extract:
        out = root / "datasets" / info["dest"]
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out, filter="data")
        print(f"[{name}] extracted to {out}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--into", type=Path, default=Path("."), help="repo root")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    root = args.into.resolve()
    sums = load_sums(root)
    ok = True
    for name, info in DATASETS.items():
        try:
            ok &= fetch_one(name, info, root, sums, args.extract and not args.check)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] FAILED: {e}")
            ok = False

    print()
    print("FallAllD: get it from IEEE DataPort (free account):")
    print("  https://ieee-dataport.org/open-access/fallalld-comprehensive-dataset")
    print("  -> place FallAllD/ (or the official pkl) under datasets/fallalld/")
    print()
    print("KFall: register at https://sites.google.com/view/kfalldataset/home")
    print("  Its terms FORBID redistribution, so it is never in this repo or its")
    print("  release assets. Place it under datasets/kfall/ yourself.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
