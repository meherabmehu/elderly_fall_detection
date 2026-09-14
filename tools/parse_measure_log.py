#!/usr/bin/env python3
"""Parse `measure` blocks from an ESP32 serial log into hardware_measurements.json.

The firmware (firmware/esp32_ai_final_live_deployment) prints a block like:

    ---- MEASURE ----
    inferences        : 1000
    invoke mean       : 2.071 ms
    invoke min/max    : 2.064 / 2.088 ms
    preprocess mean   : 0.181 ms
    end-to-end mean   : 2.252 ms
    arena high water  : 8348 bytes (8.2 KB) of 24576
    model flash       : 23080 bytes (22.5 KB)
    free heap         : 289012 bytes
    cnn alarms        : 0
    rule alarms       : 0
    threshold / k     : 0.95 / 2
    -----------------

every 1000 inferences and on demand. This tool takes the LAST complete block in
a log (the one with the most inferences) and merges it into
results/hardware_measurements.json, keeping every field it cannot read as
PENDING (null) instead of guessing.

Usage:
    python tools/parse_measure_log.py logs/serial_20260915_120000.log \
        --measurements experiments/hardware_measurement/hardware_measurements.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

PATTERNS = {
    "n_inferences_averaged": re.compile(r"inferences\s*:\s*(\d+)"),
    "inference_latency_ms": re.compile(r"invoke mean\s*:\s*([\d.]+)\s*ms"),
    "invoke_min_ms": re.compile(r"invoke min/max\s*:\s*([\d.]+)\s*/"),
    "invoke_max_ms": re.compile(r"invoke min/max\s*:\s*[\d.]+\s*/\s*([\d.]+)\s*ms"),
    "preprocess_ms": re.compile(r"preprocess mean\s*:\s*([\d.]+)\s*ms"),
    "end_to_end_latency_ms": re.compile(r"end-to-end mean\s*:\s*([\d.]+)\s*ms"),
    "arena_high_water_bytes": re.compile(r"arena high water\s*:\s*(\d+)\s*bytes"),
    "model_flash_bytes": re.compile(r"model flash\s*:\s*(\d+)\s*bytes"),
    "free_heap_bytes": re.compile(r"free heap\s*:\s*(\d+)\s*bytes"),
    "cnn_alarms": re.compile(r"cnn alarms\s*:\s*(\d+)"),
    "rule_alarms": re.compile(r"rule alarms\s*:\s*(\d+)"),
    "threshold": re.compile(r"threshold / k\s*:\s*([\d.]+)\s*/"),
    "k": re.compile(r"threshold / k\s*:\s*[\d.]+\s*/\s*(\d+)"),
}


def parse_blocks(text: str) -> list[dict]:
    blocks = []
    for m in re.finditer(r"---- MEASURE ----(.*?)-----------------", text, re.S):
        body = m.group(1)
        row = {}
        for key, pat in PATTERNS.items():
            hit = pat.search(body)
            if hit:
                v = hit.group(1)
                row[key] = float(v) if ("." in v or "latency" in key or "ms" in key or key == "threshold") else int(v)
        blocks.append(row)
    return blocks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--measurements", type=Path,
                    default=Path("experiments/hardware_measurement/hardware_measurements.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    text = args.log.read_text(errors="replace")
    blocks = parse_blocks(text)
    if not blocks:
        raise SystemExit("no complete MEASURE block found in the log")
    best = max(blocks, key=lambda b: b.get("n_inferences_averaged", 0))
    print("latest complete MEASURE block:")
    print(json.dumps(best, indent=2))

    target = args.measurements
    data = json.loads(target.read_text()) if target.exists() else {}
    # Only fill fields this protocol measures. Latency/RAM fields stay null
    # (PENDING) if absent from the block; nothing is ever fabricated.
    for key in ("n_inferences_averaged", "inference_latency_ms",
                "end_to_end_latency_ms", "arena_high_water_bytes",
                "invoke_min_ms", "invoke_max_ms", "preprocess_ms",
                "cnn_alarms", "rule_alarms", "threshold", "k"):
        if key in best:
            data[key] = best[key]
    data["date"] = dt.date.today().isoformat()
    data["source_log"] = str(args.log)
    if args.dry_run:
        print("dry run - not writing")
        return 0
    target.write_text(json.dumps(data, indent=2))
    print(f"updated {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
