#!/usr/bin/env python3
"""Capture ESP32 serial output to a timestamped log for measurement sessions.

Used by the hardware measurement protocol (firmware/MEASUREMENT.md): log the
boot banner, the periodic `measure` blocks and any FALL ALERT lines during a
bench session, then feed the log to parse_measure_log.py.

Usage:
    python tools/serial_capture.py --port COM5
    python tools/serial_capture.py --port /dev/ttyUSB0 --baud 115200 --out logs/session_01.log
    python tools/serial_capture.py --port COM5 --send test --send measure

Requires:  pip install pyserial
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="serial port, e.g. COM5 or /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", type=Path, default=None,
                    help="log file (default: logs/serial_YYYYMMDD_HHMMSS.log)")
    ap.add_argument("--send", action="append", default=[],
                    help="firmware command to send after connect (repeatable), "
                         "e.g. --send test --send measure")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds to record; 0 = until Ctrl+C")
    args = ap.parse_args()

    try:
        import serial
    except ImportError:
        sys.exit("pyserial not installed:  pip install pyserial")

    out = args.out
    if out is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("logs") / f"serial_{stamp}.log"
    out.parent.mkdir(parents=True, exist_ok=True)

    with serial.Serial(args.port, args.baud, timeout=0.2) as ser, out.open("w") as log:
        print(f"logging {args.port} @ {args.baud} -> {out}")
        log.write(f"# serial capture {dt.datetime.now().isoformat(timespec='seconds')} "
                  f"port={args.port} baud={args.baud}\n")
        time.sleep(2.0)  # ESP32 resets on connect; let it boot
        for cmd in args.send:
            ser.write((cmd.strip() + "\n").encode())
            log.write(f">>> {cmd.strip()}\n")
            time.sleep(0.5)
        t0 = time.time()
        try:
            while True:
                if args.duration and (time.time() - t0) >= args.duration:
                    break
                chunk = ser.read(4096)
                if chunk:
                    text = chunk.decode("utf-8", "replace")
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    log.write(text)
                    log.flush()
        except KeyboardInterrupt:
            pass
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
