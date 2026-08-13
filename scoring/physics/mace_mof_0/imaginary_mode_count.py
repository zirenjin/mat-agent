#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from _common import emit, finite_float

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Count phonon frequencies below an imaginary-mode threshold", allow_abbrev=False)
    p.add_argument("--frequencies", required=True, help="Comma-separated frequencies, e.g. THz")
    p.add_argument("--threshold", type=finite_float, default=-1e-4)
    p.add_argument("--output")
    args = p.parse_args()
    values = [finite_float(x.strip()) for x in args.frequencies.split(",") if x.strip()]
    count = sum(1 for x in values if x < args.threshold)
    emit("imaginary_mode_count", float(count), args.output, n=len(values), threshold=args.threshold, min_frequency=min(values) if values else None)
