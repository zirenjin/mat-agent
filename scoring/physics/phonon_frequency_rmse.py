#!/usr/bin/env python3
from __future__ import annotations

import math

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "machine_learning"))
from _common import paired_parser, paired_values, emit  # noqa: E402


if __name__ == "__main__":
    args = paired_parser("RMSE between predicted and reference phonon frequencies").parse_args()
    y_true, y_pred = paired_values(args)
    value = math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred, strict=True)) / len(y_true))
    emit("phonon_frequency_rmse", value, len(y_true), args.output)
