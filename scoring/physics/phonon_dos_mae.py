#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "machine_learning"))
from _common import paired_parser, paired_values, emit  # noqa: E402


if __name__ == "__main__":
    args = paired_parser("MAE between predicted and reference phonon DOS samples on the same grid").parse_args()
    y_true, y_pred = paired_values(args)
    value = sum(abs(a - b) for a, b in zip(y_true, y_pred, strict=True)) / len(y_true)
    emit("phonon_dos_mae", value, len(y_true), args.output)
