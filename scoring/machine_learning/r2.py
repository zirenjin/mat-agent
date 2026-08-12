#!/usr/bin/env python3
from __future__ import annotations

from _common import run_paired


def r2(y_true: list[float], y_pred: list[float]) -> float:
    mean = sum(y_true) / len(y_true)
    ss_tot = sum((x - mean) ** 2 for x in y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred, strict=True))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


if __name__ == "__main__":
    raise SystemExit(run_paired("r2", "Coefficient of determination", r2))
