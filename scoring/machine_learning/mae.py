#!/usr/bin/env python3
from __future__ import annotations

from _common import run_paired


def mae(y_true: list[float], y_pred: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred, strict=True)) / len(y_true)


if __name__ == "__main__":
    raise SystemExit(run_paired("mae", "Mean absolute error", mae))
