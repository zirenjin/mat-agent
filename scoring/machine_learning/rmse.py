#!/usr/bin/env python3
from __future__ import annotations

import math

from _common import run_paired


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred, strict=True)) / len(y_true))


if __name__ == "__main__":
    raise SystemExit(run_paired("rmse", "Root mean squared error", rmse))
