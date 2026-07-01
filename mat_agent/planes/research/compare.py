#!/usr/bin/env python3
"""Generic metric comparison helpers."""
from __future__ import annotations


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """Return after-before deltas for shared numeric metric keys."""
    return {k: float(after[k]) - float(before[k]) for k in sorted(set(before) & set(after))}
