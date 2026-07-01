#!/usr/bin/env python3
"""Evidence summary helpers."""
from __future__ import annotations

from typing import Any


def summarize_metrics(metrics: dict[str, Any]) -> str:
    """Create a compact textual metric summary."""
    return ", ".join(f"{k}={v}" for k, v in sorted(metrics.items()))
