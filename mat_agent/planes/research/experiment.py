#!/usr/bin/env python3
"""Experiment logging affordances."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from mat_agent.core.artifacts import append_jsonl
from mat_agent.core.schemas import utc_now


def log_experiment(path: str | Path, *, name: str, intent: str, command: list[str] | None = None, metrics: dict[str, Any] | None = None, notes: str = "") -> None:
    """Append a generic experiment record."""
    append_jsonl(path, {
        "timestamp": utc_now(),
        "name": name,
        "intent": intent,
        "command": command or [],
        "metrics": metrics or {},
        "notes": notes,
    })
