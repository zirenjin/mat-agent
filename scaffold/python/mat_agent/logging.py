#!/usr/bin/env python3
"""JSONL logging helpers for agent experiments."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


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
