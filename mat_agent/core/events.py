#!/usr/bin/env python3
"""Event logging helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from .artifacts import append_jsonl
from .schemas import utc_now


def emit_event(path: str | Path, event: str, **fields: Any) -> None:
    """Append a timestamped event to a JSONL log."""
    payload = {"event": event, "timestamp": utc_now(), **fields}
    append_jsonl(path, payload)
