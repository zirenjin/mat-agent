#!/usr/bin/env python3
"""Artifact utilities for CodeAct materials workflows."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def ensure_run_dir(path: str | Path) -> Path:
    """Create and return a run directory."""
    run_dir = Path(path)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL artifact."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
