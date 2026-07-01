#!/usr/bin/env python3
"""Checkpoint discovery helpers."""
from __future__ import annotations

from pathlib import Path


def newest_checkpoint(path: str | Path) -> str | None:
    """Return the newest common checkpoint file under a directory."""
    root = Path(path)
    candidates = []
    for pattern in ("*.model", "*.pt", "*.pth", "*.ckpt", "checkpoint"):
        candidates.extend(p for p in root.rglob(pattern) if p.is_file())
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))
