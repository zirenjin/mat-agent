#!/usr/bin/env python3
"""Workspace discovery for mat-agent."""
from __future__ import annotations

from pathlib import Path


def repo_root(start: str | Path | None = None) -> Path:
    """Find the repository root by walking upward to a models directory."""
    cur = Path(start or __file__).resolve()
    if cur.is_file():
        cur = cur.parent
    for parent in [cur, *cur.parents]:
        if (parent / "models").is_dir() and (parent / "tools").is_dir():
            return parent
    raise RuntimeError("could not locate mat-agent repository root")
