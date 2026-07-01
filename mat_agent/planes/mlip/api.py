#!/usr/bin/env python3
"""Resolve model API entrypoints."""
from __future__ import annotations

from pathlib import Path
from mat_agent.core.workspace import repo_root

MODEL_ALIASES = {"mace": "mace", "deepmd": "deepmd", "dpa": "deepmd", "fairchem": "fairchem", "uma": "fairchem"}


def api_script(model: str, command: str, *, root: str | Path | None = None) -> Path:
    """Return models/<model>/api/<command>.sh for a supported model."""
    key = MODEL_ALIASES.get(model)
    if key is None:
        raise ValueError(f"unknown model: {model}")
    if command not in {"train", "inference", "evaluate"}:
        raise ValueError(f"unknown model API command: {command}")
    base = Path(root) if root else repo_root()
    path = base / "models" / key / "api" / f"{command}.sh"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
