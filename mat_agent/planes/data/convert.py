#!/usr/bin/env python3
"""Generic data conversion affordances."""
from __future__ import annotations

from pathlib import Path
from ase.io import read, write


def copy_extxyz(input_path: str | Path, output_path: str | Path) -> int:
    """Read and rewrite an extxyz file using ASE canonical serialization."""
    frames = read(str(input_path), ":")
    if not isinstance(frames, list):
        frames = [frames]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write(str(out), frames, format="extxyz")
    return len(frames)
