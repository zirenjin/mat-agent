#!/usr/bin/env python3
"""Generic split helpers for small MLIP datasets."""
from __future__ import annotations

from pathlib import Path
from ase.io import read, write


def split_extxyz(input_path: str | Path, output_dir: str | Path, *, train: int, val: int, test: int) -> dict[str, str]:
    """Create deterministic contiguous train/val/test splits."""
    frames = read(str(input_path), ":")
    if not isinstance(frames, list):
        frames = [frames]
    if train + val + test > len(frames):
        raise ValueError("requested split is larger than dataset")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    parts = {
        "train": frames[:train],
        "val": frames[train:train + val],
        "test": frames[train + val:train + val + test],
    }
    paths = {}
    for name, atoms_list in parts.items():
        path = out / f"{name}.extxyz"
        write(str(path), atoms_list, format="extxyz")
        paths[name] = str(path)
    return paths
