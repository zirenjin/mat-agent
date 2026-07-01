#!/usr/bin/env python3
"""Dataset inspection helpers."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
from ase.io import read


def inspect_extxyz(path: str | Path) -> dict[str, Any]:
    """Return a compact summary of an extxyz file."""
    frames = read(str(path), ":")
    if not isinstance(frames, list):
        frames = [frames]
    element_counts: Counter[str] = Counter()
    natoms = []
    info_keys: Counter[str] = Counter()
    array_keys: Counter[str] = Counter()
    for atoms in frames:
        element_counts.update(atoms.get_chemical_symbols())
        natoms.append(len(atoms))
        info_keys.update(atoms.info.keys())
        array_keys.update(atoms.arrays.keys())
    return {
        "path": str(path),
        "n_frames": len(frames),
        "n_atoms_min": min(natoms) if natoms else 0,
        "n_atoms_max": max(natoms) if natoms else 0,
        "elements": dict(sorted(element_counts.items())),
        "info_keys": sorted(info_keys),
        "array_keys": sorted(array_keys),
    }
