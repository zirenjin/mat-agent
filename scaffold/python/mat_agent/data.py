#!/usr/bin/env python3
"""Small data inspection and validation helpers."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
from ase.io import read


def _read_frames(path: str | Path):
    frames = read(str(path), ":")
    return frames if isinstance(frames, list) else [frames]


def inspect_extxyz(path: str | Path) -> dict[str, Any]:
    """Return a compact summary of an extxyz file."""
    frames = _read_frames(path)
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


def validate_extxyz(path: str | Path, *, require_labels: bool = True) -> dict[str, Any]:
    """Validate basic extxyz structure and optional `ref_*` labels."""
    frames = _read_frames(path)
    missing_energy: list[int] = []
    missing_forces: list[int] = []
    bad_forces: list[int] = []
    elements: set[str] = set()
    for idx, atoms in enumerate(frames):
        elements.update(atoms.get_chemical_symbols())
        if require_labels and "ref_energy" not in atoms.info:
            missing_energy.append(idx)
        if require_labels and "ref_forces" not in atoms.arrays:
            missing_forces.append(idx)
        if "ref_forces" in atoms.arrays:
            forces = np.asarray(atoms.arrays["ref_forces"])
            if forces.shape != (len(atoms), 3):
                bad_forces.append(idx)
    return {
        "path": str(path),
        "n_frames": len(frames),
        "elements": sorted(elements),
        "require_labels": require_labels,
        "missing_ref_energy": missing_energy,
        "missing_ref_forces": missing_forces,
        "bad_force_shapes": bad_forces,
        "valid": not (missing_energy or missing_forces or bad_forces),
    }
