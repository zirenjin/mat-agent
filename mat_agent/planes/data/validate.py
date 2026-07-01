#!/usr/bin/env python3
"""Reusable data validation affordances for MLIP datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
from ase.io import read


def validate_extxyz(path: str | Path, *, require_labels: bool = True) -> dict[str, Any]:
    """Validate basic extxyz structure and optional ref labels."""
    frames = read(str(path), ":")
    if not isinstance(frames, list):
        frames = [frames]
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
