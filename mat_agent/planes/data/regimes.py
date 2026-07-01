#!/usr/bin/env python3
"""Lightweight structure-regime descriptors."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
from ase.io import read


def summarize_geometry(path: str | Path) -> dict[str, Any]:
    """Summarize simple geometry signals useful for agent triage."""
    frames = read(str(path), ":")
    if not isinstance(frames, list):
        frames = [frames]
    volumes = []
    min_distances = []
    for atoms in frames:
        if atoms.cell.rank == 3:
            volumes.append(float(abs(atoms.get_volume())) / max(len(atoms), 1))
        if len(atoms) > 1:
            d = atoms.get_all_distances(mic=bool(np.any(atoms.pbc)))
            d = np.asarray(d, dtype=float)
            d[d <= 0.0] = np.inf
            value = float(np.min(d))
            if np.isfinite(value):
                min_distances.append(value)
    return {
        "path": str(path),
        "volume_per_atom_min": min(volumes) if volumes else None,
        "volume_per_atom_max": max(volumes) if volumes else None,
        "min_pair_distance_A": min(min_distances) if min_distances else None,
    }
