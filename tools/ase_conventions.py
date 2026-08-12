"""
ase_conventions.py — Shared field-normalization helpers for all MLIP models.

All internal/run_*.py scripts import from here to ensure uniform output format
per docs/unified_api_spec.md §5.

Key design: ASE extxyz reserves `energy` and `forces` as special keys — when
written then re-read, ASE silently moves them to a calculator object, making
them invisible to atoms.info / atoms.arrays. This caused bugs in OC20 runs.
Solution: use ref_* for ground truth, pred_* for predictions.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms

# ---------------------------------------------------------------------------
# Canonical field names (single source of truth)
# ---------------------------------------------------------------------------
REF_ENERGY_KEY = "ref_energy"
REF_FORCES_KEY = "ref_forces"
PRED_ENERGY_KEY = "pred_energy"
PRED_FORCES_KEY = "pred_forces"

# Legacy / model-specific keys that get normalized on read
_LEGACY_ENERGY_KEYS = ("MACE_energy", "mace_energy", "dpmd_energy", "energy")
_LEGACY_FORCES_KEYS = ("MACE_forces", "mace_forces", "dpmd_forces", "forces")


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

def set_ref(atoms: Atoms, energy: float, forces: np.ndarray) -> None:
    """Store ground-truth labels with ref_ prefix (survives ASE round-trips)."""
    atoms.info[REF_ENERGY_KEY] = float(energy)
    atoms.arrays[REF_FORCES_KEY] = np.asarray(forces, dtype=np.float64)


def get_ref_energy(atoms: Atoms) -> float | None:
    """Return ref_energy (eV), or first matching legacy key as fallback.

    The fallback handles older datasets that may not yet use ref_ prefix.
    """
    if REF_ENERGY_KEY in atoms.info:
        return float(atoms.info[REF_ENERGY_KEY])
    for k in _LEGACY_ENERGY_KEYS:
        if k in atoms.info:
            return float(atoms.info[k])
    # ASE calculator fallback (last resort — flaky on re-read)
    if atoms.calc is not None:
        try:
            return float(atoms.get_potential_energy())
        except Exception:
            pass
    return None


def get_ref_forces(atoms: Atoms) -> np.ndarray | None:
    """Return ref_forces array (N,3), or first matching legacy key as fallback."""
    if REF_FORCES_KEY in atoms.arrays:
        return np.asarray(atoms.arrays[REF_FORCES_KEY], dtype=np.float64)
    for k in _LEGACY_FORCES_KEYS:
        if k in atoms.arrays:
            return np.asarray(atoms.arrays[k], dtype=np.float64)
    if atoms.calc is not None:
        try:
            return np.asarray(atoms.get_forces(), dtype=np.float64)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def set_pred(atoms: Atoms, energy: float, forces: np.ndarray) -> None:
    """Store model predictions with pred_ prefix.

    Also removes any model-specific keys (MACE_energy, dpmd_energy, etc.)
    to prevent downstream confusion.

    Call this right before writing output extxyz.
    """
    # Store canonical pred_ fields
    atoms.info[PRED_ENERGY_KEY] = float(energy)
    atoms.arrays[PRED_FORCES_KEY] = np.asarray(forces, dtype=np.float64)

    # Remove model-specific legacy keys so downstream tools only see pred_*
    for k in _LEGACY_ENERGY_KEYS:
        atoms.info.pop(k, None)
    for k in _LEGACY_FORCES_KEYS:
        if k in atoms.arrays and k != PRED_FORCES_KEY:
            del atoms.arrays[k]


def get_pred_energy(atoms: Atoms) -> float | None:
    """Return pred_energy (eV) if present, else None."""
    if PRED_ENERGY_KEY in atoms.info:
        return float(atoms.info[PRED_ENERGY_KEY])
    # Fallback: legacy keys (for reading old predictions during migration)
    for k in _LEGACY_ENERGY_KEYS:
        if k in atoms.info:
            return float(atoms.info[k])
    return None


def get_pred_forces(atoms: Atoms) -> np.ndarray | None:
    """Return pred_forces array (N,3) if present, else None."""
    if PRED_FORCES_KEY in atoms.arrays:
        return np.asarray(atoms.arrays[PRED_FORCES_KEY], dtype=np.float64)
    for k in _LEGACY_FORCES_KEYS:
        if k in atoms.arrays:
            return np.asarray(atoms.arrays[k], dtype=np.float64)
    return None


# ---------------------------------------------------------------------------
# Batch normalization (for run_inference.py output)
# ---------------------------------------------------------------------------

def normalize_predictions(atoms_list: list[Atoms]) -> list[Atoms]:
    """Ensure every atom in the list has pred_energy/pred_forces in canonical keys.

    Detaches calculator references (atoms.calc = None) so the predictions
    survive ASE extxyz round-trips.
    """
    for atoms in atoms_list:
        if atoms.calc is not None:
            try:
                energy = float(atoms.get_potential_energy())
                forces = np.asarray(atoms.get_forces(), dtype=np.float64)
                set_pred(atoms, energy, forces)
            except Exception:
                pass
            atoms.calc = None
        # If pred_ keys already set (e.g. from CLI runner), just detach
        elif PRED_ENERGY_KEY not in atoms.info:
            # Try to lift model-specific keys to pred_*
            e = get_pred_energy(atoms)
            f = get_pred_forces(atoms)
            if e is not None:
                atoms.info[PRED_ENERGY_KEY] = float(e)
            if f is not None:
                atoms.arrays[PRED_FORCES_KEY] = np.asarray(f, dtype=np.float64)
    return atoms_list
