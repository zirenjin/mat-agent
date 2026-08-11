from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    energy_mae_per_atom: float | None
    force_mae: float | None
    stress_mae: float | None
    missing_predictions: int
    n_expected: int
    n_valid: int
    hard_gate_violations: list[str]

    def as_dict(self) -> dict[str, Any]:
        missing_fraction = 0.0 if self.n_expected == 0 else self.missing_predictions / self.n_expected
        return {
            "energy_mae_per_atom": self.energy_mae_per_atom,
            "force_mae": self.force_mae,
            "stress_mae": self.stress_mae,
            "missing_predictions": self.missing_predictions,
            "n_expected": self.n_expected,
            "n_valid": self.n_valid,
            "missing_fraction": missing_fraction,
            "hard_gate_violations": self.hard_gate_violations,
        }


def mae(reference: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(predicted, dtype=float) - np.asarray(reference, dtype=float))))


def get_energy(atoms, prefix: str) -> float | None:
    key = f"{prefix}_energy"
    if key in atoms.info:
        return float(atoms.info[key])
    if prefix == "ref" and "energy" in atoms.info:
        return float(atoms.info["energy"])
    if prefix == "pred" and "pred_energy" in atoms.info:
        return float(atoms.info["pred_energy"])
    return None


def get_forces(atoms, prefix: str) -> np.ndarray | None:
    keys = [f"{prefix}_forces"]
    if prefix == "ref":
        keys.append("forces")
    if prefix == "pred":
        keys.append("pred_forces")
    for key in keys:
        if key in atoms.arrays:
            return np.asarray(atoms.arrays[key], dtype=float)
    return None


def get_stress(atoms, prefix: str) -> np.ndarray | None:
    keys = [f"{prefix}_stress"]
    if prefix == "ref":
        keys.extend(["stress", "virial", "ref_virial"])
    if prefix == "pred":
        keys.extend(["pred_stress", "pred_virial"])
    for key in keys:
        if key in atoms.info:
            return np.asarray(atoms.info[key], dtype=float)
        if key in atoms.arrays:
            return np.asarray(atoms.arrays[key], dtype=float)
    return None


def compute_metrics(label_atoms: list, pred_atoms: list, max_missing_fraction: float = 0.05, force_explosion_gate: float = 10.0) -> MetricResult:
    if len(label_atoms) != len(pred_atoms):
        raise ValueError(f"Label/prediction length mismatch: {len(label_atoms)} vs {len(pred_atoms)}")

    energy_errors: list[float] = []
    force_errors: list[np.ndarray] = []
    stress_errors: list[np.ndarray] = []
    missing = 0

    for label, pred in zip(label_atoms, pred_atoms):
        e_ref = get_energy(label, "ref")
        e_pred = get_energy(pred, "pred")
        f_ref = get_forces(label, "ref")
        f_pred = get_forces(pred, "pred")
        s_ref = get_stress(label, "ref")
        s_pred = get_stress(pred, "pred")

        if e_ref is None or e_pred is None or f_ref is None or f_pred is None:
            missing += 1
            continue

        n_atoms = max(len(label), 1)
        energy_errors.append(abs(e_pred - e_ref) / n_atoms)
        force_errors.append(np.abs(f_pred - f_ref).reshape(-1))

        if s_ref is not None and s_pred is not None:
            stress_errors.append(np.abs(np.asarray(s_pred, dtype=float) - np.asarray(s_ref, dtype=float)).reshape(-1))

    n_expected = len(label_atoms)
    n_valid = n_expected - missing
    missing_fraction = 0.0 if n_expected == 0 else missing / n_expected

    energy_mae = float(np.mean(energy_errors)) if energy_errors else None
    force_mae = float(np.mean(np.concatenate(force_errors))) if force_errors else None
    stress_mae = float(np.mean(np.concatenate(stress_errors))) if stress_errors else None

    violations: list[str] = []
    if missing_fraction > max_missing_fraction:
        violations.append("excessive_missing_predictions")
    if force_mae is not None and force_mae > force_explosion_gate:
        violations.append("force_mae_exploded")

    return MetricResult(
        energy_mae_per_atom=energy_mae,
        force_mae=force_mae,
        stress_mae=stress_mae,
        missing_predictions=missing,
        n_expected=n_expected,
        n_valid=n_valid,
        hard_gate_violations=violations,
    )
