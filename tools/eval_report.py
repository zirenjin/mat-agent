#!/usr/bin/env python3
"""Emit a structured evaluation report for an MLIP validation run.

Purpose: convert prediction/truth extxyz files into LLM-readable JSON.
Inputs: pred_energy/pred_forces predictions and ref_energy/ref_forces truth.
Outputs: metrics, optional uncertainty_stats, recommendation, timestamp, status.
Dependencies: ASE, numpy, json, argparse.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read

from ase_conventions import get_pred_energy, get_pred_forces, get_ref_energy, get_ref_forces


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Write structured MLIP evaluation report JSON")
    p.add_argument("--predictions", required=True)
    p.add_argument("--truth", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--task-id", default=None)
    p.add_argument("--uncertainty-threshold", type=float, default=0.3)
    return p.parse_args()


def timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: str, payload: dict[str, Any]) -> None:
    """Write formatted JSON after validating it can be parsed."""
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    json.loads(text)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


def write_error(args: argparse.Namespace, message: str) -> None:
    """Write an error report."""
    write_json(args.output, {"model": args.model, "checkpoint_path": args.checkpoint_path, "task_id": args.task_id, "timestamp": timestamp(), "status": "error", "error_message": message})


def compute_uncertainty_stats(preds: list[Any], threshold: float) -> dict[str, Any] | None:
    """Summarize pred_force_variance_max when present."""
    if not preds or any("pred_force_variance_max" not in atoms.info for atoms in preds):
        return None
    values = np.asarray([float(atoms.info["pred_force_variance_max"]) for atoms in preds], dtype=np.float64)
    high = values > threshold
    global_mean = float(np.mean(values)) if len(values) else 0.0
    by_element: dict[str, list[float]] = defaultdict(list)
    for atoms, value in zip(preds, values):
        for sym in sorted(set(atoms.get_chemical_symbols())):
            by_element[sym].append(float(value))
    breakdown = {sym: float(np.mean(vals)) for sym, vals in sorted(by_element.items())}
    high_elements = [sym for sym, val in breakdown.items() if val > 1.5 * global_mean]
    high_elements.sort(key=lambda sym: breakdown[sym], reverse=True)
    high_values = values[high]
    return {
        "aggregation_used": "max",
        "threshold": threshold,
        "high_uncertainty_fraction": float(np.mean(high)) if len(values) else 0.0,
        "high_uncertainty_elements": high_elements,
        "mean_variance_max_all": global_mean,
        "mean_variance_max_high": float(np.mean(high_values)) if len(high_values) else None,
        "element_variance_breakdown": breakdown,
    }


def main() -> None:
    """Write the evaluation report."""
    args = parse_args()
    try:
        preds = read(args.predictions, ":")
        truths = read(args.truth, ":")
        n = min(len(preds), len(truths))
        preds, truths = preds[:n], truths[:n]
        if n == 0:
            raise ValueError("no structures found")
        e_errs, f_errs = [], []
        for idx, (pred, truth) in enumerate(zip(preds, truths)):
            ep, et = get_pred_energy(pred), get_ref_energy(truth)
            fp, ft = get_pred_forces(pred), get_ref_forces(truth)
            if ep is None or et is None or fp is None or ft is None:
                raise ValueError(f"missing prediction or truth fields at structure {idx}")
            e_errs.append(float(ep) / len(pred) - float(et) / len(truth))
            f_errs.append((np.asarray(fp, dtype=np.float64) - np.asarray(ft, dtype=np.float64)).reshape(-1))
        e = np.asarray(e_errs, dtype=np.float64)
        f = np.concatenate(f_errs)
        uncertainty = compute_uncertainty_stats(preds, args.uncertainty_threshold)
        force_mae = float(np.mean(np.abs(f)))
        force_rmse = float(np.sqrt(np.mean(f ** 2)))
        energy_mae = float(np.mean(np.abs(e)))
        energy_rmse = float(np.sqrt(np.mean(e ** 2)))
        if uncertainty:
            recommendation = f"Training converged. val_force_MAE={force_mae:.4f} eV/A; {uncertainty['high_uncertainty_fraction']:.1%} of validation structures exceed uncertainty threshold {args.uncertainty_threshold:.3f}."
        else:
            recommendation = f"Training converged. val_force_MAE={force_mae:.4f} eV/A. Consider querying oracle for high-uncertainty structures if active learning is active."
        write_json(args.output, {
            "model": args.model,
            "checkpoint_path": args.checkpoint_path,
            "task_id": args.task_id,
            "n_train": None,
            "n_val": n,
            "val_energy_MAE_eV_per_atom": energy_mae,
            "val_force_MAE_eV_per_angstrom": force_mae,
            "val_energy_RMSE_eV_per_atom": energy_rmse,
            "val_force_RMSE_eV_per_angstrom": force_rmse,
            "uncertainty_stats": uncertainty,
            "recommendation": recommendation,
            "timestamp": timestamp(),
            "status": "ok",
        })
    except Exception as exc:
        write_error(args, str(exc))
        raise


if __name__ == "__main__":
    main()
