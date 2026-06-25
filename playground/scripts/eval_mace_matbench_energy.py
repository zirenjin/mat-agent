#!/usr/bin/env python3
"""Evaluate a MACE potential on Matbench structure tasks via energy features.

This adapter keeps the MACE output head unchanged: the model predicts total
energy for each structure, and forces remain the gradient of that energy. For
Matbench scalar regression targets, we use energy/atom as a single feature and
fit a per-fold linear calibration on the Matbench train split before scoring the
held-out test split.

This is intentionally a baseline adapter, not a new MACE prediction head.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Path to a local MACE .model checkpoint")
    p.add_argument("--tasks", nargs="+", default=["matbench_mp_e_form"], help="Matbench task names")
    p.add_argument("--folds", nargs="+", type=int, default=[0], help="Matbench folds to evaluate")
    p.add_argument("--output", required=True, help="Output metrics JSON")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--max-train", type=int, default=0, help="Optional cap for calibration train rows")
    p.add_argument("--max-test", type=int, default=0, help="Optional cap for test rows")
    p.add_argument("--batch-size", type=int, default=1, help="Reserved for future batching; MACECalculator is per-structure here")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = Path(args.model)
    if not model.is_file():
        raise SystemExit(f"model not found: {model}")

    from matbench.bench import MatbenchBenchmark
    from mace.calculators import MACECalculator

    calc = MACECalculator(model_paths=str(model), device=args.device, default_dtype=args.dtype)
    mb = MatbenchBenchmark(autoload=False, subset=args.tasks)

    results: dict[str, object] = {
        "model": str(model),
        "adapter": "mace_energy_per_atom_linear_calibration",
        "notes": "MACE head is unchanged; per-fold y = a * energy_per_atom + b is fitted on Matbench train split.",
        "tasks": {},
    }

    for task in mb.tasks:
        task.load()
        task_name = task.dataset_name
        meta = task.metadata
        input_type = getattr(meta, "input_type", None)
        task_type = getattr(meta, "task_type", None)
        if input_type != "structure" or task_type != "regression":
            results["tasks"][task_name] = {
                "skipped": True,
                "reason": f"unsupported input_type/task_type: {input_type}/{task_type}",
            }
            continue

        task_out = {
            "metadata": {
                "input_type": input_type,
                "task_type": task_type,
                "target": getattr(meta, "target", None),
                "unit": getattr(meta, "unit", None),
            },
            "folds": {},
        }
        for fold in args.folds:
            train_inputs, train_outputs = task.get_train_and_val_data(fold)
            test_inputs, test_outputs = task.get_test_data(fold, include_target=True)

            train_items = list(train_inputs.items())
            test_items = list(test_inputs.items())
            if args.max_train and len(train_items) > args.max_train:
                train_items = train_items[: args.max_train]
            if args.max_test and len(test_items) > args.max_test:
                test_items = test_items[: args.max_test]

            x_train = _energy_features(calc, train_items, desc=f"{task_name}/fold{fold}/train")
            y_train = np.asarray([float(train_outputs.loc[idx]) for idx, _ in train_items], dtype=float)
            x_test = _energy_features(calc, test_items, desc=f"{task_name}/fold{fold}/test")
            y_test = np.asarray([float(test_outputs.loc[idx]) for idx, _ in test_items], dtype=float)

            a, b = _fit_linear(x_train, y_train)
            pred = a * x_test + b
            err = pred - y_test
            mae = float(np.mean(np.abs(err)))
            rmse = float(math.sqrt(np.mean(err * err)))
            task_out["folds"][str(fold)] = {
                "n_train": int(len(x_train)),
                "n_test": int(len(x_test)),
                "calibration": {"slope": float(a), "intercept": float(b)},
                "mae": mae,
                "rmse": rmse,
            }
            print(f"{task_name} fold={fold} n_train={len(x_train)} n_test={len(x_test)} MAE={mae:.6g} RMSE={rmse:.6g}", flush=True)

        results["tasks"][task_name] = task_out

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}", flush=True)


def _energy_features(calc, items: Iterable[tuple[object, object]], desc: str) -> np.ndarray:
    adaptor = AseAtomsAdaptor()
    feats: list[float] = []
    for _, structure in tqdm(list(items), desc=desc):
        atoms = adaptor.get_atoms(structure)
        atoms.calc = calc
        energy = float(atoms.get_potential_energy())
        feats.append(energy / max(len(atoms), 1))
        atoms.calc = None
    return np.asarray(feats, dtype=float)


def _fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) == 0:
        raise ValueError("empty calibration set")
    X = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1])


if __name__ == "__main__":
    main()
