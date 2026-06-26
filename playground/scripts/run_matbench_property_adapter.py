#!/usr/bin/env python3
"""Run Matbench property adapters over structure and MLIP-derived features.

Purpose: evaluate whether multi-model MLIP features improve Matbench regression
without fine-tuning the force field itself.
Inputs: Matbench structure tasks, one or more MACE model checkpoints.
Outputs: JSON metrics with per-fold model selection and feature ablations.
Dependencies: matbench, pymatgen, mace, ASE, numpy, sklearn, tqdm.
"""
from __future__ import annotations

print("[adapter] process started; importing dependencies", flush=True)
import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pymatgen.core import Element
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

print("[adapter] dependencies imported", flush=True)


@dataclass(frozen=True)
class MlipModel:
    label: str
    path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True, help="MACE models as label=/path/model entries")
    p.add_argument("--tasks", nargs="+", default=["matbench_mp_e_form", "matbench_jdft2d", "matbench_phonons"])
    p.add_argument("--folds", nargs="+", type=int, default=[0])
    p.add_argument("--output", required=True)
    p.add_argument("--cache-dir", default="playground/cache/matbench_property_adapter")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument(
        "--feature-sets",
        nargs="+",
        default=["structure", "chem", "structure+chem", "structure+mlip_aug", "structure+chem+mlip_aug", "mlip_aug"],
    )
    return p.parse_args()


def parse_models(entries: list[str]) -> list[MlipModel]:
    out = []
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"model entry must be label=/path/model: {entry}")
        label, path = entry.split("=", 1)
        p = Path(path)
        if not p.is_file():
            raise SystemExit(f"model not found for {label}: {p}")
        out.append(MlipModel(label=label, path=p))
    return out


def json_write(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    json.loads(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def model_fingerprint(model: MlipModel) -> str:
    st = model.path.stat()
    raw = f"{model.label}:{model.path}:{st.st_size}:{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def ids_fingerprint(items: list[tuple[object, object]]) -> str:
    """Fingerprint a split/cap by Matbench ids to avoid stale feature caches."""
    raw = "\n".join(str(idx) for idx, _ in items)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def structure_features(structures: list[Any], cache_path: Path) -> np.ndarray:
    if cache_path.is_file():
        print(f"[adapter] cache hit {cache_path}", flush=True)
        return np.load(cache_path)["x"]
    rows = []
    for structure in tqdm(structures, desc=f"features/{cache_path.stem}"):
        rows.append(_one_structure_features(structure))
    x = np.asarray(rows, dtype=np.float64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=x)
    return x



def chem_features(structures: list[Any], cache_path: Path) -> np.ndarray:
    if cache_path.is_file():
        print(f"[adapter] cache hit {cache_path}", flush=True)
        return np.load(cache_path)["x"]
    from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital
    featurizers = [Stoichiometry(), ElementProperty.from_preset("magpie"), ValenceOrbital()]
    rows = []
    for structure in tqdm(structures, desc=f"features/{cache_path.stem}"):
        comp = structure.composition
        vals: list[float] = []
        for featurizer in featurizers:
            vals.extend(float(v) for v in featurizer.featurize(comp))
        rows.append(vals)
    x = np.asarray(rows, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=x)
    return x

def _one_structure_features(structure: Any) -> list[float]:
    comp = structure.composition
    elements = sorted(comp.elements, key=lambda e: e.Z)
    total_atoms = float(comp.num_atoms)
    frac = np.zeros(94, dtype=np.float64)
    zs, xs, masses, rows, groups = [], [], [], [], []
    for el in elements:
        if 1 <= el.Z <= 94:
            frac[el.Z - 1] = float(comp[el]) / total_atoms
        zs.append(float(el.Z))
        try:
            xs.append(float(el.X) if el.X is not None else 0.0)
        except Exception:
            xs.append(0.0)
        masses.append(float(el.atomic_mass))
        rows.append(float(el.row or 0))
        groups.append(float(el.group or 0))
    def stats(vals: list[float]) -> list[float]:
        arr = np.asarray(vals, dtype=np.float64)
        return [float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))]
    lat = structure.lattice
    base = [
        total_atoms,
        float(len(elements)),
        float(lat.volume) / max(total_atoms, 1.0),
        float(structure.density),
        float(lat.a), float(lat.b), float(lat.c),
        float(lat.alpha), float(lat.beta), float(lat.gamma),
    ]
    base.extend(stats(zs)); base.extend(stats(xs)); base.extend(stats(masses)); base.extend(stats(rows)); base.extend(stats(groups))
    return base + frac.tolist()


def mlip_features(models: list[MlipModel], structures: list[Any], cache_dir: Path, task: str, fold: int, split: str, device: str, dtype: str) -> np.ndarray:
    if not models:
        return np.zeros((len(structures), 0), dtype=np.float64)
    blocks = []
    for model in models:
        cache = cache_dir / task / f"fold_{fold}" / f"{split}_{model.label}_{model_fingerprint(model)}_{device}_{dtype}.npz"
        if cache.is_file():
            print(f"[adapter] cache hit {cache}", flush=True)
            blocks.append(np.load(cache)["x"])
            continue
        blocks.append(_compute_mace_block(model, structures, cache, device, dtype))
    return np.concatenate(blocks, axis=1)


def _compute_mace_block(model: MlipModel, structures: list[Any], cache: Path, device: str, dtype: str) -> np.ndarray:
    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=str(model.path), device=device, default_dtype=dtype)
    adaptor = AseAtomsAdaptor()
    rows = []
    for structure in tqdm(structures, desc=f"mace/{model.label}/{cache.parent.name}/{cache.stem}"):
        atoms = adaptor.get_atoms(structure)
        atoms.calc = calc
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        norms = np.linalg.norm(forces, axis=1)
        rows.append([
            energy / max(len(atoms), 1),
            energy,
            float(np.mean(norms)),
            float(np.std(norms)),
            float(np.max(norms)),
            float(np.percentile(norms, 90)),
            float(np.sqrt(np.mean(forces.reshape(-1) ** 2))),
        ])
        atoms.calc = None
    x = np.asarray(rows, dtype=np.float64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=x)
    return x



def augment_mlip_features(x: np.ndarray, n_models: int) -> np.ndarray:
    """Add committee-style cross-model agreement features to per-model MLIP blocks."""
    block = 7
    if n_models < 2 or x.shape[1] != n_models * block:
        return x
    cube = x.reshape(x.shape[0], n_models, block)
    summary_cols = []
    # Per-atom energy and force statistics are the useful cross-model signals;
    # total energy is size-coupled and already available in the raw block.
    for col in [0, 2, 3, 4, 5, 6]:
        vals = cube[:, :, col]
        summary_cols.extend([
            np.mean(vals, axis=1),
            np.std(vals, axis=1),
            np.min(vals, axis=1),
            np.max(vals, axis=1),
            np.max(vals, axis=1) - np.min(vals, axis=1),
        ])
    for col in [0, 6]:
        vals = cube[:, :, col]
        for i in range(n_models):
            for j in range(i + 1, n_models):
                summary_cols.append(vals[:, i] - vals[:, j])
                summary_cols.append(np.abs(vals[:, i] - vals[:, j]))
    return np.concatenate([x, np.vstack(summary_cols).T], axis=1)

def build_regressors(seed: int) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-6, 6, 25))),
        "extra_trees": ExtraTreesRegressor(n_estimators=500, random_state=seed, min_samples_leaf=1, n_jobs=-1),
        "random_forest": RandomForestRegressor(n_estimators=400, random_state=seed, min_samples_leaf=2, n_jobs=-1),
        "hist_gbdt": HistGradientBoostingRegressor(random_state=seed, max_iter=600, learning_rate=0.04, l2_regularization=1e-4),
    }


def split_train_val(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_fraction))) if n > 5 else max(1, n // 3)
    return idx[n_val:], idx[:n_val]


def score(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - y
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(math.sqrt(np.mean(err * err)))}


def fit_select_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int, val_fraction: float) -> tuple[np.ndarray, dict[str, Any]]:
    tr_idx, val_idx = split_train_val(len(y_train), val_fraction, seed)
    regressors = build_regressors(seed)
    val_scores = {}
    fitted_for_test = {}
    val_preds = []
    names = []
    for name, reg in regressors.items():
        reg.fit(x_train[tr_idx], y_train[tr_idx])
        pred_val = np.asarray(reg.predict(x_train[val_idx]), dtype=np.float64)
        val_scores[name] = score(y_train[val_idx], pred_val)
        names.append(name)
        val_preds.append(pred_val)
    if len(val_preds) >= 2:
        ens_val = np.mean(np.vstack(val_preds), axis=0)
        val_scores["mean_ensemble"] = score(y_train[val_idx], ens_val)
        names.append("mean_ensemble")
    best = min(val_scores, key=lambda k: val_scores[k]["mae"])
    if best == "mean_ensemble":
        preds = []
        for name, reg in build_regressors(seed).items():
            reg.fit(x_train, y_train)
            preds.append(np.asarray(reg.predict(x_test), dtype=np.float64))
        pred_test = np.mean(np.vstack(preds), axis=0)
    else:
        reg = build_regressors(seed)[best]
        reg.fit(x_train, y_train)
        pred_test = np.asarray(reg.predict(x_test), dtype=np.float64)
    return pred_test, {"best_model": best, "validation": val_scores, "n_inner_train": int(len(tr_idx)), "n_inner_val": int(len(val_idx))}


def main() -> None:
    args = parse_args()
    models = parse_models(args.models)
    cache_dir = Path(args.cache_dir)
    print("[adapter] importing MatbenchBenchmark", flush=True)
    from matbench.bench import MatbenchBenchmark
    print("[adapter] constructing MatbenchBenchmark", flush=True)
    mb = MatbenchBenchmark(autoload=False, subset=args.tasks)
    out: dict[str, Any] = {
        "adapter": "matbench_property_adapter_v1",
        "models": [{"label": m.label, "path": str(m.path)} for m in models],
        "feature_sets": args.feature_sets,
        "tasks": {},
    }
    for task in mb.tasks:
        print(f"[adapter] loading task {task.dataset_name}", flush=True)
        task.load()
        task_name = task.dataset_name
        print(f"[adapter] loaded task {task_name}", flush=True)
        meta = task.metadata
        if getattr(meta, "input_type", None) != "structure" or getattr(meta, "task_type", None) != "regression":
            out["tasks"][task_name] = {"skipped": True, "reason": "not a structure regression task"}
            continue
        task_out: dict[str, Any] = {
            "metadata": {"target": getattr(meta, "target", None), "unit": getattr(meta, "unit", None), "mad": getattr(meta, "mad", None)},
            "folds": {},
        }
        for fold in args.folds:
            train_inputs, train_outputs = task.get_train_and_val_data(fold)
            test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
            train_items = list(train_inputs.items())
            test_items = list(test_inputs.items())
            if args.max_train and len(train_items) > args.max_train:
                train_items = train_items[:args.max_train]
            if args.max_test and len(test_items) > args.max_test:
                test_items = test_items[:args.max_test]
            train_sig = ids_fingerprint(train_items)
            test_sig = ids_fingerprint(test_items)
            print(f"[adapter] {task_name} fold={fold} n_train={len(train_items)} n_test={len(test_items)} train_sig={train_sig} test_sig={test_sig}", flush=True)
            train_structures = [s for _, s in train_items]
            test_structures = [s for _, s in test_items]
            y_train = np.asarray([float(train_outputs.loc[idx]) for idx, _ in train_items], dtype=np.float64)
            y_test = np.asarray([float(test_outputs.loc[idx]) for idx, _ in test_items], dtype=np.float64)
            struct_train = structure_features(train_structures, cache_dir/task_name/f"fold_{fold}"/f"train_structure_{train_sig}.npz")
            struct_test = structure_features(test_structures, cache_dir/task_name/f"fold_{fold}"/f"test_structure_{test_sig}.npz")
            chem_train = chem_features(train_structures, cache_dir/task_name/f"fold_{fold}"/f"train_chem_{train_sig}.npz")
            chem_test = chem_features(test_structures, cache_dir/task_name/f"fold_{fold}"/f"test_chem_{test_sig}.npz")
            mlip_train = mlip_features(models, train_structures, cache_dir, task_name, fold, f"train_{train_sig}", args.device, args.dtype)
            mlip_test = mlip_features(models, test_structures, cache_dir, task_name, fold, f"test_{test_sig}", args.device, args.dtype)
            mlip_aug_train = augment_mlip_features(mlip_train, len(models))
            mlip_aug_test = augment_mlip_features(mlip_test, len(models))
            structure_chem_train = np.concatenate([struct_train, chem_train], axis=1)
            structure_chem_test = np.concatenate([struct_test, chem_test], axis=1)
            feature_blocks = {
                "structure": (struct_train, struct_test),
                "chem": (chem_train, chem_test),
                "structure+chem": (structure_chem_train, structure_chem_test),
                "mlip": (mlip_train, mlip_test),
                "mlip_aug": (mlip_aug_train, mlip_aug_test),
                "structure+mlip": (np.concatenate([struct_train, mlip_train], axis=1), np.concatenate([struct_test, mlip_test], axis=1)),
                "structure+mlip_aug": (np.concatenate([struct_train, mlip_aug_train], axis=1), np.concatenate([struct_test, mlip_aug_test], axis=1)),
                "chem+mlip_aug": (np.concatenate([chem_train, mlip_aug_train], axis=1), np.concatenate([chem_test, mlip_aug_test], axis=1)),
                "structure+chem+mlip_aug": (np.concatenate([structure_chem_train, mlip_aug_train], axis=1), np.concatenate([structure_chem_test, mlip_aug_test], axis=1)),
            }
            fold_out: dict[str, Any] = {"n_train": len(y_train), "n_test": len(y_test), "feature_sets": {}}
            best_name = None
            best_val_mae = float("inf")
            best_test_name = None
            best_test_mae = float("inf")
            for fs in args.feature_sets:
                xtr, xte = feature_blocks[fs]
                pred, selection = fit_select_predict(xtr, y_train, xte, args.seed + fold, args.val_fraction)
                metrics = score(y_test, pred)
                metrics.update(selection)
                metrics["n_features"] = int(xtr.shape[1])
                val_mae = float(selection["validation"][selection["best_model"]]["mae"])
                metrics["selection_mae"] = val_mae
                fold_out["feature_sets"][fs] = metrics
                print(
                    f"{task_name} fold={fold} fs={fs} best={selection['best_model']} "
                    f"val_MAE={val_mae:.6g} test_MAE={metrics['mae']:.6g} test_RMSE={metrics['rmse']:.6g}",
                    flush=True,
                )
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_name = fs
                if metrics["mae"] < best_test_mae:
                    best_test_mae = metrics["mae"]
                    best_test_name = fs
            fold_out["selected_feature_set"] = best_name
            fold_out["selected_feature_set_selection_mae"] = best_val_mae
            fold_out["best_feature_set_by_test_diagnostic"] = best_test_name
            task_out["folds"][str(fold)] = fold_out
            json_write(Path(args.output), out | {"tasks": {**out["tasks"], task_name: task_out}})
        out["tasks"][task_name] = task_out
    json_write(Path(args.output), out)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
