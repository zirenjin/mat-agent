#!/usr/bin/env python3
"""Run Matbench Discovery stability F1 for one or more MACE checkpoints.

This is meant as a lightweight first pass for the current experiment: evaluate
individual frozen MACE models as discovery scorers before spending GPU time on
full WBM relaxations or adapter-style committees.

The script writes one compressed CSV of WBM formation-energy predictions per
model plus a JSON metrics summary. By default it evaluates single-point energies
on WBM initial structures (IS2E). Use --structures relaxed to score the DFT
relaxed WBM structures instead. Full official UIP submissions usually relax WBM
initial structures before evaluating energies (IS2RE); that is intentionally not
the default here because it is much more expensive.
"""

from __future__ import annotations

# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "ase>=3.23",
#   "mace-torch",
#   "matbench-discovery>=1.3.1",
#   "monty",
#   "numpy",
#   "pandas",
#   "pymatgen",
#   "scikit-learn",
#   "tqdm",
# ]
# ///

import argparse
import gzip
import hashlib
import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
import pandas as pd
from ase import Atoms
from pymatviz.enums import Key
from tqdm import tqdm


@dataclass(frozen=True)
class ModelSpec:
    label: str
    spec: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help=(
            "MACE models as label=/path/to/model.model entries. Built-in MACE-MP "
            "can be used as label=mace_mp:small|medium|large."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="playground/cache/matbench_discovery_mace_f1")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--structures", default="initial", choices=["initial", "relaxed"])
    parser.add_argument("--max-samples", type=int, default=0, help="Debug cap on WBM structures")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--element-refs-json",
        default="",
        help=(
            "Optional JSON mapping element symbols to reference energies in eV/atom. "
            "If omitted, MP elemental references bundled with matbench-discovery are used."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute model energy caches even if present.",
    )
    return parser.parse_args()


def parse_models(entries: list[str]) -> list[ModelSpec]:
    models: list[ModelSpec] = []
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"model entry must be label=/path or label=mace_mp:size: {entry}")
        label, spec = entry.split("=", 1)
        if not spec.startswith("mace_mp:") and not Path(spec).is_file():
            raise SystemExit(f"model file not found for {label}: {spec}")
        models.append(ModelSpec(label=label, spec=spec))
    return models


def model_fingerprint(model: ModelSpec) -> str:
    if model.spec.startswith("mace_mp:"):
        raw = model.spec
    else:
        path = Path(model.spec)
        st = path.stat()
        raw = f"{path.resolve()}:{st.st_size}:{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    json.loads(text)
    path.write_text(text)


def load_element_refs(path: str) -> dict[str, float]:
    if path:
        refs = json.loads(Path(path).read_text())
        return {str(key): float(val) for key, val in refs.items()}

    from monty.serialization import loadfn
    from matbench_discovery.enums import DataFiles

    entries = loadfn(DataFiles.mp_elemental_ref_entries.path)
    refs: dict[str, float] = {}
    for entry in entries:
        comp = entry.composition
        if len(comp.elements) != 1:
            continue
        symbol = comp.elements[0].symbol
        refs[symbol] = float(entry.energy_per_atom)
    if not refs:
        raise RuntimeError("No elemental reference entries loaded from matbench-discovery")
    return refs


def wbm_atoms_path(kind: str) -> str:
    from matbench_discovery.enums import DataFiles

    return (
        DataFiles.wbm_initial_atoms.path
        if kind == "initial"
        else DataFiles.wbm_relaxed_atoms.path
    )


def read_atoms_zip(path: str, *, limit: int = 0) -> dict[str, Atoms]:
    atoms_by_id: dict[str, Atoms] = {}
    with zipfile.ZipFile(path) as zip_file:
        names = [name for name in zip_file.namelist() if name.endswith(".extxyz")]
        if limit:
            names = names[:limit]
        for name in tqdm(names, desc=f"read/{Path(path).name}"):
            with zip_file.open(name) as file:
                text = gzip.open(file, mode="rt").read() if name.endswith(".gz") else file.read().decode()
            atoms_obj = ase.io.read(
                filename=io.StringIO(text),
                format="extxyz",
                index=-1,
            )
            if not isinstance(atoms_obj, Atoms):
                atoms_obj = atoms_obj[-1]
            mat_id = str(
                atoms_obj.info.get(str(Key.mat_id))
                or atoms_obj.info.get("material_id")
                or Path(name).stem
            )
            atoms_obj.info[str(Key.mat_id)] = mat_id
            atoms_by_id[mat_id] = atoms_obj
    return atoms_by_id


def make_calculator(model: ModelSpec, device: str, dtype: str) -> Any:
    if model.spec.startswith("mace_mp:"):
        from mace.calculators import mace_mp

        model_name = model.spec.split(":", 1)[1]
        return mace_mp(model_name=model_name, device=device, default_dtype=dtype)

    from mace.calculators import MACECalculator

    return MACECalculator(model_paths=model.spec, device=device, default_dtype=dtype)


def energy_cache_path(cache_dir: Path, model: ModelSpec, structures: str, device: str, dtype: str) -> Path:
    fp = model_fingerprint(model)
    safe_label = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model.label)
    return cache_dir / f"{safe_label}_{fp}_{structures}_{device}_{dtype}_energies.csv.gz"


def predict_total_energies(
    model: ModelSpec,
    atoms_by_id: dict[str, Atoms],
    cache_dir: Path,
    structures: str,
    device: str,
    dtype: str,
    overwrite: bool,
) -> pd.DataFrame:
    cache = energy_cache_path(cache_dir, model, structures, device, dtype)
    if cache.is_file() and not overwrite:
        print(f"[discovery-f1] cache hit {cache}", flush=True)
        return pd.read_csv(cache)

    calc = make_calculator(model, device=device, dtype=dtype)
    rows: list[dict[str, Any]] = []
    for mat_id, atoms in tqdm(atoms_by_id.items(), desc=f"mace/{model.label}"):
        atoms = atoms.copy()
        atoms.calc = calc
        try:
            energy = float(atoms.get_potential_energy())
            rows.append(
                {
                    "material_id": mat_id,
                    "energy": energy,
                    "n_atoms": len(atoms),
                    "formula": atoms.get_chemical_formula(),
                }
            )
        except Exception as exc:  # keep benchmark alignment; NaNs count as unstable
            rows.append(
                {
                    "material_id": mat_id,
                    "energy": math.nan,
                    "n_atoms": len(atoms),
                    "formula": atoms.get_chemical_formula(),
                    "error": repr(exc),
                }
            )
    df = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def formation_energy_per_atom(atoms: Atoms, total_energy: float, refs: dict[str, float]) -> float:
    if not np.isfinite(total_energy):
        return math.nan
    missing = sorted(set(atoms.symbols) - set(refs))
    if missing:
        raise KeyError(f"missing elemental refs for {missing}")
    ref_energy = sum(refs[symbol] for symbol in atoms.symbols)
    return (total_energy - ref_energy) / len(atoms)


def add_discovery_metrics(df_wbm: pd.DataFrame, pred: pd.Series) -> dict[str, Any]:
    from matbench_discovery.enums import MbdKey
    from matbench_discovery.metrics.discovery import stable_metrics

    df = df_wbm.copy()
    df["e_form_pred"] = pred
    df["each_pred"] = df[MbdKey.each_true] + df["e_form_pred"] - df[MbdKey.e_form_dft]

    subsets: dict[str, pd.DataFrame] = {
        "full_test_set": df,
        "unique_prototypes": df.query(MbdKey.uniq_proto),
        "most_stable_10k": df.nsmallest(min(10_000, len(df)), "each_pred"),
    }
    out = {}
    for name, subset_df in subsets.items():
        metrics = stable_metrics(subset_df[MbdKey.each_true], subset_df["each_pred"])
        out[name] = {key: float(val) if isinstance(val, np.floating) else val for key, val in metrics.items()}
        out[name]["n"] = int(len(subset_df))
    return out


def main() -> None:
    args = parse_args()
    models = parse_models(args.models)
    out_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from matbench_discovery.data import df_wbm

    refs = load_element_refs(args.element_refs_json)
    atoms_path = wbm_atoms_path(args.structures)
    atoms_by_id = read_atoms_zip(atoms_path, limit=args.max_samples)
    wbm = df_wbm.loc[list(atoms_by_id)].copy()

    summary: dict[str, Any] = {
        "task": "matbench_discovery",
        "structures": args.structures,
        "energy_to_e_form": "subtract elemental references from predicted total energy",
        "n_structures": int(len(wbm)),
        "models": {},
    }
    for model in models:
        df_energy = predict_total_energies(
            model,
            atoms_by_id,
            cache_dir,
            args.structures,
            args.device,
            args.dtype,
            args.overwrite,
        )
        energy_by_id = df_energy.set_index("material_id")["energy"]
        e_form_pred = pd.Series(index=wbm.index, dtype=float, name="e_form_per_atom_pred")
        for mat_id, atoms in atoms_by_id.items():
            e_form_pred.loc[mat_id] = formation_energy_per_atom(atoms, energy_by_id.loc[mat_id], refs)

        pred_path = out_dir / f"{model.label}-wbm-{args.structures}-IS2E.csv.gz"
        pd.DataFrame({"material_id": e_form_pred.index, "e_form_per_atom_pred": e_form_pred.values}).to_csv(
            pred_path, index=False
        )
        metrics = add_discovery_metrics(wbm, e_form_pred)
        summary["models"][model.label] = {
            "spec": model.spec,
            "pred_file": str(pred_path),
            "metrics": metrics,
        }
        json_write(out_dir / "matbench_discovery_mace_f1_summary.json", summary)
        print(
            f"[discovery-f1] {model.label}: "
            f"F1(full)={metrics['full_test_set']['F1']:.4f} "
            f"F1(unique)={metrics['unique_prototypes']['F1']:.4f} "
            f"F1(top10k)={metrics['most_stable_10k']['F1']:.4f}",
            flush=True,
        )

    json_write(out_dir / "matbench_discovery_mace_f1_summary.json", summary)
    print(f"[discovery-f1] wrote {out_dir / 'matbench_discovery_mace_f1_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
