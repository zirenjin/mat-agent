#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
from ase import Atoms
from ase.optimize import FIRE
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

MAT_ID = "material_id"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run MACE WBM IS2RE-SR relaxation with official MBD-compatible outputs."
    )
    p.add_argument("--model-name", required=True)
    p.add_argument("--model", required=True, help="Local checkpoint path or MACE foundation URL/name")
    p.add_argument("--atoms-zip", default="/work/scoring/matbench-discovery/data/wbm/2024-08-04-wbm-initial-atoms.extxyz.zip")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--n-chunks", type=int, default=200)
    p.add_argument("--chunk-id", type=int, required=True, help="1-based chunk id")
    p.add_argument("--limit", type=int, default=0, help="Debug only: read at most this many zip members before chunking")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def as_dict(obj: object) -> Any:
    fn = getattr(obj, "as_dict", None)
    return fn() if callable(fn) else obj


def frechet_cell_filter_cls():
    try:
        from ase.filters import FrechetCellFilter

        return FrechetCellFilter
    except Exception:
        from ase.constraints import FrechetCellFilter

        return FrechetCellFilter


def read_atoms_zip(path: str | Path, limit: int = 0) -> list[Atoms]:
    atoms_list: list[Atoms] = []
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if limit:
            names = names[:limit]
        for idx, name in tqdm(
            enumerate(names),
            desc=f"Reading ASE Atoms from zip_filename={str(path)!r}",
            mininterval=5,
        ):
            if not name.endswith(".extxyz"):
                continue
            with zf.open(name) as file:
                content = io.TextIOWrapper(file, encoding="utf-8").read()
            atoms = ase.io.read(io.StringIO(content), format="extxyz", index=slice(None))
            frames = [atoms] if isinstance(atoms, Atoms) else list(atoms)
            for frame in frames:
                frame.info[MAT_ID] = frame.info.get(MAT_ID) or frame.info.get("mat_id") or Path(name).stem
                atoms_list.append(frame)
    return atoms_list


def chunk_by_lens(inputs: list[Atoms], n_chunks: int) -> list[list[Atoms]]:
    if not inputs:
        return []
    n_chunks = min(max(1, n_chunks), len(inputs))
    lens = np.array([len(obj) for obj in inputs])
    sorted_inputs = [inputs[i] for i in np.argsort(lens)[::-1]]
    chunks: list[list[Atoms]] = [[] for _ in range(n_chunks)]
    chunk_sizes = np.zeros(n_chunks, dtype=float)
    for obj in sorted_inputs:
        idx = int(np.argmin(chunk_sizes))
        chunks[idx].append(obj)
        chunk_sizes[idx] += len(obj)
    print(
        f"Split {len(inputs):,} structures into {n_chunks:,} chunks: "
        f"mean atoms/chunk={chunk_sizes.mean():.1f}, min={chunk_sizes.min():.0f}, max={chunk_sizes.max():.0f}",
        flush=True,
    )
    return chunks


def make_calc(model: str, device: str, dtype: str):
    try:
        from mace.calculators import mace_mp

        return mace_mp(model=model, device=device, default_dtype=dtype, enable_cueq=device == "cuda")
    except Exception as exc:
        print(f"mace_mp load failed ({exc!r}); falling back to MACECalculator", flush=True)
        from mace.calculators import MACECalculator

        return MACECalculator(model_paths=model, device=device, default_dtype=dtype)


def relax_one(atoms0: Atoms, calc, fmax: float, max_steps: int) -> dict[str, Any]:
    atoms = deepcopy(atoms0)
    atoms.calc = calc
    mat_id = str(atoms.info[MAT_ID])
    try:
        filtered_atoms = frechet_cell_filter_cls()(atoms)
        optimizer = FIRE(filtered_atoms, logfile=None)
        converged = bool(optimizer.run(fmax=fmax, steps=max_steps))
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        max_force = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0
        structure: Structure = AseAtomsAdaptor.get_structure(atoms)
        return {
            MAT_ID: mat_id,
            "mace_structure": structure.as_dict(),
            "mace_energy": energy,
            "max_force": max_force,
            "n_steps": int(getattr(optimizer, "nsteps", -1)),
            "converged": converged,
            "error": "",
        }
    except Exception as exc:
        return {
            MAT_ID: mat_id,
            "mace_structure": None,
            "mace_energy": math.nan,
            "max_force": math.nan,
            "n_steps": -1,
            "converged": False,
            "error": repr(exc),
        }


def main() -> None:
    args = parse_args()
    if args.chunk_id < 1 or args.chunk_id > args.n_chunks:
        raise SystemExit(f"--chunk-id must be in 1..{args.n_chunks}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model_name}-wbm-IS2RE-FIRE-{args.chunk_id:03d}-of-{args.n_chunks:03d}.json.gz"
    if out_path.is_file() and not args.overwrite:
        raise SystemExit(f"{out_path} exists; use --overwrite to rerun")

    atoms_list = read_atoms_zip(args.atoms_zip, args.limit)
    chunk = chunk_by_lens(atoms_list, args.n_chunks)[args.chunk_id - 1]
    print(f"Running {args.model_name} chunk {args.chunk_id}/{args.n_chunks}: {len(chunk):,} structures", flush=True)
    calc = make_calc(args.model, args.device, args.dtype)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with gzip.open(tmp_path, "wt", encoding="utf-8") as file:
        for row in tqdm(chunk, desc=f"{args.model_name} relax", mininterval=5, unit="struct"):
            file.write(json.dumps(relax_one(row, calc, args.fmax, args.max_steps), default=as_dict) + "\n")
            file.flush()
    os.replace(tmp_path, out_path)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
