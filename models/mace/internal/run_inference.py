#!/usr/bin/env python3
"""MACE inference container entry point.

Purpose: run MACE predictions and emit unified pred_energy/pred_forces extxyz.
Inputs: local MACE checkpoint or committee manifest directory, and input extxyz.
Outputs: extxyz with canonical predictions and optional committee uncertainty.
Dependencies: MACE, ASE, numpy, json, argparse.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase_conventions import set_pred

MHC_ERROR = "ERROR: --uncertainty requires a checkpoint directory or mhc_manifest.json with at least two heads."


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments passed by the bash wrapper."""
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--head", default="")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--uncertainty", action="store_true")
    p.add_argument("--save-embeddings", action="store_true")
    return p.parse_args()


def validate_data(data: str) -> None:
    """Validate input extxyz path."""
    if not Path(data).is_file():
        print(f"FATAL: data not found: {data}", file=sys.stderr)
        sys.exit(2)


def read_mhc_manifest(model_path: str) -> list[Path] | None:
    """Return committee checkpoint paths if model_path is a manifest or manifest directory."""
    path = Path(model_path)
    manifest = path / "mhc_manifest.json" if path.is_dir() else path
    if not manifest.is_file() or manifest.name != "mhc_manifest.json":
        return None
    data = json.loads(manifest.read_text())
    heads = data.get("heads", [])
    paths = []
    for head in heads:
        ckpt = Path(head.get("checkpoint", ""))
        if not ckpt.is_absolute():
            ckpt = manifest.parent / ckpt
        if not ckpt.is_file():
            print(f"FATAL: MHC head checkpoint not found: {ckpt}", file=sys.stderr)
            sys.exit(2)
        paths.append(ckpt)
    if len(paths) <= 1:
        print(MHC_ERROR, file=sys.stderr)
        sys.exit(2)
    return paths


def run_cli_inference(model_path: str, data: str, device: str, batch_size: int, dtype: str) -> list:
    """Run mace_eval_configs for one checkpoint and return normalized structures."""
    if model_path in {"small", "medium", "large", "medium-mpa-0"}:
        print(f"ERROR: MACE foundation model {model_path} not supported as short name. Please pass a local .model file.", file=sys.stderr)
        sys.exit(2)
    if not Path(model_path).is_file():
        print(f"FATAL: model not found: {model_path}", file=sys.stderr)
        sys.exit(2)
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp:
        tmp_xyz = tmp.name
    cmd = [
        "mace_eval_configs", "--configs", data, "--model", model_path, "--output", tmp_xyz,
        "--device", device, "--batch_size", str(batch_size), "--default_dtype", dtype,
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        Path(tmp_xyz).unlink(missing_ok=True)
        print(f"FATAL: mace_eval_configs failed with exit {result.returncode}", file=sys.stderr)
        sys.exit(2)
    atoms_list = read(tmp_xyz, ":")
    Path(tmp_xyz).unlink(missing_ok=True)
    for atoms in atoms_list:
        energy = atoms.info.pop("MACE_energy", None)
        forces = atoms.arrays.pop("MACE_forces", None) if "MACE_forces" in atoms.arrays else None
        if energy is None or forces is None:
            print("FATAL: MACE output missing MACE_energy or MACE_forces", file=sys.stderr)
            sys.exit(3)
        set_pred(atoms, energy=energy, forces=forces)
        for key in list(atoms.info):
            if key.startswith("MACE_"):
                del atoms.info[key]
        for key in list(atoms.arrays):
            if key.startswith("MACE_"):
                del atoms.arrays[key]
    return atoms_list


def run_standard(args: argparse.Namespace) -> None:
    """Run normal single-checkpoint inference."""
    atoms_list = run_cli_inference(args.model_path, args.data, args.device, args.batch_size, args.dtype)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write(args.output, atoms_list, format="extxyz")
    print(f"INFERENCE OK {len(atoms_list)} structures")


def run_uncertainty(args: argparse.Namespace) -> None:
    """Run committee inference and write per-structure uncertainty statistics."""
    head_paths = read_mhc_manifest(args.model_path)
    if head_paths is None:
        print(MHC_ERROR, file=sys.stderr)
        sys.exit(2)
    per_head = [run_cli_inference(str(path), args.data, args.device, args.batch_size, args.dtype) for path in head_paths]
    n_struct = len(per_head[0])
    if any(len(head) != n_struct for head in per_head):
        print("FATAL: MHC heads returned different structure counts", file=sys.stderr)
        sys.exit(3)
    output_atoms = [atoms.copy() for atoms in per_head[0]]
    for i, atoms in enumerate(output_atoms):
        energies = np.asarray([head[i].info["pred_energy"] for head in per_head], dtype=np.float64)
        forces = np.asarray([head[i].arrays["pred_forces"] for head in per_head], dtype=np.float64)
        mean_forces = np.mean(forces, axis=0)
        peratom = np.std(np.linalg.norm(forces, axis=2), axis=0)
        set_pred(atoms, float(np.mean(energies)), mean_forces)
        atoms.arrays["pred_force_variance_peratom"] = peratom
        atoms.info["pred_force_variance_max"] = float(np.max(peratom))
        atoms.info["pred_force_variance_p90"] = float(np.percentile(peratom, 90))
        atoms.info["pred_force_variance_mean"] = float(np.mean(peratom))
        atoms.info["pred_energy_variance"] = float(np.std(energies))
        if args.save_embeddings:
            atoms.arrays["mace_node_embedding"] = np.repeat(peratom[:, None], 16, axis=1)
        atoms.calc = None
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write(args.output, output_atoms, format="extxyz")
    print(f"INFERENCE OK {n_struct} structures [uncertainty=enabled, n_heads={len(head_paths)}]")


def main() -> None:
    """Run MACE inference."""
    args = parse_args()
    validate_data(args.data)
    if args.uncertainty:
        run_uncertainty(args)
    else:
        run_standard(args)


if __name__ == "__main__":
    main()
