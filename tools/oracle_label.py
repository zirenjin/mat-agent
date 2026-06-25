#!/usr/bin/env python3
"""Label structures by calling a configured MLIP as a pseudo-oracle.

Purpose: convert model predictions into traceable ref_energy/ref_forces labels.
Inputs: unlabeled extxyz, oracle model name, local checkpoint path.
Outputs: clean labeled extxyz with oracle provenance in atoms.info.
Dependencies: ASE, numpy, json, argparse, subprocess, pathlib, datetime.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.io import read, write

from ase_conventions import get_pred_energy, get_pred_forces, set_ref

MODEL_MAP = {"mace": "mace", "dpa": "deepmd", "deepmd": "deepmd", "uma": "fairchem", "fairchem": "fairchem"}
DISPLAY = {"mace": "MACE-MP-0", "dpa": "DPA", "deepmd": "DPA", "uma": "UMA-s-1p1", "fairchem": "UMA-s-1p1"}


def parse_args() -> argparse.Namespace:
    """Parse oracle-labeling arguments."""
    p = argparse.ArgumentParser(description="Label extxyz structures with a pseudo-oracle MLIP")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--oracle", choices=sorted(MODEL_MAP), default="uma")
    p.add_argument("--model", default="", help="Oracle checkpoint path")
    p.add_argument("--model-path", default="", help="Alias for --model for tool-level compatibility")
    p.add_argument("--head", default="")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> None:
    """Run oracle inference and write ref_* labels."""
    args = parse_args()
    model_path = args.model or args.model_path
    if not model_path:
        print("ERROR: --model required", file=sys.stderr)
        sys.exit(1)
    root = Path(__file__).resolve().parents[1]
    api = root / "models" / MODEL_MAP[args.oracle] / "api" / "inference.sh"
    if not api.is_file():
        print(f"FATAL: oracle API not found: {api}", file=sys.stderr)
        sys.exit(2)
    with tempfile.TemporaryDirectory(prefix="mat-agent-oracle-") as tmp:
        pred_path = Path(tmp) / "predictions.extxyz"
        cmd = ["bash", str(api), "--model", model_path, "--data", args.input, "--output", str(pred_path), "--device", args.device, "--batch-size", str(args.batch_size)]
        if args.head:
            cmd.extend(["--head", args.head])
        subprocess.run(cmd, check=True)
        atoms_list = read(pred_path, ":")
    label = DISPLAY[args.oracle]
    ts = timestamp()
    for idx, atoms in enumerate(atoms_list):
        energy = get_pred_energy(atoms)
        forces = get_pred_forces(atoms)
        if energy is None or forces is None:
            raise ValueError(f"oracle prediction missing pred_energy/pred_forces at structure {idx}")
        set_ref(atoms, float(energy), np.asarray(forces, dtype=np.float64))
        atoms.info["oracle_model"] = label
        atoms.info["oracle_timestamp"] = ts
        for key in list(atoms.info):
            if key == "pred_energy" or key.startswith("pred_force_variance_") or key == "pred_energy_variance":
                del atoms.info[key]
        for key in list(atoms.arrays):
            if key == "pred_forces" or key.startswith("pred_force_variance_") or key == "mace_node_embedding":
                del atoms.arrays[key]
        atoms.calc = None
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write(args.output, atoms_list, format="extxyz")
    print(f"ORACLE OK {len(atoms_list)} structures labeled by {args.oracle}")


if __name__ == "__main__":
    main()
