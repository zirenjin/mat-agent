#!/usr/bin/env python3
"""MACE inference — container entry point.

Calls mace_eval_configs CLI, then renames MACE_energy / MACE_forces
to pred_energy / pred_forces to conform to unified_api_spec.md.
"""
from __future__ import annotations
import argparse, subprocess, sys, tempfile, os
from pathlib import Path
from ase.io import read, write
from ase_conventions import set_pred


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    # Reject foundation model short names — must be absolute path
    FOUNDATION_SHORTS = {"small", "medium", "large", "medium-mpa-0"}
    if args.model_path in FOUNDATION_SHORTS:
        print(
            f"ERROR: MACE foundation model {args.model_path} not supported as short name. "
            f"Please pass an absolute path to a local .model file. "
            f"Download logic lives on the host, not inside the container.",
            file=sys.stderr,
        )
        sys.exit(2)

    if not Path(args.model_path).is_file():
        print(f"FATAL: model not found: {args.model_path}", file=sys.stderr)
        sys.exit(2)
    if not Path(args.data).is_file():
        print(f"FATAL: data not found: {args.data}", file=sys.stderr)
        sys.exit(2)

    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp:
        tmp_xyz = tmp.name

    cmd = [
        "mace_eval_configs",
        "--configs", args.data,
        "--model", args.model_path,
        "--output", tmp_xyz,
        "--device", args.device,
        "--batch_size", str(args.batch_size),
        "--default_dtype", "float64",
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"FATAL: mace_eval_configs failed with exit {result.returncode}", file=sys.stderr)
        os.unlink(tmp_xyz)
        sys.exit(2)

    # Read back mace output, rename fields
    atoms_list = read(tmp_xyz, ":")
    n = len(atoms_list)
    print(f"Loaded {n} structures from mace output", file=sys.stderr)

    for atoms in atoms_list:
        e = atoms.info.pop("MACE_energy", None)
        f = atoms.arrays.pop("MACE_forces", None) if "MACE_forces" in atoms.arrays else None
        if e is None:
            print(f"WARN: structure missing MACE_energy", file=sys.stderr)
        set_pred(atoms, energy=e, forces=f)
        # Remove any remaining MACE_* keys
        for key in list(atoms.info.keys()):
            if key.startswith("MACE_"):
                del atoms.info[key]
        for key in list(atoms.arrays.keys()):
            if key.startswith("MACE_"):
                del atoms.arrays[key]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write(args.output, atoms_list, format="extxyz")
    os.unlink(tmp_xyz)
    print(f"INFERENCE OK {n} structures")


if __name__ == "__main__":
    main()
