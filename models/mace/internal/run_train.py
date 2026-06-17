#!/usr/bin/env python3
"""MACE fine-tuning — container entry point."""
from __future__ import annotations
import argparse, subprocess, sys, os
from pathlib import Path
from ase.io import read, write
from ase_conventions import set_pred
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--train-data", required=True)
    p.add_argument("--val-data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--test-data", default="")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for label, p in [("model", args.model_path), ("train", args.train_data), ("val", args.val_data)]:
        if not Path(p).is_file():
            print(f"FATAL: {label} not found: {p}", file=sys.stderr)
            sys.exit(2)

    print("=== MACE fine-tuning ===", file=sys.stderr)
    cmd = [
        "mace_run_train",
        "--name", "mace_ft",
        "--model", args.model_path,
        "--train_file", args.train_data,
        "--valid_file", args.val_data,
        "--max_num_epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--device", args.device,
        "--default_dtype", "float64",
        "--output_dir", str(out),
        "--restart_latest",
    ]
    print("[mace] " + " ".join(cmd), file=sys.stderr)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"FATAL: mace_run_train failed with exit {result.returncode}", file=sys.stderr)
        sys.exit(2)

    ft_model = str(out / "mace_ft.model")
    print(f"TRAIN OK (model: {ft_model})", file=sys.stderr)

    # Post-train inference if test data provided
    if args.test_data and Path(args.test_data).exists():
        print("=== Post-train inference ===", file=sys.stderr)
        from mace.calculators import MACECalculator
        calc = MACECalculator(model_path=ft_model, device=args.device, default_dtype="float64")
        atoms_list = read(args.test_data, ":")
        n_ok = 0
        for atoms in atoms_list:
            try:
                atoms.calc = calc
                e = float(atoms.get_potential_energy())
                f = np.asarray(atoms.get_forces(), dtype=np.float64)
                set_pred(atoms, energy=e, forces=f)
                atoms.calc = None
                n_ok += 1
            except Exception as exc:
                print(f"  WARN struct: {exc}", file=sys.stderr)
                atoms.calc = None
        pred_path = str(out / "ft_predictions.xyz")
        write(pred_path, atoms_list, format="extxyz")
        print(f"Post-train inference: {n_ok}/{len(atoms_list)} -> {pred_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
