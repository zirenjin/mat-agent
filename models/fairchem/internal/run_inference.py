#!/usr/bin/env python3
"""FairChem/UMA inference — container entry point."""
from __future__ import annotations
import argparse, sys, os, time
from pathlib import Path
import numpy as np
from ase.io import read, write
from ase_conventions import set_pred


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--head", default="")
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    # Require --head
    if not args.head:
        print("ERROR: FairChem requires --head (task_name): oc20/omat/omol/odac", file=sys.stderr)
        sys.exit(1)

    # Validate local model path
    if not Path(args.model_path).is_file():
        print(f"FATAL: model not found: {args.model_path}", file=sys.stderr)
        sys.exit(2)
    if not Path(args.data).is_file():
        print(f"FATAL: data not found: {args.data}", file=sys.stderr)
        sys.exit(2)

    from fairchem.core import pretrained_mlip
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    # load_predict_unit (LOCAL ckpt) NOT get_predict_unit (HF gating)
    print(f"[fairchem] loading {args.model_path} head={args.head} device={args.device}", file=sys.stderr)
    predictor = pretrained_mlip.load_predict_unit(args.model_path, device=args.device)
    calc = FAIRChemCalculator(predict_unit=predictor, task_name=args.head)

    atoms_list = read(args.data, ":")
    n = len(atoms_list)
    n_ok = 0
    t0 = time.time()

    for i, atoms in enumerate(atoms_list):
        try:
            atoms.calc = calc
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=np.float64)
            atoms.calc = None
            set_pred(atoms, energy=energy, forces=forces)
            # Clear any fairchem-specific keys
            for k in list(atoms.info.keys()):
                if k.startswith("fairchem_") or k.startswith("FAIRChem"):
                    del atoms.info[k]
            n_ok += 1
        except Exception as e:
            print(f"  WARN struct {i}: {e}", file=sys.stderr)
            atoms.calc = None
        if (i + 1) % max(1, n // 5) == 0:
            print(f"  {i+1}/{n}", file=sys.stderr)

    elapsed = time.time() - t0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write(args.output, atoms_list, format="extxyz")

    if n_ok == 0:
        print("FATAL: all structures failed", file=sys.stderr)
        sys.exit(3)

    print(f"INFERENCE OK {n_ok} structures in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
