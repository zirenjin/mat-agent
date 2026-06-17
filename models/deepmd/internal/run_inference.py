#!/usr/bin/env python3
"""DeePMD-kit inference (Docker container entry point)."""
from __future__ import annotations
import argparse, sys, time, numpy as np
from pathlib import Path
from ase.io import read, write
sys.path.insert(0, "/opt/mat-agent-tools")
from ase_conventions import set_pred, normalize_predictions

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--head", default="MP_traj_v024_alldata_mixu")
    p.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--dtype", default="float32", choices=["float32","float64"])
    return p.parse_args()

def load_calc(model_path, head, device):
    try:
        from deepmd.calculator import DP
        return DP(model=model_path, head=head)
    except Exception:
        from deepmd.pt.calculator import DP
        try:
            return DP(model=model_path, head=head, device=device)
        except TypeError:
            return DP(model=model_path, head=head)

def main():
    args = parse_args()
    if not Path(args.model_path).exists():
        print(f"FATAL: model not found: {args.model_path}", file=sys.stderr); sys.exit(2)
    if not Path(args.data).exists():
        print(f"FATAL: data not found: {args.data}", file=sys.stderr); sys.exit(2)
    atoms_list = read(args.data, ":")
    print(f"Loaded {len(atoms_list)} structures", file=sys.stderr)
    calc = load_calc(args.model_path, args.head, args.device)
    n_success = 0
    t0 = time.time()
    for i, atoms in enumerate(atoms_list):
        try:
            atoms.calc = calc
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=np.float64)
            set_pred(atoms, energy, forces)
            atoms.calc = None
            n_success += 1
        except Exception as e:
            print(f"  WARN struct {i}: {e}", file=sys.stderr)
            atoms.calc = None
        if (i+1) % max(1, len(atoms_list)//10) == 0:
            print(f"  {i+1}/{len(atoms_list)}", file=sys.stderr)
    elapsed = time.time() - t0
    print(f"Inference: {n_success}/{len(atoms_list)} in {elapsed:.1f}s", file=sys.stderr)
    normalize_predictions(atoms_list)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write(args.output, atoms_list, format="extxyz")
    if n_success == 0:
        print("FATAL: all structures failed", file=sys.stderr); sys.exit(3)
    print(f"INFERENCE OK {n_success} structures")

if __name__ == "__main__":
    main()
