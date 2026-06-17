#!/usr/bin/env python3
"""
run_train.py — DeePMD-kit fine-tuning entry point (runs inside Docker container).

Fine-tunes a DPA pretrained checkpoint on new data, freezes the model,
and runs post-train inference on test data.

Usage (inside container):
    python /opt/internal/run_train.py \
        --model-path /work/models/pretrained.pt \
        --train-data /work/data/train.extxyz \
        --val-data /work/data/val.extxyz \
        --test-data /work/data/test.extxyz \
        --output-dir /work/output \
        --head MP_traj_v024_alldata_mixu \
        --device cuda \
        --epochs 10
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/opt/mat-agent-tools")
from ase_conventions import set_pred, normalize_predictions

# DeePMD type_map for general inorganic chemistry (WBM / Materials Project coverage)
TYPE_MAP = [
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S",
    "Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga",
    "Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd",
    "Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm",
    "Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os",
    "Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa",
    "U","Np","Pu",
]


def parse_args():
    p = argparse.ArgumentParser(description="DeePMD-kit fine-tuning")
    p.add_argument("--model-path", required=True, help="Pretrained .pt checkpoint")
    p.add_argument("--train-data", required=True, help="Train extxyz")
    p.add_argument("--val-data", required=True, help="Validation extxyz")
    p.add_argument("--test-data", default="", help="Test extxyz for post-train inference")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--head", default="MP_traj_v024_alldata_mixu", help="Multitask head")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--epochs", type=int, default=10, help="Training steps (small for smoke test)")
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def convert_extxyz_to_npy(extxyz_path: str, npy_dir: str, split_name: str) -> str:
    """Convert extxyz to DeePMD npy format. Returns the system directory path."""
    import dpdata
    ms = dpdata.MultiSystems.from_file(extxyz_path, fmt="quip/gap/xyz")
    ms.to_deepmd_npy(npy_dir)
    # Find the system directories created
    system_dirs = sorted(Path(npy_dir).rglob("type.raw"))
    system_paths = [str(d.parent) for d in system_dirs]
    n_frames = sum(s.get_nframes() for s in ms)
    print(f"  {split_name}: {n_frames} frames → {len(system_paths)} systems", file=sys.stderr)
    return system_paths


def write_input_json(train_systems: list[str], val_systems: list[str],
                     output_dir: str, epochs: int, seed: int = 42) -> str:
    """Write DeePMD input.json for energy-only fine-tuning."""
    config = {
        "model": {
            "type_map": TYPE_MAP,
            "descriptor": {"type": "se_e2_a", "sel": "auto"},
            "fitting_net": {"neuron": [240, 240, 240], "resnet_dt": True},
        },
        "learning_rate": {
            "type": "exp", "decay_steps": 100, "start_lr": 1e-4, "stop_lr": 1e-6,
        },
        "loss": {
            "type": "ener",
            "start_pref_e": 0.02, "limit_pref_e": 1,
            "start_pref_f": 0,    "limit_pref_f": 0,
        },
        "training": {
            "training_data":   {"systems": train_systems, "batch_size": "auto:32"},
            "validation_data": {"systems": val_systems,   "batch_size": "auto:32"},
            "numb_steps": epochs,
            "seed": seed,
            "disp_file":  os.path.join(output_dir, "lcurve.out"),
            "disp_freq":  min(50, max(1, epochs // 5)),
            "save_freq":  max(1, epochs // 2),
        },
    }
    config_path = os.path.join(output_dir, "input.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {config_path}", file=sys.stderr)
    return config_path


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Convert data to DeePMD npy format
    print("=== Converting data ===", file=sys.stderr)
    t0 = time.time()

    train_systems = convert_extxyz_to_npy(args.train_data, str(out / "dp_train"), "train")
    val_systems   = convert_extxyz_to_npy(args.val_data,   str(out / "dp_val"),   "val")

    # 2. Write training config
    print("=== Writing config ===", file=sys.stderr)
    config_path = write_input_json(train_systems, val_systems, str(out), args.epochs)

    # 3. Fine-tune
    print("=== Training ===", file=sys.stderr)
    train_t0 = time.time()
    abs_model = os.path.abspath(args.model_path)

    result = subprocess.run(
        ["dp", "--pt", "train", config_path, "--finetune", abs_model],
        cwd=str(out),
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"FATAL: dp train failed with exit {result.returncode}", file=sys.stderr)
        sys.exit(2)

    train_elapsed = time.time() - train_t0
    print(f"Training done in {train_elapsed:.1f}s", file=sys.stderr)

    # 4. Freeze model
    print("=== Freezing ===", file=sys.stderr)
    freeze_result = subprocess.run(
        ["dp", "--pt", "freeze", "-o", str(out / "model_ft.pt")],
        cwd=str(out),
        capture_output=False,
    )
    if freeze_result.returncode != 0:
        print(f"FATAL: dp freeze failed with exit {freeze_result.returncode}", file=sys.stderr)
        sys.exit(2)

    ft_model = str(out / "model_ft.pt")
    print(f"Frozen model: {ft_model}", file=sys.stderr)

    # 5. Post-train inference (if test data provided)
    if args.test_data and Path(args.test_data).exists():
        print("=== Post-train inference ===", file=sys.stderr)
        from ase.io import read, write

        try:
            from deepmd.pt.calculator import DP
        except ImportError:
            from deepmd.calculator import DP

        try:
            calc = DP(model=ft_model, head=args.head, device=args.device)
        except TypeError:
            calc = DP(model=ft_model, head=args.head)

        atoms_list = read(args.test_data, ":")
        n_ok = 0
        for i, atoms in enumerate(atoms_list):
            try:
                atoms.calc = calc
                energy = float(atoms.get_potential_energy())
                forces = np.asarray(atoms.get_forces(), dtype=np.float64)
                set_pred(atoms, energy, forces)
                atoms.calc = None
                n_ok += 1
            except Exception as e:
                print(f"  WARN: struct {i}: {e}", file=sys.stderr)
                atoms.calc = None

        normalize_predictions(atoms_list)
        pred_path = str(out / "ft_predictions.xyz")
        write(pred_path, atoms_list, format="extxyz")
        print(f"Post-train inference: {n_ok}/{len(atoms_list)} → {pred_path}", file=sys.stderr)

    total_elapsed = time.time() - t0
    print(f"TRAIN OK (total {total_elapsed:.1f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
