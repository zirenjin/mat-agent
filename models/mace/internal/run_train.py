#!/usr/bin/env python3
"""MACE fine-tuning — container entry point.

The public bash API passes a foundation checkpoint plus extxyz data. This
runner keeps MACE-specific choices here: canonical label keys, force-dominated
loss, optional SWA, and output locations.
"""
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


def env(name, default=""):
    return os.environ.get(name, default)


def env_float(name, default):
    value = env(name)
    return default if value == "" else value


def env_int(name, default):
    value = env(name)
    return default if value == "" else str(value)


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for label, p in [("model", args.model_path), ("train", args.train_data), ("val", args.val_data)]:
        if not Path(p).is_file():
            print(f"FATAL: {label} not found: {p}", file=sys.stderr)
            sys.exit(2)

    print("=== MACE fine-tuning ===", file=sys.stderr)
    name = env("MAT_AGENT_RUN_NAME", "mace_ft")
    lr = env_float("MAT_AGENT_LR", "0.005")
    weight_decay = env_float("MAT_AGENT_WEIGHT_DECAY", "5e-7")
    energy_weight = env_float("MAT_AGENT_ENERGY_WEIGHT", "1")
    forces_weight = env_float("MAT_AGENT_FORCES_WEIGHT", "1000")
    seed = env_int("MAT_AGENT_SEED", "1234")
    grad_clip = env("MAT_AGENT_GRAD_CLIP")

    cmd = [
        "mace_run_train",
        "--name", name,
        "--foundation_model", args.model_path,
        "--multiheads_finetuning", "False",
        "--E0s", "average",
        "--train_file", args.train_data,
        "--valid_file", args.val_data,
        "--energy_key", "ref_energy",
        "--forces_key", "ref_forces",
        "--loss", "weighted",
        "--energy_weight", str(energy_weight),
        "--forces_weight", str(forces_weight),
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--max_num_epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--valid_batch_size", str(max(args.batch_size, 8)),
        "--device", args.device,
        "--default_dtype", "float64",
        "--seed", str(seed),
        "--eval_interval", "1",
        "--log_dir", str(out / "logs"),
        "--model_dir", str(out),
        "--checkpoints_dir", str(out / "checkpoints"),
        "--results_dir", str(out / "results"),
        "--save_cpu",
        "--restart_latest",
    ]
    if args.test_data and Path(args.test_data).exists():
        cmd.extend(["--test_file", args.test_data])
    if grad_clip:
        cmd.extend(["--clip_grad", str(grad_clip)])
    if env("MAT_AGENT_ENABLE_SWA", "1") != "0" and args.epochs >= 8:
        start_swa = env_int("MAT_AGENT_START_SWA", str(max(1, int(args.epochs * 0.8))))
        swa_energy_weight = env_float("MAT_AGENT_SWA_ENERGY_WEIGHT", "1000")
        swa_forces_weight = env_float("MAT_AGENT_SWA_FORCES_WEIGHT", "100")
        cmd.extend([
            "--swa",
            "--start_swa", str(start_swa),
            "--swa_energy_weight", str(swa_energy_weight),
            "--swa_forces_weight", str(swa_forces_weight),
        ])
    if env("MAT_AGENT_ENABLE_EMA", "1") != "0":
        cmd.extend(["--ema", "--ema_decay", env_float("MAT_AGENT_EMA_DECAY", "0.99")])
    print("[mace] " + " ".join(cmd), file=sys.stderr)
    result = subprocess.run(cmd, capture_output=False)
    ft_model = str(out / f"{name}.model")
    if result.returncode != 0:
        if Path(ft_model).is_file():
            print(
                f"WARNING: mace_run_train exited {result.returncode}, "
                f"but model exists at {ft_model}; continuing",
                file=sys.stderr,
            )
        else:
            print(f"FATAL: mace_run_train failed with exit {result.returncode}", file=sys.stderr)
            sys.exit(2)

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
