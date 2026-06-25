#!/usr/bin/env python3
"""Prepare a MACE committee explorer for uncertainty-driven active learning.

Purpose: create an MHC-compatible explorer manifest from several MACE head checkpoints.
Inputs: base checkpoint, calibration extxyz, output checkpoint path/directory, head count.
Outputs: mhc_manifest.json plus committee checkpoint files usable by inference.sh --uncertainty.
Dependencies: ASE, numpy, json, argparse, subprocess, pathlib, datetime.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ase.io import read


def parse_args() -> argparse.Namespace:
    """Parse MHC preparation arguments."""
    p = argparse.ArgumentParser(description="Prepare a MACE committee explorer manifest")
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--calibration-data", required=True)
    p.add_argument("--output-checkpoint", required=True)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--head-train-steps", type=int, default=500)
    p.add_argument("--freeze-backbone", action="store_true", help="Documented intent; current backend uses API-supported fine-tuning")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write validated JSON."""
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    json.loads(text)
    path.write_text(text)


def find_checkpoint(directory: Path) -> Path:
    """Find the newest MACE checkpoint produced by training."""
    candidates = [p for p in directory.rglob("*.model") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"no .model checkpoint found under {directory}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run(cmd: list[str], log_path: Path) -> None:
    """Run a subprocess and tee stdout/stderr to a log file."""
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit {proc.returncode}: {' '.join(cmd)}; log={log_path}")


def main() -> None:
    """Train/copy committee heads and write an MHC manifest."""
    args = parse_args()
    if args.n_heads < 2:
        print("ERROR: --n-heads must be >= 2", file=sys.stderr)
        sys.exit(1)
    base = Path(args.base_checkpoint).resolve()
    calib = Path(args.calibration_data).resolve()
    if not base.is_file():
        print(f"ERROR: base checkpoint not found: {base}", file=sys.stderr)
        sys.exit(1)
    if not calib.is_file():
        print(f"ERROR: calibration data not found: {calib}", file=sys.stderr)
        sys.exit(1)
    output_arg = Path(args.output_checkpoint).resolve()
    out_dir = output_arg if output_arg.suffix != ".json" else output_arg.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "mhc_manifest.json"
    root = Path(__file__).resolve().parents[1]
    train_api = root / "models" / "mace" / "api" / "train.sh"
    val_data = calib
    try:
        structures = read(str(calib), ":")
        if len(structures) >= 10:
            val_data = calib
    except Exception:
        pass
    heads = []
    for i in range(args.n_heads):
        head_dir = out_dir / f"head_{i}"
        head_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / f"head_{i}.log"
        if args.head_train_steps <= 0:
            ckpt = head_dir / f"head_{i}.model"
            shutil.copy2(base, ckpt)
        else:
            cmd = [
                "bash", str(train_api),
                "--foundation-model", str(base),
                "--train-data", str(calib),
                "--val-data", str(val_data),
                "--output-dir", str(head_dir),
                "--finetune-mode", "single",
                "--max-steps", str(args.head_train_steps),
                "--batch-size", "4",
                "--device", args.device,
                "--seed", str(args.seed + i),
            ]
            run(cmd, log_path)
            ckpt = find_checkpoint(head_dir)
        heads.append({"name": f"head_{i}", "checkpoint": str(ckpt.relative_to(out_dir) if ckpt.is_relative_to(out_dir) else ckpt), "seed": args.seed + i})
        print(f"head {i}: checkpoint={ckpt}")
    write_json(manifest_path, {"format": "mat-agent-mace-committee-v1", "base_checkpoint": str(base), "calibration_data": str(calib), "n_heads": args.n_heads, "freeze_backbone_requested": bool(args.freeze_backbone), "created_at": timestamp(), "heads": heads})
    if output_arg != out_dir and output_arg.name != "mhc_manifest.json":
        output_arg.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, output_arg)
    print(f"MHC EXPLORER READY n_heads={args.n_heads} checkpoint={manifest_path}")
    print("SANITY CHECK SKIPPED mean_variance=null eV/A (run inference.sh --uncertainty on the manifest after training to validate)")


if __name__ == "__main__":
    main()
