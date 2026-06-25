#!/usr/bin/env python3
"""Run one active-learning round for mat-agent.

Purpose: coordinate uncertainty inference, query selection, oracle labeling, training,
evaluation, and summary emission for one AL iteration.
Inputs: JSON config path, round number, output directory.
Outputs: round artifacts, step logs, and round_N_summary.json.
Dependencies: ASE, numpy, json, argparse, subprocess, pathlib, datetime.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ase.io import read, write


def parse_args() -> argparse.Namespace:
    """Parse loop coordinator arguments."""
    p = argparse.ArgumentParser(description="Run one mat-agent active-learning iteration")
    p.add_argument("--config", required=True)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write validated JSON."""
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    json.loads(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run_step(step: int, name: str, cmd: list[str], log_dir: Path) -> None:
    """Run one step and log stdout/stderr."""
    log_path = log_dir / f"round_step{step}_{name}.log"
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"step {step} {name} failed with exit {proc.returncode}; log={log_path}")


def model_api(root: Path, model: str, script: str) -> Path:
    """Resolve model aliases to API scripts."""
    aliases = {"dpa": "deepmd", "deepmd": "deepmd", "uma": "fairchem", "fairchem": "fairchem", "mace": "mace"}
    key = aliases.get(model)
    if key is None:
        raise ValueError(f"unknown model alias: {model}")
    path = root / "models" / key / "api" / script
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def merge_extxyz(current_train: str, labeled: Path, output: Path) -> int:
    """Merge current train data and newly labeled structures."""
    merged = read(current_train, ":") + read(labeled, ":")
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, merged, format="extxyz")
    return len(merged)


def find_checkpoint(checkpoint_dir: Path) -> Path:
    """Find the newest checkpoint under a training output directory."""
    candidates: list[Path] = []
    for pattern in ("*.model", "*.pt", "*.pth", "*.ckpt", "checkpoint"):
        candidates.extend(p for p in checkpoint_dir.rglob(pattern) if p.is_file())
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found under {checkpoint_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def previous_summary(out: Path, round_id: int) -> dict[str, Any] | None:
    """Read previous round summary if available."""
    prev = out / f"round_{round_id - 1}_summary.json"
    return json.loads(prev.read_text()) if prev.is_file() else None


def main() -> None:
    """Run the configured active-learning round."""
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / f"round_{args.round}_summary.json"
    paths = {
        "pool_uncertainty": out / f"round_{args.round}_pool_uncertainty.extxyz",
        "selected": out / f"round_{args.round}_selected.extxyz",
        "remaining": out / f"round_{args.round}_remaining.extxyz",
        "al_summary": out / f"round_{args.round}_al_summary.json",
        "labeled": out / f"round_{args.round}_labeled.extxyz",
        "train_augmented": out / f"round_{args.round}_train_augmented.extxyz",
        "checkpoint_dir": out / f"round_{args.round}_checkpoint",
        "metrics": out / f"round_{args.round}_metrics.json",
        "eval_report": out / f"round_{args.round}_eval_report.json",
    }
    current_step = 0
    try:
        current_step = 1
        explorer_cmd = ["bash", str(model_api(root, "mace", "inference.sh")), "--model", cfg["explorer_checkpoint"], "--data", cfg["candidate_pool"], "--output", str(paths["pool_uncertainty"]), "--device", cfg.get("device", "cuda"), "--uncertainty"]
        if cfg.get("explorer_save_embeddings"):
            explorer_cmd.append("--save-embeddings")
        run_step(1, "explorer", explorer_cmd, out)
        current_step = 2
        run_step(2, "select", [sys.executable, str(root / "tools" / "active_learning.py"), "--pool", str(paths["pool_uncertainty"]), "--output-high", str(paths["selected"]), "--output-remaining", str(paths["remaining"]), "--summary", str(paths["al_summary"]), "--threshold", str(cfg.get("uncertainty_threshold", 0.3)), "--max-select", str(cfg.get("max_select", 50)), "--strategy", cfg.get("al_strategy", "combined"), "--uncertainty-field", cfg.get("uncertainty_field", "max"), "--diversity-weight", str(cfg.get("diversity_weight", 0.3))], out)
        current_step = 3
        oracle_cmd = [sys.executable, str(root / "tools" / "oracle_label.py"), "--input", str(paths["selected"]), "--output", str(paths["labeled"]), "--oracle", cfg.get("oracle_model", "uma"), "--model", cfg["oracle_checkpoint"], "--device", cfg.get("device", "cuda")]
        if cfg.get("oracle_head"):
            oracle_cmd.extend(["--head", cfg["oracle_head"]])
        run_step(3, "oracle", oracle_cmd, out)
        current_step = 4
        n_total_train = merge_extxyz(cfg["current_train_data"], paths["labeled"], paths["train_augmented"])
        current_step = 5
        train_cmd = ["bash", str(model_api(root, cfg["finetune_model"], "train.sh")), "--foundation-model", cfg["finetune_checkpoint"], "--train-data", str(paths["train_augmented"]), "--val-data", cfg["val_data"], "--output-dir", str(paths["checkpoint_dir"]), "--finetune-mode", cfg.get("finetune_mode", "single"), "--max-steps", str(cfg.get("finetune_max_steps", 1000)), "--device", cfg.get("device", "cuda")]
        if cfg.get("finetune_head"):
            train_cmd.extend(["--head", cfg["finetune_head"]])
        run_step(5, "train", train_cmd, out)
        checkpoint = find_checkpoint(paths["checkpoint_dir"])
        current_step = 6
        eval_cmd = ["bash", str(model_api(root, cfg["finetune_model"], "evaluate.sh")), "--model", str(checkpoint), "--data", cfg["val_data"], "--output", str(paths["metrics"]), "--report-json", str(paths["eval_report"]), "--task-id", cfg.get("task_id", f"round_{args.round}"), "--uncertainty-threshold", str(cfg.get("uncertainty_threshold", 0.3)), "--device", cfg.get("device", "cuda")]
        if cfg.get("finetune_head"):
            eval_cmd.extend(["--head", cfg["finetune_head"]])
        run_step(6, "evaluate", eval_cmd, out)
        al = json.loads(paths["al_summary"].read_text())
        ev = json.loads(paths["eval_report"].read_text())
        prev = previous_summary(out, args.round)
        before_force = None if prev is None else prev.get("val_force_MAE_after")
        before_energy = None if prev is None else prev.get("val_energy_MAE_after")
        rec = f"Round {args.round} complete. Force MAE {ev.get('val_force_MAE_eV_per_angstrom')} eV/A after labeling {al.get('n_selected')} structures."
        write_json(summary_path, {
            "round": args.round,
            "task_id": cfg.get("task_id"),
            "n_newly_labeled": al.get("n_selected"),
            "n_total_train": n_total_train,
            "n_pool_remaining": al.get("n_remaining"),
            "explorer_model": cfg.get("explorer_model", "mace"),
            "oracle_model": cfg.get("oracle_model", "uma"),
            "finetune_model": cfg.get("finetune_model"),
            "uncertainty_field_used": al.get("uncertainty_field_used"),
            "diversity_method_used": al.get("diversity_method_used"),
            "val_force_MAE_before": before_force,
            "val_force_MAE_after": ev.get("val_force_MAE_eV_per_angstrom"),
            "val_energy_MAE_before": before_energy,
            "val_energy_MAE_after": ev.get("val_energy_MAE_eV_per_atom"),
            "high_uncertainty_elements": al.get("high_uncertainty_elements"),
            "element_variance_breakdown": al.get("element_variance_breakdown"),
            "recommendation": rec,
            "timestamp": timestamp(),
            "status": "ok",
        })
        print(f"ACTIVE LEARNING LOOP OK round={args.round} summary={summary_path}")
    except Exception as exc:
        write_json(summary_path, {"round": args.round, "status": "error", "error_step": current_step, "error_message": str(exc), "timestamp": timestamp()})
        raise


if __name__ == "__main__":
    main()
