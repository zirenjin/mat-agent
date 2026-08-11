#!/usr/bin/env python3
"""Command builders for the MatterTune-centered runtime."""
from __future__ import annotations

from pathlib import Path
from mat_agent.registry import repo_root

MODEL_ALIASES = {"mace": "mace", "deepmd": "deepmd", "dpa": "deepmd", "fairchem": "fairchem", "uma": "fairchem"}
COMMANDS = {"train", "inference", "evaluate"}


def api_script(model: str, command: str, *, root: str | Path | None = None) -> Path:
    """Return `models/<model>/api/<command>.sh` for a supported model."""
    key = MODEL_ALIASES.get(model)
    if key is None:
        raise ValueError(f"unknown model: {model}")
    if command not in COMMANDS:
        raise ValueError(f"unknown model API command: {command}")
    base = Path(root) if root else repo_root()
    path = base / "mattertune" / key / "api" / f"{command}.sh"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def train_cmd(model: str, foundation_model: str, train_data: str, val_data: str, output_dir: str, *, device: str = "cuda", max_steps: int = 1000, head: str = "") -> list[str]:
    """Build a canonical model training command."""
    cmd = ["bash", str(api_script(model, "train")), "--foundation-model", foundation_model, "--train-data", train_data, "--val-data", val_data, "--output-dir", output_dir, "--max-steps", str(max_steps), "--device", device]
    if head:
        cmd.extend(["--head", head])
    return cmd


def inference_cmd(model: str, checkpoint: str, data: str, output: str, *, device: str = "cuda", head: str = "") -> list[str]:
    """Build a canonical model inference command."""
    cmd = ["bash", str(api_script(model, "inference")), "--model", checkpoint, "--data", data, "--output", output, "--device", device]
    if head:
        cmd.extend(["--head", head])
    return cmd


def evaluate_cmd(model: str, checkpoint: str, data: str, output: str, *, device: str = "cuda", head: str = "") -> list[str]:
    """Build a canonical model evaluation command."""
    cmd = ["bash", str(api_script(model, "evaluate")), "--model", checkpoint, "--data", data, "--output", output, "--device", device]
    if head:
        cmd.extend(["--head", head])
    return cmd
