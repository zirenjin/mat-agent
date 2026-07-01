#!/usr/bin/env python3
"""Training command helpers."""
from __future__ import annotations

from .api import api_script


def train_cmd(model: str, foundation_model: str, train_data: str, val_data: str, output_dir: str, *, device: str = "cuda", max_steps: int = 1000, head: str = "") -> list[str]:
    """Build a canonical model training command."""
    cmd = [
        "bash", str(api_script(model, "train")),
        "--foundation-model", foundation_model,
        "--train-data", train_data,
        "--val-data", val_data,
        "--output-dir", output_dir,
        "--max-steps", str(max_steps),
        "--device", device,
    ]
    if head:
        cmd.extend(["--head", head])
    return cmd
