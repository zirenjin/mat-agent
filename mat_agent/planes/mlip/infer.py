#!/usr/bin/env python3
"""Inference command helpers."""
from __future__ import annotations

from .api import api_script


def inference_cmd(model: str, checkpoint: str, data: str, output: str, *, device: str = "cuda", head: str = "") -> list[str]:
    """Build a canonical model inference command."""
    cmd = ["bash", str(api_script(model, "inference")), "--model", checkpoint, "--data", data, "--output", output, "--device", device]
    if head:
        cmd.extend(["--head", head])
    return cmd
