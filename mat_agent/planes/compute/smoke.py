#!/usr/bin/env python3
"""Smoke-run command construction helpers."""
from __future__ import annotations


def clamp_steps(args: list[str], *, max_steps: int = 4) -> list[str]:
    """Return a copy of command args with --max-steps set to a smoke value."""
    out = list(args)
    for idx, item in enumerate(out):
        if item == "--max-steps" and idx + 1 < len(out):
            out[idx + 1] = str(max_steps)
            return out
        if item.startswith("--max-steps="):
            out[idx] = f"--max-steps={max_steps}"
            return out
    out.extend(["--max-steps", str(max_steps)])
    return out
