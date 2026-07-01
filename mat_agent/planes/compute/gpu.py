#!/usr/bin/env python3
"""GPU inspection helpers."""
from __future__ import annotations

import subprocess


def nvidia_smi_query() -> str:
    """Return nvidia-smi output, or an explanatory message when unavailable."""
    result = subprocess.run(["nvidia-smi"], text=True, capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else result.stderr
