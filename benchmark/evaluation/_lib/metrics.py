"""Single entry point to the frozen scorers. Nothing else computes a metric."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCORING = REPO / "scoring"


def run_scorer(family: str, metric: str, args: list[str]) -> dict:
    """Call scoring/<family>/<metric>.py (or scoring/<family>/mace_mof_0/<metric>.py)."""
    candidates = [
        SCORING / family / f"{metric}.py",
        SCORING / family / "mace_mof_0" / f"{metric}.py",
    ]
    script = next((c for c in candidates if c.is_file()), None)
    if script is None:
        raise FileNotFoundError(f"no frozen scorer for {family}/{metric}")
    proc = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"scorer {family}/{metric} exited {proc.returncode}: {proc.stderr.strip()}"
        )
    last = proc.stdout.strip().splitlines()[-1]
    return json.loads(last)
