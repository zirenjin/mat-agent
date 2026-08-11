from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def backend_status() -> dict[str, bool]:
    return {
        "mace": importlib.util.find_spec("mace") is not None,
        "chgnet": importlib.util.find_spec("chgnet") is not None,
        "ase": importlib.util.find_spec("ase") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-shot or score existing zero-shot predictions.")
    parser.add_argument("--labels")
    parser.add_argument("--predictions")
    parser.add_argument("--model", default="zero_shot")
    parser.add_argument("--output", default="benchmark/envs/softening_liznino/results/zero_shot_hidden.json")
    args = parser.parse_args()

    status = backend_status()
    if args.labels and args.predictions:
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("score.py")),
            "--labels",
            args.labels,
            "--predictions",
            args.predictions,
            "--baseline",
            "zero_shot",
            "--model",
            args.model,
            "--budget",
            "0",
            "--output",
            args.output,
        ]
        raise SystemExit(subprocess.call(cmd))

    out = {
        "status": "blocked",
        "reason": "No prediction file was provided, and model-backed zero-shot inference is not wired yet.",
        "backend_status": status,
        "next_step": "Provide an extxyz prediction file or run this inside a verified MACE/CHGNet backend.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
