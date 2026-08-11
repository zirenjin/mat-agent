from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Naive fine-tune baseline placeholder.")
    parser.add_argument("--env-config", default="benchmark/envs/softening_liznino/env_config.yaml")
    parser.add_argument("--selected-ids", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = {
        "status": "blocked",
        "reason": "No MACE or CHGNet backend is installed in the default Python environment.",
        "env_config": args.env_config,
        "selected_ids": args.selected_ids,
        "next_step": "Run inside a verified MACE/CHGNet environment or add a model-backend adapter.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
