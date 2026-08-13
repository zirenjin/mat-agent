#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a requested action against a budget JSON file.")
    parser.add_argument("--budget", required=True)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--train-minutes", type=float, default=0.0)
    parser.add_argument("--oracle-labels", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=0)
    args = parser.parse_args()

    budget = json.loads(Path(args.budget).read_text())
    requested_values = {
        "max_epochs": args.epochs,
        "max_train_minutes_per_round": args.train_minutes,
        "max_oracle_labels": args.oracle_labels,
        "max_gpus": args.gpus,
    }
    violations = []
    for key, requested in requested_values.items():
        limit = budget.get(key)
        if limit is not None and requested > limit:
            violations.append(f"{key}: requested {requested} > limit {limit}")

    result = {"ok": not violations, "violations": violations}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
