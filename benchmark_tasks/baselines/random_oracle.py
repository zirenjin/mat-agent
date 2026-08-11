from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml


def read_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def sample_ids(ids: list[str], budget: int, seed: int) -> list[str]:
    if budget > len(ids):
        raise ValueError(f"Budget {budget} exceeds oracle pool size {len(ids)}")
    rng = random.Random(seed)
    return sorted(rng.sample(ids, budget))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample deterministic random-oracle frame IDs.")
    parser.add_argument("--split-manifest", default="benchmark/envs/softening_liznino/split_manifest.yaml")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = yaml.safe_load(Path(args.split_manifest).read_text())
    ids_file = Path(manifest["oracle_pool"]["ids_file"])
    ids = read_ids(ids_file)
    selected = sample_ids(ids, args.budget, args.seed)

    out = {
        "split_manifest": args.split_manifest,
        "budget": args.budget,
        "seed": args.seed,
        "n_oracle_pool": len(ids),
        "selected_ids": selected,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
