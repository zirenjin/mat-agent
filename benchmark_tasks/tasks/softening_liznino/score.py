from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from ase.io import read

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.scoring.metrics import compute_metrics


def read_split_ids(split_manifest: Path, split: str) -> list[str] | None:
    manifest = yaml.safe_load(split_manifest.read_text())
    entry = manifest.get(split)
    if not isinstance(entry, dict):
        raise KeyError(f"Split {split!r} not found in {split_manifest}")
    ids_file = Path(entry["ids_file"])
    if not ids_file.exists():
        return None
    return [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]


def select_atoms(atoms: list, ids: list[str] | None) -> list:
    if ids is None:
        return atoms
    by_id = {str(a.info.get("frame_id", i)): a for i, a in enumerate(atoms)}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} split IDs were not found; first missing ID: {missing[0]}")
    return [by_id[i] for i in ids]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score predictions for the softening minimal environment.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split-manifest", default="benchmark/envs/softening_liznino/split_manifest.yaml")
    parser.add_argument("--split", default="hidden_test")
    parser.add_argument("--env-id", default="BENCHMARK-SOFT-LIZNINIO-004")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    labels = read(args.labels, ":")
    predictions = read(args.predictions, ":")
    ids = read_split_ids(Path(args.split_manifest), args.split)
    labels = select_atoms(labels, ids)
    predictions = select_atoms(predictions, ids)
    result = compute_metrics(labels, predictions)

    out = {
        "env_id": args.env_id,
        "baseline": args.baseline,
        "model": args.model,
        "seed": args.seed,
        "budget": args.budget,
        "round": args.round,
        "split": args.split,
        "metrics": result.as_dict(),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
