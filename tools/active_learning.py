#!/usr/bin/env python3
"""Select high-value structures from an uncertainty-annotated pool.

Purpose: active-learning query selection with uncertainty and diversity.
Inputs: extxyz pool with pred_force_variance_* and optional mace_node_embedding.
Outputs: selected extxyz, remaining extxyz, and JSON selection summary.
Dependencies: ASE, numpy, json, argparse.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ase import Atoms
from ase.io import read, write

FIELD_MAP = {"max": "pred_force_variance_max", "p90": "pred_force_variance_p90", "mean": "pred_force_variance_mean"}


def parse_args() -> argparse.Namespace:
    """Parse query-selection arguments."""
    p = argparse.ArgumentParser(description="Select active-learning queries from an uncertainty pool")
    p.add_argument("--pool", required=True)
    p.add_argument("--output-high", required=True)
    p.add_argument("--output-remaining", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--max-select", type=int, default=50)
    p.add_argument("--strategy", choices=["threshold", "topk", "combined"], default="combined")
    p.add_argument("--uncertainty-field", choices=sorted(FIELD_MAP), default="max")
    p.add_argument("--diversity-weight", type=float, default=0.3)
    p.add_argument("--diversity-method", choices=["embedding", "composition"], default="embedding")
    return p.parse_args()


def timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: str, payload: dict[str, Any]) -> None:
    """Write validated JSON."""
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    json.loads(text)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


def uncertainty_scores(pool: Sequence[Atoms], field: str) -> np.ndarray:
    """Read the requested uncertainty scalar from every structure."""
    key = FIELD_MAP[field]
    scores = []
    for idx, atoms in enumerate(pool):
        if key not in atoms.info:
            raise ValueError(f"missing {key} in structure {idx}")
        scores.append(float(atoms.info[key]))
    return np.asarray(scores, dtype=np.float64)


def composition_features(pool: Sequence[Atoms]) -> np.ndarray:
    """Build normalized element-count features."""
    elements = sorted({sym for atoms in pool for sym in atoms.get_chemical_symbols()})
    rows = []
    for atoms in pool:
        counts = defaultdict(int)
        for sym in atoms.get_chemical_symbols():
            counts[sym] += 1
        row = np.asarray([counts[e] for e in elements], dtype=np.float64)
        total = float(np.sum(row))
        rows.append(row / total if total else row)
    return np.asarray(rows, dtype=np.float64)


def embedding_features(pool: Sequence[Atoms]) -> np.ndarray | None:
    """Mean-pool mace_node_embedding arrays when all structures contain them."""
    if not pool or any("mace_node_embedding" not in atoms.arrays for atoms in pool):
        return None
    rows = []
    for idx, atoms in enumerate(pool):
        emb = np.asarray(atoms.arrays["mace_node_embedding"], dtype=np.float64)
        if emb.ndim != 2 or emb.shape[0] != len(atoms):
            raise ValueError(f"mace_node_embedding shape mismatch in structure {idx}: {emb.shape}")
        rows.append(np.mean(emb, axis=0))
    return np.asarray(rows, dtype=np.float64)


def similarity(a: np.ndarray, b: np.ndarray, method: str) -> float:
    """Return similarity in [0, 1]-ish where larger means more redundant."""
    if method == "embedding":
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def choose(pool: list[Atoms], scores: np.ndarray, features: np.ndarray, method: str, args: argparse.Namespace) -> list[int]:
    """Select structure indices according to the configured strategy."""
    limit = max(0, min(args.max_select, len(pool)))
    order = list(np.argsort(-scores))
    if limit == 0:
        return []
    if args.strategy == "threshold":
        return [i for i in order if scores[i] > args.threshold][:limit]
    if args.strategy == "topk":
        return order[:limit]
    selected: list[int] = []
    remaining = set(order)
    max_score = float(np.max(scores)) if len(scores) else 1.0
    while remaining and len(selected) < limit:
        best_idx, best_value = None, -float("inf")
        for idx in list(remaining):
            penalty = max((similarity(features[idx], features[j], method) for j in selected), default=0.0)
            value = float(scores[idx]) - args.diversity_weight * penalty * max_score
            if value > best_value:
                best_idx, best_value = idx, value
        assert best_idx is not None
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def element_breakdown(pool: Sequence[Atoms], scores: np.ndarray) -> dict[str, float]:
    """Compute mean uncertainty by element presence."""
    values: dict[str, list[float]] = defaultdict(list)
    for atoms, score in zip(pool, scores):
        for sym in sorted(set(atoms.get_chemical_symbols())):
            values[sym].append(float(score))
    return {sym: float(np.mean(vals)) for sym, vals in sorted(values.items())}


def main() -> None:
    """Run active-learning selection."""
    args = parse_args()
    pool = read(args.pool, ":")
    scores = uncertainty_scores(pool, args.uncertainty_field)
    embeddings = embedding_features(pool) if args.diversity_method == "embedding" else None
    if embeddings is not None:
        features, method = embeddings, "embedding"
    else:
        features, method = composition_features(pool), "composition"
    print(f"DIVERSITY METHOD {method}", file=sys.stderr)
    chosen = set(choose(pool, scores, features, method, args))
    selected = [atoms for i, atoms in enumerate(pool) if i in chosen]
    remaining = [atoms for i, atoms in enumerate(pool) if i not in chosen]
    for path, atoms_list in [(args.output_high, selected), (args.output_remaining, remaining)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        write(path, atoms_list, format="extxyz")
    breakdown = element_breakdown(pool, scores)
    global_mean = float(np.mean(scores)) if len(scores) else 0.0
    high_elements = [sym for sym, val in breakdown.items() if val > 1.5 * global_mean]
    high_elements.sort(key=lambda sym: breakdown[sym], reverse=True)
    selected_scores = scores[list(chosen)] if chosen else np.asarray([], dtype=np.float64)
    remaining_scores = np.asarray([scores[i] for i in range(len(pool)) if i not in chosen], dtype=np.float64)
    write_json(args.summary, {
        "n_pool_input": len(pool),
        "n_selected": len(selected),
        "n_remaining": len(remaining),
        "strategy": args.strategy,
        "uncertainty_field_used": args.uncertainty_field,
        "diversity_method_used": method,
        "threshold": args.threshold,
        "max_select": args.max_select,
        "mean_uncertainty_selected": float(np.mean(selected_scores)) if len(selected_scores) else None,
        "mean_uncertainty_remaining": float(np.mean(remaining_scores)) if len(remaining_scores) else None,
        "high_uncertainty_elements": high_elements,
        "element_variance_breakdown": breakdown,
        "timestamp": timestamp(),
    })
    print(f"ACTIVE LEARNING OK {len(selected)} selected / {len(pool)} pool")


if __name__ == "__main__":
    main()
