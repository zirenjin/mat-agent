#!/usr/bin/env python3
"""Thin, additional scalar objective derived from an existing verdict.json.

The frozen evaluator (evaluate_mof0.py + _lib/aggregate.py) produces a
four-condition hard-gate verdict (physics_primary, no_ml_regression,
reproducible, compute_legal -> beats_baseline) plus per-MOF aggregate_ratio
values -- but no single scalar "how good is this candidate" number. This
file adds exactly one, computed ONLY from fields evaluate_mof0.py/aggregate.py
already produce. It changes no scoring logic in either of those files and
introduces no new scientific metric.

    scientific_score = 0.0                                         if not beats_baseline
                      = mean(1 - aggregate_ratio) over scored MOFs  if beats_baseline

Rationale (see benchmark/harbor/README.md "Objective" section for the fuller
version handed to reviewers):

1. Baseline-relative: built entirely from aggregate_ratio, itself a
   candidate/baseline ratio (aggregate.py's per-MOF geometric mean).
2. No trivial compensation for a catastrophic failure: the score is a hard
   0 unless beats_baseline is True, and beats_baseline is already an AND of
   all four conditions plus aggregate.py's own catastrophic-endpoint
   override (a single MOF that regresses badly enough fails the whole
   verdict regardless of how good the others are) -- this file does not
   relax that in any way, only reports a magnitude once it already holds.
3. Both categories required: beats_baseline requires BOTH physics_primary
   (Stage B) and no_ml_regression (Stage A), so a win on one axis alone
   cannot produce a nonzero score.
4. No unreproducible metric becomes a required term: inherited unchanged
   from aggregate.py's _geomean, which already drops null endpoints
   (phonon_frequency_rmse is always null in this release, per protocol.md).
5. Frozen: this formula and file are pinned by
   benchmark/integrity/verify_frozen.py like the rest of benchmark/**.

Usage:
    python benchmark/evaluation/scientific_score.py --verdict runs/SUBMISSION/eval/verdict.json
    python benchmark/evaluation/scientific_score.py --verdict verdict.json --json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compute_scientific_score(verdict_doc: dict) -> dict:
    verdict = verdict_doc["verdict"]
    beats = bool(verdict["beats_baseline"])
    sb = verdict_doc.get("stage_b_comparison", {})

    margins = [
        1.0 - v["aggregate_ratio"]
        for v in sb.values()
        if v.get("status") != "incomplete" and v.get("aggregate_ratio") is not None
    ]

    score = 0.0 if not beats or not margins else sum(margins) / len(margins)

    return {
        "scientific_score": score,
        "beats_baseline": beats,
        "n_mofs_in_margin": len(margins),
        "per_mof_margin": {
            m: (1.0 - v["aggregate_ratio"])
            for m, v in sb.items()
            if v.get("status") != "incomplete" and v.get("aggregate_ratio") is not None
        },
        "conditions": verdict["conditions"],
        "formula": "0.0 if not beats_baseline else mean(1 - aggregate_ratio) over scored MOFs",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdict", required=True, help="path to a verdict.json written by evaluate_mof0.py")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    verdict_doc = json.loads(Path(args.verdict).read_text())
    result = compute_scientific_score(verdict_doc)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"scientific_score: {result['scientific_score']:.4f}")
        print(f"beats_baseline:   {result['beats_baseline']}")
        print(f"n_mofs_in_margin: {result['n_mofs_in_margin']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
