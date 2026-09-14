#!/usr/bin/env python3
"""Run the evaluator N times on one model and write noise_estimate.json.

    python benchmark/evaluation/estimate_noise.py \
        --submission benchmark/baselines/mace_mof0/submissions/mofs_v2.json \
        --baseline --repeats 3 --out-dir runs/_noise --stages b

Each repeat perturbs the relaxation/phonopy seed. Tolerances in the verdict come
from the file this writes; without it the protocol defaults are used and marked
provisional.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmark.evaluation._lib.noise import write_estimate  # noqa: E402
from benchmark.evaluation.evaluate_mof0 import PROTOCOL, run_evaluation  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repeats", type=int, default=PROTOCOL["repro_repeats"])
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--stages", default="b")
    ap.add_argument("--no-sandbox", action="store_true")
    args = ap.parse_args()

    cards = []
    for k in range(args.repeats):
        seed = PROTOCOL["seed"] + k
        od = Path(args.out_dir) / f"repeat_{k}"
        proto = dict(PROTOCOL, seed=seed)
        # run_evaluation reads PROTOCOL for seed via stage_b arg; pass through a shim
        import benchmark.evaluation.evaluate_mof0 as ev

        ev.PROTOCOL = proto
        card = run_evaluation(args.submission, od, f"noise_repeat_{k}",
                              is_baseline=args.baseline, stages=args.stages,
                              sandbox=not args.no_sandbox, with_qha=False)
        ev.PROTOCOL = PROTOCOL
        cards.append(card)

    est = write_estimate(cards)
    print(json.dumps(est, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
