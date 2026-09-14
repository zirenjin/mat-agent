"""Estimate evaluator noise so tolerances come from measurement, not a guess.

Runs the evaluator N times on the same submission with perturbed seeds, then
turns the spread into ``benchmark/evaluation/noise_estimate.json``:

  tol_mof   -> P90 of the per-MOF aggregate-ratio spread across repeats
  tol_ml    -> P90 relative std of the Stage-A overall metrics
  tol_repro -> max relative drift of any verdict-critical quantity
  catastrophic_ratio -> 1 + 6*sigma of the worst single endpoint (floor 2.0)

Until this has been run against the real baseline, ``aggregate._tolerances``
falls back to the protocol defaults and marks them "provisional".
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "noise_estimate.json"


def _geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def _p90(xs):
    xs = sorted(x for x in xs if x is not None and not math.isnan(x))
    if not xs:
        return None
    k = min(len(xs) - 1, int(math.ceil(0.9 * len(xs))) - 1)
    return xs[k]


def summarize_repeats(scorecards: list[dict]) -> dict:
    """scorecards: >=2 scorecard dicts from evaluate_mof0 for the SAME model."""
    n = len(scorecards)
    if n < 2:
        raise ValueError("need at least 2 repeat scorecards")

    endpoint_rel_std: dict[str, dict[str, float]] = {}
    per_mof_agg_spread: list[float] = []
    worst_endpoint_rel_std = 0.0

    mofs = list((scorecards[0].get("stage_b") or {}).keys())
    for mof in mofs:
        agg_ratios = []
        for card in scorecards:
            ep = (card["stage_b"][mof]["endpoints"])
            vals = [v for v in ep.values() if isinstance(v, (int, float))]
            agg_ratios.append(_geomean([abs(v) + 1e-9 for v in vals]))
        if len([a for a in agg_ratios if not math.isnan(a)]) >= 2:
            m = statistics.mean(a for a in agg_ratios if not math.isnan(a))
            sd = statistics.pstdev(a for a in agg_ratios if not math.isnan(a))
            per_mof_agg_spread.append(sd / m if m else 0.0)

        endpoint_rel_std[mof] = {}
        for name in (scorecards[0]["stage_b"][mof]["endpoints"]):
            series = [c["stage_b"][mof]["endpoints"].get(name) for c in scorecards]
            series = [v for v in series if isinstance(v, (int, float))]
            if len(series) >= 2:
                mean = statistics.mean(series)
                sd = statistics.pstdev(series)
                rel = sd / abs(mean) if mean else 0.0
                endpoint_rel_std[mof][name] = rel
                worst_endpoint_rel_std = max(worst_endpoint_rel_std, rel)

    ml_rel_std = {}
    if scorecards[0].get("stage_a"):
        for k in scorecards[0]["stage_a"]["overall"]:
            series = [c["stage_a"]["overall"].get(k) for c in scorecards]
            series = [v for v in series if isinstance(v, (int, float))]
            if len(series) >= 2 and statistics.mean(series):
                ml_rel_std[k] = statistics.pstdev(series) / abs(statistics.mean(series))

    tol_mof = round(max(_p90(per_mof_agg_spread) or 0.02, 0.01), 4)
    tol_ml = round(max(_p90(list(ml_rel_std.values())) or 0.03, 0.01), 4)
    tol_repro = round(max(worst_endpoint_rel_std, tol_mof), 4)
    catastrophic_ratio = round(max(2.0, 1.0 + 6.0 * worst_endpoint_rel_std), 3)

    return {
        "n_repeats": n,
        "tol_mof": tol_mof,
        "tol_ml": tol_ml,
        "tol_repro": tol_repro,
        "catastrophic_ratio": catastrophic_ratio,
        "measured": {
            "per_mof_aggregate_rel_std": per_mof_agg_spread,
            "endpoint_rel_std": endpoint_rel_std,
            "stage_a_overall_rel_std": ml_rel_std,
            "worst_endpoint_rel_std": worst_endpoint_rel_std,
        },
        "note": "tolerances derived from repeat spread; regenerate whenever the evaluator or protocol changes",
    }


def write_estimate(scorecards: list[dict]) -> dict:
    est = summarize_repeats(scorecards)
    OUT.write_text(json.dumps(est, indent=2, sort_keys=True) + "\n")
    return est
