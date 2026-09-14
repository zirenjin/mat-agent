"""Combine Stage A + Stage B into ratios vs baseline and the four-condition verdict.

Design points that answer the Part-5 review:

* A per-endpoint **catastrophic override**: any single endpoint at more than
  ``catastrophic_ratio`` x the baseline error (or a lost space group, or new
  imaginary modes where the baseline had none) forces that MOF to ``worse`` and
  fails the verdict, no matter what the geometric mean says.
* The per-MOF aggregate is a geometric mean of *available* endpoint ratios only;
  null endpoints (no DFT reference) are dropped for candidate and baseline alike.
* ``tol_mof`` / ``tol_ml`` are read from ``noise_estimate.json`` when present
  (measured), else from the protocol defaults (clearly marked provisional).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NOISE = Path(__file__).resolve().parent.parent / "noise_estimate.json"

LOWER_IS_BETTER = {
    "phonon_dos_mae", "phonon_frequency_rmse", "imaginary_mode_count",
    "unit_cell_length_percent_error", "bulk_modulus_relative_error",
    "thermal_expansion_relative_error",
}


def _tolerances(protocol: dict) -> dict:
    tol = {"tol_mof": protocol["tol_mof"], "tol_ml": protocol["tol_ml"],
           "tol_repro": protocol["tol_repro"], "source": "protocol_default (provisional)"}
    if NOISE.is_file():
        n = json.loads(NOISE.read_text())
        tol.update({k: n[k] for k in ("tol_mof", "tol_ml", "tol_repro") if k in n})
        tol["source"] = f"noise_estimate.json ({n.get('n_repeats', '?')} repeats)"
    return tol


def _endpoint_ratio(name: str, cand, base, eps_count: float = 0.5):
    """Return (ratio, catastrophic: bool, note)."""
    if cand is None or base is None:
        return None, False, "unavailable"
    if name == "space_group_match":
        if base and not cand:
            return 10.0, True, "lost space group the baseline retained"
        if cand and not base:
            return 0.5, False, "recovered a space group the baseline lost"
        return 1.0, False, "match state unchanged"
    if name == "imaginary_mode_count":
        r = (cand + eps_count) / (base + eps_count)
        cat = (base <= 0.5 and cand >= 2) or r > 3.0
        return r, cat, f"{cand:.0f} vs {base:.0f} imaginary modes"
    # generic lower-is-better error metric
    r = (1.0 if abs(cand) < 1e-12 else 5.0) if abs(base) < 1e-12 else cand / base
    return r, r > 2.0, f"ratio {r:.3f}"


def _geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and x > 0]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def compare_stage_b(cand_b: dict, base_b: dict, protocol: dict, tol: dict) -> dict:
    out: dict = {}
    for mof, cand in cand_b.items():
        base = base_b.get(mof, {})
        ratios, catastrophic, notes = {}, [], {}
        for name, cval in cand["endpoints"].items():
            bval = base.get("endpoints", {}).get(name)
            r, cat, note = _endpoint_ratio(name, cval, bval)
            ratios[name] = r
            notes[name] = note
            if cat:
                catastrophic.append(name)
        agg = _geomean(list(ratios.values()))
        if catastrophic:
            status = "worse"
        elif math.isnan(agg):
            status = "incomplete"
        elif agg < 1.0 - tol["tol_mof"]:
            status = "better"
        elif agg > 1.0 + tol["tol_mof"]:
            status = "worse"
        else:
            status = "equal"
        out[mof] = {
            "endpoints": cand["endpoints"],
            "baseline_endpoints": base.get("endpoints", {}),
            "endpoint_ratio_vs_baseline": ratios,
            "endpoint_notes": notes,
            "catastrophic_endpoints": catastrophic,
            "aggregate_ratio": None if math.isnan(agg) else agg,
            "status": status,
        }
    return out


def compare_stage_a(cand_a: dict, base_a: dict, tol: dict) -> dict:
    reg = {}
    worst = 0.0
    for k, cv in cand_a["overall"].items():
        bv = base_a["overall"].get(k)
        if bv is None or bv == 0:
            reg[k] = {"candidate": cv, "baseline": bv, "ratio": None}
            continue
        ratio = cv / bv
        reg[k] = {"candidate": cv, "baseline": bv, "ratio": ratio,
                  "regressed": ratio > 1.0 + tol["tol_ml"]}
        worst = max(worst, ratio)
    return {"metrics": reg, "worst_ratio": worst,
            "no_regression": all(not v.get("regressed", False) for v in reg.values())}


def build_verdict(sa_cmp: dict, sb_cmp: dict, repro: dict, compute: dict, tol: dict) -> dict:
    scored = [m for m, v in sb_cmp.items() if v["status"] != "incomplete"]
    n_better = sum(1 for m in scored if sb_cmp[m]["status"] == "better")
    n_worse = sum(1 for m in scored if sb_cmp[m]["status"] == "worse")
    need = math.ceil(0.75 * len(scored)) if scored else 99

    reasons = []
    physics_ok = bool(scored) and n_better >= need and n_worse == 0
    if not scored:
        reasons.append("no MOF produced a complete Stage-B scorecard")
    else:
        reasons.append(f"Stage B: {n_better}/{len(scored)} MOFs better (need >= {need}), {n_worse} worse")
    ml_ok = sa_cmp["no_regression"]
    if not ml_ok:
        reasons.append(f"Stage A regression: worst ratio {sa_cmp['worst_ratio']:.3f} > 1+{tol['tol_ml']}")
    repro_ok = bool(repro.get("checked")) and bool(repro.get("within_tolerance"))
    if not repro.get("checked"):
        reasons.append("reproducibility re-run not performed")
    elif not repro.get("within_tolerance"):
        reasons.append(f"reproducibility drift {repro.get('max_relative_drift')} > {tol['tol_repro']}")
    compute_ok = bool(compute.get("ledger_ok"))
    if not compute_ok:
        reasons.append("compute ledger invalid (benchmark/compute/account_compute.py)")

    beats = physics_ok and ml_ok and repro_ok and compute_ok
    return {
        "beats_baseline": beats,
        "tier": "unverified" if beats else "not_a_win",
        "conditions": {
            "physics_primary": physics_ok,
            "no_ml_regression": ml_ok,
            "reproducible": repro_ok,
            "compute_legal": compute_ok,
        },
        "tolerances_used": tol,
        "reasons": reasons,
    }
