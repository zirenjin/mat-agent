#!/usr/bin/env python3
"""Unified evaluator for the MACE-MP-MOF0 Autonomous Discovery task.

The ONLY component that produces a reportable number. It consumes a submission
descriptor (a checkpoint + a declared inference function), never the agent's
training code, then runs every stage of ``benchmark/evaluation/protocol.md``
itself:

  submission -> validate -> sandboxed candidate inference
             -> Stage A (ML)  + Stage B (physics)
             -> compare vs the frozen baseline scorecard
             -> four-condition verdict -> scorecard.json

Every metric value comes from the frozen scorers in ``scoring/`` via
``_lib.metrics.run_scorer``. Nothing here re-implements a metric.

Runtime deps (evaluation image only): torch, mace-torch, phonopy, spglib, ase.
This file imports them lazily so ``--help`` and ``--compare`` work anywhere.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmark.evaluation._lib import aggregate  # noqa: E402
from benchmark.evaluation._lib.submission import SubmissionError, load_and_validate  # noqa: E402

BASELINE_SCORECARD = Path(__file__).resolve().parent / "baselines_mace_mof0_baseline_scores.json"
DEFAULT_BASELINE_SCORECARD = REPO / "benchmark/baselines/mace_mof0/baseline_scores.json"

# ---- FROZEN PROTOCOL CONSTANTS (see protocol.md) --------------------------
PROTOCOL = {
    "schema_version": 1,
    "seed": 2024,
    # tolerances -- provisional until noise_estimate.json exists (see aggregate._tolerances)
    "tol_mof": 0.03,
    "tol_ml": 0.05,
    "tol_repro": 0.05,
    # relaxation
    "relax_fmax_eV_per_A": 1e-4,
    "relax_max_steps": 500,
    # phonons
    "phonon_displacement_A": 0.01,
    "phonon_mesh": [12, 12, 12],
    "supercell_min_length_A": 8.0,
    "supercell_cap": 2,
    "imaginary_threshold_THz": -0.1,
    "dos_points": 1200,
    "dos_sigma_THz": 0.3,
    # EOS / bulk modulus
    "eos_strain_grid": [-0.02, -0.01, 0.0, 0.01, 0.02],
    "eos_fmax_eV_per_A": 1e-3,
    "eos_max_steps": 200,
    # QHA (off unless --with-qha)
    "qha_strains": [-0.04, -0.02, 0.0, 0.02, 0.04],
    "qha_reference_T_K": 300.0,
    # cost guards
    "stage_b_timeout_s": 21600,
    "repro_repeats": 3,
}

MOFS = ("MOF-5", "UiO-66", "MOF-74", "MIL-53")


def _utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _descriptor(validated: dict) -> dict:
    inf = validated["inference"]
    return {
        "kind": inf["kind"],
        "checkpoint": str(validated["checkpoint_path"]),
        "dtype": inf.get("dtype", "float64"),
        "head": inf.get("head"),
        "entrypoint": inf.get("entrypoint"),
        "device": inf.get("device", "cpu"),
    }


def _compute_ledger() -> dict:
    script = REPO / "benchmark/compute/account_compute.py"
    try:
        proc = subprocess.run([sys.executable, str(script), "--task", "mace_mof0_discovery", "--json"],
                              capture_output=True, text=True, check=False)
        data = json.loads(proc.stdout or "{}")
        return {"ledger_ok": bool(data.get("ok")), "total_a100_hours_used": data.get("total_a100_hours_used"),
                "violations": data.get("violations", [])}
    except Exception as exc:  # noqa: BLE001
        return {"ledger_ok": False, "violations": [f"account_compute failed: {exc}"]}


def run_evaluation(submission_path: str, out_dir: Path, label: str, *, is_baseline: bool,
                   stages: str, sandbox: bool, with_qha: bool) -> dict:
    validated = load_and_validate(submission_path, is_baseline=is_baseline)
    desc = _descriptor(validated)
    out_dir.mkdir(parents=True, exist_ok=True)

    scorecard: dict = {
        "schema_version": 1,
        "generated_utc": _utc(),
        "evaluator_rev": _git_rev(),
        "protocol_constants": PROTOCOL,
        "subject": {
            "label": label,
            "checkpoint_sha256": validated["checkpoint_sha256"],
            "is_baseline": is_baseline,
            "inference_kind": desc["kind"],
        },
        "submission_validation": {
            "external_inputs": validated["external_inputs"],
            "training_data_report": validated["training_data_report"],
        },
        "stage_a": None,
        "stage_b": None,
    }

    if "a" in stages:
        from benchmark.evaluation._lib.stage_a import run_stage_a

        scorecard["stage_a"] = run_stage_a(desc, out_dir / "stage_a", sandbox=sandbox)
    if "b" in stages:
        from benchmark.evaluation._lib.stage_b import run_stage_b

        scorecard["stage_b"] = run_stage_b(desc, out_dir / "stage_b", mofs=list(MOFS),
                                           sandbox=sandbox, protocol=PROTOCOL,
                                           seed=PROTOCOL["seed"], with_qha=with_qha)

    (out_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n")
    return scorecard


def compare(candidate: dict, baseline: dict, *, repro: dict | None = None) -> dict:
    tol = aggregate._tolerances(PROTOCOL)
    sa = aggregate.compare_stage_a(candidate["stage_a"], baseline["stage_a"], tol) \
        if candidate.get("stage_a") and baseline.get("stage_a") else {"no_regression": False, "worst_ratio": None, "metrics": {}}
    sb = aggregate.compare_stage_b(candidate["stage_b"] or {}, baseline["stage_b"] or {}, PROTOCOL, tol)
    compute = _compute_ledger()
    repro = repro or {"checked": False}
    verdict = aggregate.build_verdict(sa, sb, repro, compute, tol)
    return {
        "schema_version": 1,
        "candidate_label": candidate["subject"]["label"],
        "baseline_label": baseline["subject"]["label"],
        "stage_a_comparison": sa,
        "stage_b_comparison": sb,
        "compute": compute,
        "reproducibility": repro,
        "verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("score", help="run the evaluator on a submission")
    ev.add_argument("--submission", required=True)
    ev.add_argument("--out-dir", required=True)
    ev.add_argument("--label", default="candidate")
    ev.add_argument("--baseline", action="store_true", help="this submission IS the human baseline")
    ev.add_argument("--stages", default="ab", help="subset of 'ab'")
    ev.add_argument("--with-qha", action="store_true")
    ev.add_argument("--no-sandbox", action="store_true", help="TEST ONLY -- disables candidate isolation")
    ev.add_argument("--baseline-scorecard", default=str(DEFAULT_BASELINE_SCORECARD))
    ev.add_argument("--compare-to-baseline", action="store_true")

    cp = sub.add_parser("compare", help="verdict from two existing scorecards")
    cp.add_argument("--candidate", required=True)
    cp.add_argument("--baseline", required=True)
    cp.add_argument("--repro", help="optional reproducibility JSON {checked, within_tolerance, max_relative_drift}")

    args = ap.parse_args()

    if args.cmd == "compare":
        cand = json.loads(Path(args.candidate).read_text())
        base = json.loads(Path(args.baseline).read_text())
        repro = json.loads(Path(args.repro).read_text()) if args.repro else None
        print(json.dumps(compare(cand, base, repro=repro), indent=2, sort_keys=True))
        return 0

    try:
        card = run_evaluation(args.submission, Path(args.out_dir), args.label,
                              is_baseline=args.baseline, stages=args.stages,
                              sandbox=not args.no_sandbox, with_qha=args.with_qha)
    except SubmissionError as exc:
        print(json.dumps({"rejected": True, "reason": str(exc)}, indent=2))
        return 3

    print(f"scorecard -> {args.out_dir}/scorecard.json")
    if args.compare_to_baseline and not args.baseline:
        base = json.loads(Path(args.baseline_scorecard).read_text())
        verdict = compare(card, base)
        (Path(args.out_dir) / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
        print(json.dumps(verdict["verdict"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
