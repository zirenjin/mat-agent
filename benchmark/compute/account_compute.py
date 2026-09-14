#!/usr/bin/env python3
"""Sum the compute ledger for a benchmark task and check it against policy.

Reads every ``runs/*/run_manifest.json``, converts measured GPU time to
A100-equivalent hours using ``benchmark/compute/policy.yaml``, and reports the
total, per-experiment peaks, model-size compliance, and whether the task budget
is exceeded.

Usage:
    python benchmark/compute/account_compute.py --task mace_mof0_discovery
    python benchmark/compute/account_compute.py --task mace_mof0_discovery --runs-dir runs --json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
POLICY = Path(__file__).resolve().parent / "policy.yaml"


def _load_policy() -> dict:
    # Tiny YAML subset loader to avoid a hard PyYAML dependency for accounting.
    try:
        import yaml  # type: ignore
    except ImportError as err:
        raise SystemExit(
            "PyYAML not available; install pyyaml or read policy.yaml manually. "
            "Accounting requires the parsed policy."
        ) from err
    return yaml.safe_load(POLICY.read_text())


def _num(x: object) -> float:
    if isinstance(x, str):
        return float(x.replace("_", ""))
    return float(x)  # type: ignore[arg-type]


def collect(runs_dir: Path) -> list[dict]:
    out = []
    for mf in sorted(runs_dir.glob("*/run_manifest.json")):
        try:
            out.append({"path": mf.relative_to(REPO).as_posix(), **json.loads(mf.read_text())})
        except json.JSONDecodeError as exc:
            out.append({"path": mf.relative_to(REPO).as_posix(), "_error": str(exc)})
    return out


def account(task: str, runs_dir: Path) -> dict:
    policy = _load_policy()
    common = policy["common"]
    tpol = policy["tasks"][task]
    equiv = common["gpu_equivalence"]

    manifests = collect(runs_dir)
    rows = []
    total = 0.0
    violations: list[str] = []
    max_params = _num(tpol["max_param_count"])
    per_exp_cap = _num(tpol["per_experiment_a100_hours"])

    for m in manifests:
        if "_error" in m:
            violations.append(f"unparseable manifest: {m['path']}")
            continue
        missing = [f for f in common["run_manifest"]["required_fields"] if f not in m]
        if missing:
            violations.append(f"{m['path']}: missing fields {missing}")
        gpu_model = m.get("gpu_model", "other")
        factor = equiv.get(gpu_model, equiv["other"])
        a100h = _num(m.get("measured_gpu_seconds", 0)) / 3600.0 * factor
        total += a100h
        if a100h > per_exp_cap:
            violations.append(
                f"{m['path']}: {a100h:.1f} A100-h exceeds per-experiment cap {per_exp_cap:.0f} "
                f"(needs justification, still counts)"
            )
        if "param_count" in m and _num(m["param_count"]) > max_params:
            violations.append(
                f"{m['path']}: param_count {m['param_count']} exceeds cap {tpol['max_param_count']}"
            )
        rows.append({"path": m["path"], "kind": m.get("kind"), "gpu_model": gpu_model, "a100_hours": round(a100h, 3)})

    if total > _num(tpol["total_a100_hours"]):
        violations.append(
            f"total {total:.1f} A100-h exceeds task budget {tpol['total_a100_hours']}"
        )

    return {
        "task": task,
        "runs": rows,
        "total_a100_hours_used": round(total, 2),
        "total_a100_hours_budget": _num(tpol["total_a100_hours"]),
        "n_manifests": len(rows),
        "ok": not violations,
        "violations": violations,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--task", default="mace_mof0_discovery")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--report", action="store_true", help="alias for a human summary")
    args = ap.parse_args()

    runs_dir = (REPO / args.runs_dir) if not Path(args.runs_dir).is_absolute() else Path(args.runs_dir)
    if not runs_dir.is_dir():
        raise SystemExit(f"runs dir not found: {runs_dir}")

    result = account(args.task, runs_dir)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"task: {result['task']}")
        print(f"manifests: {result['n_manifests']}")
        print(f"A100-hours used: {result['total_a100_hours_used']} / {result['total_a100_hours_budget']}")
        for r in result["runs"]:
            print(f"  {r['a100_hours']:>8.2f} h  {r['kind'] or '?':<10} {r['gpu_model']:<10} {r['path']}")
        if result["violations"]:
            print("\nVIOLATIONS:")
            for v in result["violations"]:
                print(f"  - {v}")
        print(f"\nok: {result['ok']}")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
