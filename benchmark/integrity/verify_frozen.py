#!/usr/bin/env python3
"""Verify (or regenerate) the frozen-file manifest for the benchmark.

The manifest pins the SHA-256 of every file that must stay immutable while an
agent works on a benchmark task: the evaluator, the scorers, data manifests,
the compute policy, baseline artifacts, the Docker environment, task cards, and
a fixed list of infrastructure tools.

Usage:
    python benchmark/integrity/verify_frozen.py            # verify, exit 2 on drift
    python benchmark/integrity/verify_frozen.py --json     # machine-readable report
    python benchmark/integrity/verify_frozen.py --update   # rewrite the manifest

`--update` is a maintainer action. It should never run inside an agent task.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).resolve().parent / "frozen_manifest.json"

# Glob patterns, relative to the repo root, that define the frozen set.
FROZEN_GLOBS: list[str] = [
    "benchmark/**/*.py",
    "benchmark/**/*.md",
    "benchmark/**/*.yaml",
    "benchmark/**/*.yml",
    "benchmark/**/*.json",
    "benchmark/**/*.sh",
    "benchmark/**/*.toml",  # Harbor task.toml (agent wall-clock, budgets) must be tamper-evident too
    "scoring/**/*.py",
    "scoring/**/*.yaml",
    "scoring/**/*.md",
    "tasks/**/task_card.md",
    "data/registry.yaml",
    "data/mace_mof_0/manifest.json",
    # Frozen baseline+warm-start training config for the Harbor discovery task
    # (benchmark/baselines/mace_mof0/reproduced_baseline_manifest.json records
    # its SHA256 too, redundantly -- both must stay in sync with the file).
    "data/mace_mof_0/runs/mof0_paper_reproduce_force_mh_lr_20260829T124117Z/configs/ft_medium.mof0_paper_reproduce.yml",
    "docker/Dockerfile",
    "docker/entrypoints/*.sh",
    "docker/compose.yaml",
    # Infrastructure tools an agent must not silently reshape.
    "tools/verdict.py",
    "tools/eval_report.py",
    "tools/budget_guard.py",
    "tools/dataset_guard.py",
    "tools/time_logger.py",
    "tools/feedback_logger.py",
    "tools/anomaly_sniffer.py",
    "tools/ase_conventions.py",
]

# Patterns excluded even if matched above (vendored third-party trees change on
# their own cadence and are not part of our integrity surface).
EXCLUDE_GLOBS: list[str] = [
    "scoring/adapters/*/upstream/**",
    "**/__pycache__/**",
    "benchmark/integrity/frozen_manifest.json",  # this file is the ledger, not a frozen input
]

# Files named by FROZEN_GLOBS that are allowed to be absent (created later).
OPTIONAL_MISSING: list[str] = [
    "benchmark/baselines/mace_mof0/baseline_scores.json",
]


def _excluded(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, pat) for pat in EXCLUDE_GLOBS)


def discover() -> list[str]:
    seen: set[str] = set()
    for pat in FROZEN_GLOBS:
        for path in REPO.glob(pat):
            if not path.is_file():
                continue
            rel = path.relative_to(REPO).as_posix()
            if not _excluded(rel):
                seen.add(rel)
    return sorted(seen)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest() -> dict[str, object]:
    entries = {rel: sha256(REPO / rel) for rel in discover()}
    return {
        "manifest_version": 1,
        "description": "SHA-256 of files that must stay immutable during a benchmark task.",
        "frozen_globs": FROZEN_GLOBS,
        "exclude_globs": EXCLUDE_GLOBS,
        "files": entries,
    }


def load_manifest() -> dict[str, object]:
    if not MANIFEST.is_file():
        raise SystemExit(f"frozen manifest missing: {MANIFEST} (run with --update to create it)")
    return json.loads(MANIFEST.read_text())


def verify() -> tuple[bool, dict[str, list[str]]]:
    manifest = load_manifest()
    recorded: dict[str, str] = dict(manifest.get("files", {}))
    current = {rel: sha256(REPO / rel) for rel in discover()}

    modified = sorted(r for r in recorded if r in current and recorded[r] != current[r])
    missing = sorted(
        r for r in recorded if r not in current and r not in OPTIONAL_MISSING
    )
    added = sorted(r for r in current if r not in recorded)

    ok = not (modified or missing or added)
    return ok, {"modified": modified, "missing": missing, "untracked_new": added}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--update", action="store_true", help="rewrite the manifest (maintainer only)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    if args.update:
        MANIFEST.write_text(json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {MANIFEST.relative_to(REPO)} with {len(build_manifest()['files'])} files")
        return 0

    ok, report = verify()
    if args.json:
        print(json.dumps({"ok": ok, **report}, indent=2, sort_keys=True))
    else:
        if ok:
            print("frozen manifest OK")
        else:
            for kind, items in report.items():
                for it in items:
                    print(f"{kind}: {it}")
            print("\nFROZEN INFRASTRUCTURE CHANGED — submission is out of contract.", file=sys.stderr)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
