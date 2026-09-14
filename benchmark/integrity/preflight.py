#!/usr/bin/env python3
"""Gate that must pass before an autonomous run starts and before a submission is scored.

Checks, in order:
  1. frozen manifest intact              (verify_frozen.py)
  2. discovery splits built + sequestered file present and not group-readable
  3. compute ledger valid so far        (account_compute.py)
  4. (submission mode) the submission validates against the contract

Exit 0 = clear to proceed. Non-zero = blockers printed.
"""
from __future__ import annotations

import argparse
import json
import stat
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _run(mod: list[str]) -> tuple[int, str]:
    p = subprocess.run([sys.executable, *mod], capture_output=True, text=True, check=False)
    return p.returncode, (p.stdout + p.stderr).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--submission", help="also validate this submission descriptor")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    blockers: list[str] = []

    rc, out = _run([str(REPO / "benchmark/integrity/verify_frozen.py")])
    if rc != 0:
        blockers.append(f"frozen manifest changed:\n{out}")

    seq = REPO / "data/mace_mof_0/discovery/_sequestered/sequestered_ml.xyz"
    dev = REPO / "data/mace_mof_0/discovery/dev_train.xyz"
    if not dev.is_file() or not seq.is_file():
        blockers.append("discovery splits missing -- run benchmark/data/build_discovery_splits.py")
    elif stat.S_IMODE(seq.stat().st_mode) & (stat.S_IRGRP | stat.S_IROTH):
        blockers.append(f"sequestered file is group/other-readable: {seq} (chmod 400)")

    rc, out = _run([str(REPO / "benchmark/compute/account_compute.py"),
                    "--task", "mace_mof0_discovery", "--json"])
    try:
        led = json.loads(out)
        if not led.get("ok"):
            blockers.append(f"compute ledger invalid: {led.get('violations')}")
    except json.JSONDecodeError:
        blockers.append(f"account_compute did not return JSON: {out}")

    if args.submission:
        sys.path.insert(0, str(REPO))
        from benchmark.evaluation._lib.submission import SubmissionError, load_and_validate

        try:
            load_and_validate(args.submission)
        except SubmissionError as exc:
            blockers.append(f"submission invalid: {exc}")

    result = {"ok": not blockers, "blockers": blockers}
    print(json.dumps(result, indent=2) if args.json else
          ("preflight OK" if not blockers else "\n".join(f"BLOCKER: {b}" for b in blockers)))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
