#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROTECTED_PARTS = {"raw", "test", "ood", "hidden", "hidden_test"}
DEFAULT_WRITE_ROOTS = ("playground/runs", "playground/cache", ".tmp", "/tmp")


def is_protected(path: str) -> bool:
    candidate = Path(path)
    return bool(set(candidate.parts) & PROTECTED_PARTS) or candidate.stem in PROTECTED_PARTS


def under_allowed_root(path: str, repo: str, roots: tuple[str, ...]) -> bool:
    repo_path = Path(repo)
    resolved = (Path(path) if Path(path).is_absolute() else repo_path / path).resolve(strict=False)
    for root in roots:
        allowed = (Path(root) if Path(root).is_absolute() else repo_path / root).resolve(strict=False)
        try:
            resolved.relative_to(allowed)
        except ValueError:
            continue
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check dataset read/write paths against sandbox policy.")
    parser.add_argument("path")
    parser.add_argument("--mode", choices=["read", "write"], default="read")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--allowed-write-root", action="append", default=[])
    args = parser.parse_args()

    violations = []
    if args.mode == "write":
        if is_protected(args.path):
            violations.append("write_targets_protected_split_or_raw_data")
        allowed_roots = tuple(args.allowed_write_root) or DEFAULT_WRITE_ROOTS
        if not under_allowed_root(args.path, args.repo, allowed_roots):
            violations.append("write_outside_allowed_roots")

    result = {"ok": not violations, "mode": args.mode, "path": args.path, "violations": violations}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
