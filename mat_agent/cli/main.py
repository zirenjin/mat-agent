#!/usr/bin/env python3
"""Minimal mat-agent command dispatcher."""
from __future__ import annotations

import argparse
import json
from mat_agent.planes.data.inspect import inspect_extxyz
from mat_agent.planes.data.validate import validate_extxyz
from mat_agent.planes.compute.failures import classify_log


def main() -> int:
    parser = argparse.ArgumentParser(description="Materials CodeAct scaffold utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_inspect = sub.add_parser("inspect-extxyz")
    p_inspect.add_argument("path")
    p_validate = sub.add_parser("validate-extxyz")
    p_validate.add_argument("path")
    p_validate.add_argument("--no-labels", action="store_true")
    p_log = sub.add_parser("classify-log")
    p_log.add_argument("path")
    args = parser.parse_args()
    if args.cmd == "inspect-extxyz":
        payload = inspect_extxyz(args.path)
    elif args.cmd == "validate-extxyz":
        payload = validate_extxyz(args.path, require_labels=not args.no_labels)
    elif args.cmd == "classify-log":
        payload = classify_log(args.path)
    else:
        raise AssertionError(args.cmd)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
