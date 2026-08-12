#!/usr/bin/env python3
"""CLI helpers for lightweight physics metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def finite_float(value: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise argparse.ArgumentTypeError(f"non-finite value: {value}")
    return out


def emit(metric: str, value: float | bool, output: str | None = None, **extra: object) -> None:
    payload = {"metric": metric, "value": value, **extra}
    text = json.dumps(payload, sort_keys=True)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text + "\n")
    print(text)


def read_table(path: str) -> list[dict[str, str]]:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"file not found: {path}")
    with p.open(newline="") as handle:
        return list(csv.DictReader(handle))


def scalar_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, allow_abbrev=False)
    p.add_argument("--pred", type=finite_float, required=True)
    p.add_argument("--ref", type=finite_float, required=True)
    p.add_argument("--output")
    return p
