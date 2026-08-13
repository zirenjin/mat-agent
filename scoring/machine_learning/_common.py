#!/usr/bin/env python3
"""Small CLI helpers shared by scalar scoring metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Callable
from pathlib import Path


def finite_float(value: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise argparse.ArgumentTypeError(f"non-finite value: {value}")
    return out


def read_vector(path: str | None, column: str | None, literal: str | None) -> list[float]:
    if literal is not None:
        values = [finite_float(part.strip()) for part in literal.split(",") if part.strip()]
        if not values:
            raise SystemExit("literal vector is empty")
        return values
    if path is None:
        raise SystemExit("provide either --values or --file")
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"file not found: {path}")
    if p.suffix == ".json":
        data = json.loads(p.read_text())
        if column:
            data = data[column]
        return [finite_float(str(x)) for x in data]
    with p.open(newline="") as handle:
        reader = csv.DictReader(handle) if column else csv.reader(handle)
        values: list[float] = []
        for row in reader:
            item = row[column] if column else row[0]
            values.append(finite_float(str(item)))
    if not values:
        raise SystemExit(f"no values read from {path}")
    return values


def paired_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, allow_abbrev=False)
    p.add_argument("--true-file")
    p.add_argument("--pred-file")
    p.add_argument("--true-column")
    p.add_argument("--pred-column")
    p.add_argument("--y-true", help="Comma-separated true values")
    p.add_argument("--y-pred", help="Comma-separated predicted values")
    p.add_argument("--output", help="Optional JSON output path")
    return p


def paired_values(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    y_true = read_vector(args.true_file, args.true_column, args.y_true)
    y_pred = read_vector(args.pred_file, args.pred_column, args.y_pred)
    if len(y_true) != len(y_pred):
        raise SystemExit(f"length mismatch: true={len(y_true)} pred={len(y_pred)}")
    if not y_true:
        raise SystemExit("empty input")
    return y_true, y_pred


def emit(metric: str, value: float, n: int, output: str | None = None, **extra: object) -> None:
    payload = {"metric": metric, "value": value, "n": n, **extra}
    text = json.dumps(payload, sort_keys=True)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text + "\n")
    print(text)


def run_paired(metric: str, description: str, fn: Callable[[list[float], list[float]], float]) -> int:
    args = paired_parser(description).parse_args()
    y_true, y_pred = paired_values(args)
    emit(metric, fn(y_true, y_pred), len(y_true), args.output)
    return 0
