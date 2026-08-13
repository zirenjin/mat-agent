#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from _common import emit


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Whether predicted and reference space-group labels match", allow_abbrev=False)
    p.add_argument("--pred", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--output")
    args = p.parse_args()
    emit("space_group_match", args.pred.strip() == args.ref.strip(), args.output, pred=args.pred, ref=args.ref)
