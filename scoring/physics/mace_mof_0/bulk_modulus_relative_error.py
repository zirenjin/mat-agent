#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import emit, scalar_parser

if __name__ == "__main__":
    args = scalar_parser("Relative error in bulk modulus").parse_args()
    if args.ref == 0:
        raise SystemExit("--ref must be non-zero")
    emit("bulk_modulus_relative_error", (args.pred - args.ref) / args.ref, args.output, pred=args.pred, ref=args.ref)
