#!/usr/bin/env python3
from __future__ import annotations

from _common import emit, scalar_parser


if __name__ == "__main__":
    args = scalar_parser("Percent deviation of equilibrium unit-cell length relative to reference").parse_args()
    if args.ref == 0:
        raise SystemExit("--ref must be non-zero")
    emit("unit_cell_length_percent_error", 100.0 * (args.pred - args.ref) / args.ref, args.output, pred=args.pred, ref=args.ref)
