#!/usr/bin/env python3
"""FairChem training entry point.

FairChem v2 exposes training through the `fairchem -c <yaml>` Hydra CLI.  This
wrapper keeps the public mat-agent API uniform while letting the YAML own the
model-specific runner details.  The YAML may reference the exported environment
variables:

  MAT_AGENT_TRAIN_DATA, MAT_AGENT_VAL_DATA, MAT_AGENT_TEST_DATA,
  MAT_AGENT_OUTPUT_DIR, MAT_AGENT_DEVICE, MAT_AGENT_HEAD,
  MAT_AGENT_MAX_STEPS, MAT_AGENT_BATCH_SIZE.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--train-data", required=True)
    p.add_argument("--val-data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--head", default="")
    p.add_argument("--max-steps", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--test-data", default="")
    return p.parse_args()


def latest_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for pattern in ("*.pt", "*.pth", "*.ckpt"):
        candidates.extend(output_dir.rglob(pattern))
    candidates = [p for p in candidates if p.name != "frozen.pt"]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> int:
    args = parse_args()
    config = Path(args.config)
    train_data = Path(args.train_data)
    val_data = Path(args.val_data)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for label, path in (("config", config), ("train", train_data), ("val", val_data)):
        if not path.is_file():
            print(f"FATAL: {label} not found: {path}", file=sys.stderr)
            return 1

    env = os.environ.copy()
    env.update(
        {
            "MAT_AGENT_TRAIN_DATA": str(train_data),
            "MAT_AGENT_VAL_DATA": str(val_data),
            "MAT_AGENT_TEST_DATA": args.test_data,
            "MAT_AGENT_OUTPUT_DIR": str(out),
            "MAT_AGENT_DEVICE": args.device,
            "MAT_AGENT_HEAD": args.head,
            "MAT_AGENT_MAX_STEPS": str(args.max_steps),
            "MAT_AGENT_BATCH_SIZE": str(args.batch_size),
        }
    )

    log_path = out / "train.log"
    cmd = ["fairchem", "-c", str(config)]
    print("[fairchem] " + " ".join(cmd), file=sys.stderr)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[fairchem] " + " ".join(cmd) + "\n")
        log.flush()
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)

    if result.returncode != 0:
        print(f"FATAL: fairchem training failed with exit {result.returncode}; see {log_path}", file=sys.stderr)
        return 2

    ckpt = latest_checkpoint(out)
    if ckpt is None:
        print(f"FATAL: fairchem training finished but no checkpoint found under {out}", file=sys.stderr)
        return 3

    frozen = out / "frozen.pt"
    if ckpt.resolve() != frozen.resolve():
        shutil.copy2(ckpt, frozen)

    print(f"TRAIN OK (model: {frozen})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
