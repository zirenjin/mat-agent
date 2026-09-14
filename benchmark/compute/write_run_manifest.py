#!/usr/bin/env python3
"""Emit runs/<run_id>/run_manifest.json for the compute ledger.

Wrap every training / inference run:

    python benchmark/compute/write_run_manifest.py \
        --run-id ft_003 --kind train \
        --started 2026-09-05T10:00:00Z --ended 2026-09-05T13:30:00Z \
        --gpu-model A100-80GB --gpu-count 4 \
        --measured-gpu-seconds 50400 \
        --config workspace/configs/ft_003.yaml \
        --param-count 4700000

Fields map 1:1 to benchmark/compute/policy.yaml -> common.run_manifest.required_fields.
``--measured-gpu-seconds`` must be real (wall_seconds * gpu_count), not estimated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FOUNDATION = REPO / "data/mace_mof_0/models/mace_agnesi_medium.model"


def _sha256(p: Path) -> str:
    if not p.is_file():
        return "MISSING"
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_rev(path: Path) -> str:
    try:
        r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "workspace"],
                               capture_output=True, text=True, check=False).stdout.strip()
        return r + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "content:" + _sha256(path)[:16] if path.exists() else "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--kind", required=True, choices=["train", "infer", "eval-proxy", "other"])
    ap.add_argument("--started", required=True)
    ap.add_argument("--ended", required=True)
    ap.add_argument("--gpu-model", required=True)
    ap.add_argument("--gpu-count", type=int, required=True)
    ap.add_argument("--measured-gpu-seconds", type=float, required=True)
    ap.add_argument("--config", required=True, help="resolved config / argv file actually used")
    ap.add_argument("--param-count", type=int, required=True)
    ap.add_argument("--foundation-checkpoint", default=str(FOUNDATION))
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    run_dir = REPO / args.runs_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = Path(args.config)
    manifest = {
        "run_id": args.run_id,
        "kind": args.kind,
        "started_utc": args.started,
        "ended_utc": args.ended,
        "gpu_model": args.gpu_model,
        "gpu_count": args.gpu_count,
        "measured_gpu_seconds": args.measured_gpu_seconds,
        "resolved_config_path": args.config,
        "resolved_config_sha256": _sha256(cfg if cfg.is_absolute() else REPO / cfg),
        "foundation_checkpoint_sha256": _sha256(Path(args.foundation_checkpoint)),
        "code_rev": _code_rev(REPO / "workspace"),
        "param_count": args.param_count,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {run_dir / 'run_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
