#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: predict.sh --checkpoint CHECKPOINT --data DATA --output OUTPUT [options]

Planned MatterTune prediction entrypoint. This command is currently a documented
placeholder and exits 64 after argument/help handling.

Options:
  --checkpoint PATH  MatterTune checkpoint produced by train.sh.
  --data PATH        Structures to evaluate.
  --output PATH      Prediction output path.
  --device DEVICE    Runtime device.
  --batch-size N     Inference batch size.
  --head NAME        Head to use for multi-head checkpoints.
  -h, --help         Show this help.
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

usage >&2
echo "predict.sh: MatterTune prediction is not wired yet." >&2
exit 64
