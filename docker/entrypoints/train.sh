#!/usr/bin/env bash
set -euo pipefail

ROOT="${MAT_AGENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="$ROOT/mattertune/src:$ROOT:${PYTHONPATH:-}"

usage() {
  cat <<'USAGE'
Usage: train.sh MODEL [MODEL OPTIONS]

Models:
  deepmd    DeepMD/DPA single-branch and multi-head protocols
  sevennet  SevenNet native and continual protocols
  chgnet    CHGNet native objective protocols
  mace      MACE trusted local checkpoint protocols

Run `train.sh MODEL --help` for model-specific flags.
USAGE
}

if [ $# -eq 0 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

exec python "$ROOT/tools/mattertune_train.py" "$@"
