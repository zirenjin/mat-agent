#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: score.sh --scorer NAME --predictions PATH --output PATH [options]

Planned scoring entrypoint. This command is currently a documented placeholder
and exits 64 after argument/help handling.

Options:
  --scorer NAME       Scorer registered in scoring/registry.yaml.
  --predictions PATH  Prediction file produced by predict.sh.
  --labels PATH       Reference labels, usually task-controlled.
  --split NAME        train, val, test, or ood.
  --task-config PATH  Benchmark task metadata.
  --output PATH       JSON score report path.
  -h, --help          Show this help.
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

usage >&2
echo "score.sh: scorer dispatch is not wired yet." >&2
exit 64
