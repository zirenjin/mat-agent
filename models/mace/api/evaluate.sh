#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

usage() {
    cat >&2 << 'USAGE'
Usage: bash models/mace/api/evaluate.sh \
    --predictions <path>    Output extxyz from inference.sh
    --truth <path>          Ground-truth extxyz
    --output <path>         Path for verdict JSON
    [--metrics <list>]      Reserved for metric selection
USAGE
    exit 1
}

PREDICTIONS=""; TRUTH=""; OUTPUT=""; METRICS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --predictions) PREDICTIONS="$2"; shift 2 ;;
        --truth)       TRUTH="$2"; shift 2 ;;
        --output)      OUTPUT="$2"; shift 2 ;;
        --metrics)     METRICS="$2"; shift 2 ;;
        -h|--help)     usage ;;
        *) echo "Unknown flag: $1" >&2; usage ;;
    esac
done

[ -z "$PREDICTIONS" ] && { echo "ERROR: --predictions required" >&2; usage; }
[ -z "$TRUTH" ]       && { echo "ERROR: --truth required" >&2; usage; }
[ -z "$OUTPUT" ]      && { echo "ERROR: --output required" >&2; usage; }

[ -f "$PREDICTIONS" ] || { echo "ERROR: predictions not found: $PREDICTIONS" >&2; exit 1; }
[ -f "$TRUTH" ]       || { echo "ERROR: truth not found: $TRUTH" >&2; exit 1; }

pred_path=$(realpath "$PREDICTIONS" 2>/dev/null || echo "$PREDICTIONS")
truth_path=$(realpath "$TRUTH" 2>/dev/null || echo "$TRUTH")
output_dir=$(dirname "$(realpath "$OUTPUT" 2>/dev/null || echo "$OUTPUT")")
output_name=$(basename "$OUTPUT")
mkdir -p "$output_dir"

docker run --rm \
    -v "$pred_path:/work/predictions.extxyz:ro" \
    -v "$truth_path:/work/truth.extxyz:ro" \
    -v "$output_dir:/work/output" \
    -v "$ROOT/tools:/opt/mat-agent-tools:ro" \
    -e PYTHONPATH="/opt/mat-agent-tools" \
    -w /work \
    "mat-agent/base:latest" \
    python3 /opt/mat-agent-tools/verdict.py \
        --predictions /work/predictions.extxyz \
        --truth /work/truth.extxyz \
        --output "/work/output/$output_name" \
        --model-label "mace"

echo "EVALUATE OK"
