#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

usage() {
    cat >&2 << 'USAGE'
Usage: bash models/deepmd/api/evaluate.sh \
    --model <path>          Model checkpoint
    --data <path>           Test extxyz with ref_energy/ref_forces
    --output <path>         Metrics JSON
    [--head <string>]       Multitask head
    [--compute-stress]      Request stress prediction when supported
    [--metrics <mae|rmse|both>] Default: both
    [--device <cpu|cuda>]   Default: cpu
    [--batch-size <int>]    Default: 4
    [--num-workers <int>]   Default: 0
    [--dtype <float32|float64>] Default: float32
    [--seed <int>]          Default: 0
USAGE
    exit 1
}

MODEL=""; DATA=""; OUTPUT=""; HEAD=""; COMPUTE_STRESS=0
METRICS="both"; DEVICE="cpu"; BATCH_SIZE=4; NUM_WORKERS=0; DTYPE="float32"; SEED=0

while [ $# -gt 0 ]; do
    case "$1" in
        --model)          MODEL="$2"; shift 2 ;;
        --data)           DATA="$2"; shift 2 ;;
        --output)         OUTPUT="$2"; shift 2 ;;
        --head)           HEAD="$2"; shift 2 ;;
        --compute-stress) COMPUTE_STRESS=1; shift ;;
        --metrics)        METRICS="$2"; shift 2 ;;
        --device)         DEVICE="$2"; shift 2 ;;
        --batch-size)     BATCH_SIZE="$2"; shift 2 ;;
        --num-workers)    NUM_WORKERS="$2"; shift 2 ;;
        --dtype)          DTYPE="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        -h|--help)        usage ;;
        *) echo "Unknown flag: $1" >&2; usage ;;
    esac
done

[ -z "$MODEL" ]  && { echo "ERROR: --model required" >&2; usage; }
[ -z "$DATA" ]   && { echo "ERROR: --data required" >&2; usage; }
[ -z "$OUTPUT" ] && { echo "ERROR: --output required" >&2; usage; }

[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL" >&2; exit 1; }
[ -f "$DATA" ]  || { echo "ERROR: data not found: $DATA" >&2; exit 1; }

tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT
predictions="$tmp_dir/predictions.extxyz"

stress_args=()
[ "$COMPUTE_STRESS" -eq 0 ] || stress_args=(--compute-stress)

bash "$ROOT/models/deepmd/api/inference.sh" \
    --model "$MODEL" \
    --data "$DATA" \
    --output "$predictions" \
    --head "$HEAD" \
    "${stress_args[@]}" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --dtype "$DTYPE" \
    --seed "$SEED"

pred_path=$(realpath "$predictions" 2>/dev/null || echo "$predictions")
truth_path=$(realpath "$DATA" 2>/dev/null || echo "$DATA")
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
        --model-label "deepmd"

echo "EVALUATE OK"
