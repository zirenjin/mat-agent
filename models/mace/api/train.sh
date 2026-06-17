#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$ROOT/tools/docker_helpers.sh"

usage() {
    cat >&2 << 'USAGE'
Usage: bash models/mace/api/train.sh \
    --model-path <path>     Pretrained checkpoint path (.model)
    --train-data <path>     Train extxyz
    --val-data <path>       Validation extxyz
    --output-dir <path>     Output directory
    --device <cpu|cuda>     Compute device
    [--config <path>]       Alias for --model-path
    [--max-steps <int>]     Default: 4
    [--batch-size <int>]    Default: 4
    [--test-data <path>]    Optional post-train test extxyz
USAGE
    exit 1
}

MODEL_PATH=""; TRAIN_DATA=""; VAL_DATA=""; OUTPUT_DIR=""; DEVICE="cpu"
MAX_STEPS=4; BATCH_SIZE=4; TEST_DATA=""

while [ $# -gt 0 ]; do
    case "$1" in
        --model-path|--config) MODEL_PATH="$2"; shift 2 ;;
        --train-data)         TRAIN_DATA="$2"; shift 2 ;;
        --val-data)           VAL_DATA="$2"; shift 2 ;;
        --output-dir)         OUTPUT_DIR="$2"; shift 2 ;;
        --device)             DEVICE="$2"; shift 2 ;;
        --max-steps)          MAX_STEPS="$2"; shift 2 ;;
        --batch-size)         BATCH_SIZE="$2"; shift 2 ;;
        --test-data)          TEST_DATA="$2"; shift 2 ;;
        -h|--help)            usage ;;
        *) echo "Unknown flag: $1" >&2; usage ;;
    esac
done

[ -z "$MODEL_PATH" ]  && { echo "ERROR: --model-path required (--config is accepted as an alias)" >&2; usage; }
[ -z "$TRAIN_DATA" ]  && { echo "ERROR: --train-data required" >&2; usage; }
[ -z "$VAL_DATA" ]    && { echo "ERROR: --val-data required" >&2; usage; }
[ -z "$OUTPUT_DIR" ]  && { echo "ERROR: --output-dir required" >&2; usage; }

[ -f "$MODEL_PATH" ]  || { echo "ERROR: model not found: $MODEL_PATH" >&2; exit 1; }
[ -f "$TRAIN_DATA" ]  || { echo "ERROR: train data not found: $TRAIN_DATA" >&2; exit 1; }
[ -f "$VAL_DATA" ]    || { echo "ERROR: val data not found: $VAL_DATA" >&2; exit 1; }

run_train_in_container \
    "mat-agent/mace:latest" \
    "$MODEL_PATH" "$TRAIN_DATA" "$VAL_DATA" "$OUTPUT_DIR" "$DEVICE" \
    "" "$MAX_STEPS" "$BATCH_SIZE" "$TEST_DATA"
