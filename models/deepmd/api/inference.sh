#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$ROOT/tools/docker_helpers.sh"

usage() {
    cat >&2 << 'USAGE'
Usage: bash models/deepmd/api/inference.sh \
    --model-path <path>     Model checkpoint path (.pt)
    --data <path>           Input extxyz
    --output <path>         Output extxyz with pred_energy/pred_forces
    --device <cpu|cuda>     Compute device
    [--config <path>]       Alias for --model-path
    [--head <string>]       Multitask head (e.g. H2O_H2O_PD)
    [--batch-size <int>]    Default: 4
    [--dtype <float32|float64>] Default: float32
USAGE
    exit 1
}

MODEL_PATH=""; DATA=""; OUTPUT=""; HEAD=""; DEVICE="cpu"; BATCH_SIZE=4; DTYPE="float32"

while [ $# -gt 0 ]; do
    case "$1" in
        --model-path|--config) MODEL_PATH="$2"; shift 2 ;;
        --data)               DATA="$2"; shift 2 ;;
        --output)             OUTPUT="$2"; shift 2 ;;
        --head)               HEAD="$2"; shift 2 ;;
        --device)             DEVICE="$2"; shift 2 ;;
        --batch-size)         BATCH_SIZE="$2"; shift 2 ;;
        --dtype)              DTYPE="$2"; shift 2 ;;
        -h|--help)            usage ;;
        *) echo "Unknown flag: $1" >&2; usage ;;
    esac
done

[ -z "$MODEL_PATH" ] && { echo "ERROR: --model-path required (--config is accepted as an alias)" >&2; usage; }
[ -z "$DATA" ]       && { echo "ERROR: --data required" >&2; usage; }
[ -z "$OUTPUT" ]     && { echo "ERROR: --output required" >&2; usage; }

[ -f "$MODEL_PATH" ] || { echo "ERROR: model not found: $MODEL_PATH" >&2; exit 1; }
[ -f "$DATA" ]       || { echo "ERROR: data not found: $DATA" >&2; exit 1; }

run_in_container \
    "mat-agent/deepmd:latest" \
    "$MODEL_PATH" "$DATA" "$OUTPUT" "$DEVICE" "$HEAD" "$BATCH_SIZE"
