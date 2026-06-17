#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$ROOT/tools/docker_helpers.sh"

usage() {
    cat >&2 << 'USAGE'
Usage: bash models/fairchem/api/inference.sh \
    --model-path <path>     Model checkpoint path (.pt)
    --data <path>           Input extxyz
    --output <path>         Output extxyz with pred_energy/pred_forces
    --head <string>         REQUIRED. Task name: oc20/omat/omol/odac
    --device <cpu|cuda>     Compute device
    [--config <path>]       Alias for --model-path
    [--batch-size <int>]    Default: 4
USAGE
    exit 1
}

MODEL_PATH=""; DATA=""; OUTPUT=""; HEAD=""; DEVICE="cpu"; BATCH_SIZE=4

while [ $# -gt 0 ]; do
    case "$1" in
        --model-path|--config) MODEL_PATH="$2"; shift 2 ;;
        --data)               DATA="$2"; shift 2 ;;
        --output)             OUTPUT="$2"; shift 2 ;;
        --head)               HEAD="$2"; shift 2 ;;
        --device)             DEVICE="$2"; shift 2 ;;
        --batch-size)         BATCH_SIZE="$2"; shift 2 ;;
        -h|--help)            usage ;;
        *) echo "Unknown flag: $1" >&2; usage ;;
    esac
done

[ -z "$MODEL_PATH" ] && { echo "ERROR: --model-path required (--config is accepted as an alias)" >&2; usage; }
[ -z "$DATA" ]       && { echo "ERROR: --data required" >&2; usage; }
[ -z "$OUTPUT" ]     && { echo "ERROR: --output required" >&2; usage; }
[ -z "$HEAD" ]       && { echo "ERROR: --head required (oc20/omat/omol/odac)" >&2; usage; }

[ -f "$MODEL_PATH" ] || { echo "ERROR: model not found: $MODEL_PATH" >&2; exit 1; }
[ -f "$DATA" ]       || { echo "ERROR: data not found: $DATA" >&2; exit 1; }

run_in_container \
    "mat-agent/fairchem:latest" \
    "$MODEL_PATH" "$DATA" "$OUTPUT" "$DEVICE" "$HEAD" "$BATCH_SIZE"
