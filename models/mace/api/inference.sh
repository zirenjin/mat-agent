#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$ROOT/tools/docker_helpers.sh"

usage() {
    cat >&2 << 'USAGE'
Usage: bash models/mace/api/inference.sh \
    --model <path>          Model checkpoint (.model)
    --data <path>           Input extxyz
    --output <path>         Output extxyz
    [--head <string>]       Accepted for API uniformity
    [--compute-stress]      Request stress prediction when supported
    [--device <cpu|cuda>]   Default: cpu
    [--batch-size <int>]    Default: 4
    [--num-workers <int>]   Default: 0
    [--dtype <float32|float64>] Default: float32
    [--seed <int>]          Default: 0
    [--uncertainty]        Enable MACE MHC uncertainty
    [--save-embeddings]    Save node embeddings when supported
USAGE
    exit 1
}

MODEL=""; DATA=""; OUTPUT=""; HEAD=""; COMPUTE_STRESS=0; UNCERTAINTY=0; SAVE_EMBEDDINGS=0
DEVICE="cpu"; BATCH_SIZE=4; NUM_WORKERS=0; DTYPE="float32"; SEED=0

while [ $# -gt 0 ]; do
    case "$1" in
        --model)          MODEL="$2"; shift 2 ;;
        --data)           DATA="$2"; shift 2 ;;
        --output)         OUTPUT="$2"; shift 2 ;;
        --head)           HEAD="$2"; shift 2 ;;
        --compute-stress) COMPUTE_STRESS=1; shift ;;
        --device)         DEVICE="$2"; shift 2 ;;
        --batch-size)     BATCH_SIZE="$2"; shift 2 ;;
        --num-workers)    NUM_WORKERS="$2"; shift 2 ;;
        --dtype)          DTYPE="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        --uncertainty)    UNCERTAINTY=1; shift ;;
        --save-embeddings) SAVE_EMBEDDINGS=1; shift ;;
        -h|--help)        usage ;;
        *) echo "Unknown flag: $1" >&2; usage ;;
    esac
done

[ -z "$MODEL" ]  && { echo "ERROR: --model required" >&2; usage; }
[ -z "$DATA" ]   && { echo "ERROR: --data required" >&2; usage; }
[ -z "$OUTPUT" ] && { echo "ERROR: --output required" >&2; usage; }

[ -f "$MODEL" ]  || { echo "ERROR: model not found: $MODEL" >&2; exit 1; }
[ -f "$DATA" ]   || { echo "ERROR: data not found: $DATA" >&2; exit 1; }

export MAT_AGENT_COMPUTE_STRESS="$COMPUTE_STRESS"
export MAT_AGENT_NUM_WORKERS="$NUM_WORKERS"
export MAT_AGENT_DTYPE="$DTYPE"
export MAT_AGENT_SEED="$SEED"
export MAT_AGENT_UNCERTAINTY="$UNCERTAINTY"
export MAT_AGENT_SAVE_EMBEDDINGS="$SAVE_EMBEDDINGS"

extra_args=()
[ "$UNCERTAINTY" -eq 0 ] || extra_args+=(--uncertainty)
[ "$SAVE_EMBEDDINGS" -eq 0 ] || extra_args+=(--save-embeddings)
[ -z "$DTYPE" ] || extra_args+=(--dtype "$DTYPE")

run_in_container \
    "mat-agent/mace:latest" \
    "$MODEL" "$DATA" "$OUTPUT" "$DEVICE" "$HEAD" "$BATCH_SIZE" "${extra_args[*]}"
