#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
source "$ROOT/tools/docker_helpers.sh"

usage() {
    cat >&2 << 'USAGE'
Usage: bash models/deepmd/api/train.sh \
    --foundation-model <path> Pretrained checkpoint
    [--config <path>]       Training hyperparameter YAML
    --train-data <path>     Train extxyz
    --val-data <path>       Validation extxyz
    --output-dir <path>     Output directory
    [--replay-data <path>]  Replay extxyz for multihead/multitask
    [--head <string>]       Multitask head
    [--finetune-mode <single|multihead|multitask|lora>]
    [--max-steps <int>]     Default: 4
    [--batch-size <int>]    Default: 4
    [--lr <float>]
    [--weight-decay <float>]
    [--grad-clip <float>]
    [--energy-weight <float>]
    [--forces-weight <float>]
    [--stress-weight <float>]
    [--replay-weight <float>]
    [--device <cpu|cuda>]   Default: cpu
    [--num-workers <int>]   Default: 0
    [--seed <int>]          Default: 0
    [--save-every <int>]
    [--log-every <int>]
USAGE
    exit 1
}

FOUNDATION_MODEL=""; CONFIG=""; TRAIN_DATA=""; VAL_DATA=""; OUTPUT_DIR=""; REPLAY_DATA=""
HEAD=""; FINETUNE_MODE="single"; MAX_STEPS=4; BATCH_SIZE=4
LR=""; WEIGHT_DECAY=""; GRAD_CLIP=""
ENERGY_WEIGHT=""; FORCES_WEIGHT=""; STRESS_WEIGHT=""; REPLAY_WEIGHT=""
DEVICE="cpu"; NUM_WORKERS=0; SEED=0; SAVE_EVERY=""; LOG_EVERY=""

while [ $# -gt 0 ]; do
    case "$1" in
        --foundation-model) FOUNDATION_MODEL="$2"; shift 2 ;;
        --config)           CONFIG="$2"; shift 2 ;;
        --train-data)       TRAIN_DATA="$2"; shift 2 ;;
        --val-data)         VAL_DATA="$2"; shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --replay-data)      REPLAY_DATA="$2"; shift 2 ;;
        --head)             HEAD="$2"; shift 2 ;;
        --finetune-mode)    FINETUNE_MODE="$2"; shift 2 ;;
        --max-steps)        MAX_STEPS="$2"; shift 2 ;;
        --batch-size)       BATCH_SIZE="$2"; shift 2 ;;
        --lr)               LR="$2"; shift 2 ;;
        --weight-decay)     WEIGHT_DECAY="$2"; shift 2 ;;
        --grad-clip)        GRAD_CLIP="$2"; shift 2 ;;
        --energy-weight)    ENERGY_WEIGHT="$2"; shift 2 ;;
        --forces-weight)    FORCES_WEIGHT="$2"; shift 2 ;;
        --stress-weight)    STRESS_WEIGHT="$2"; shift 2 ;;
        --replay-weight)    REPLAY_WEIGHT="$2"; shift 2 ;;
        --device)           DEVICE="$2"; shift 2 ;;
        --num-workers)      NUM_WORKERS="$2"; shift 2 ;;
        --seed)             SEED="$2"; shift 2 ;;
        --save-every)       SAVE_EVERY="$2"; shift 2 ;;
        --log-every)        LOG_EVERY="$2"; shift 2 ;;
        -h|--help)          usage ;;
        *) echo "Unknown flag: $1" >&2; usage ;;
    esac
done

[ -z "$FOUNDATION_MODEL" ] && { echo "ERROR: --foundation-model required" >&2; usage; }
[ -z "$TRAIN_DATA" ] && { echo "ERROR: --train-data required" >&2; usage; }
[ -z "$VAL_DATA" ]   && { echo "ERROR: --val-data required" >&2; usage; }
[ -z "$OUTPUT_DIR" ] && { echo "ERROR: --output-dir required" >&2; usage; }

[ -f "$FOUNDATION_MODEL" ] || { echo "ERROR: foundation model not found: $FOUNDATION_MODEL" >&2; exit 1; }
[ -z "$CONFIG" ]     || [ -f "$CONFIG" ]      || { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
[ -f "$TRAIN_DATA" ] || { echo "ERROR: train data not found: $TRAIN_DATA" >&2; exit 1; }
[ -f "$VAL_DATA" ]   || { echo "ERROR: val data not found: $VAL_DATA" >&2; exit 1; }
[ -z "$REPLAY_DATA" ] || [ -f "$REPLAY_DATA" ] || { echo "ERROR: replay data not found: $REPLAY_DATA" >&2; exit 1; }

export MAT_AGENT_CONFIG="$CONFIG"
export MAT_AGENT_REPLAY_DATA="$REPLAY_DATA"
export MAT_AGENT_FINETUNE_MODE="$FINETUNE_MODE"
export MAT_AGENT_LR="$LR"
export MAT_AGENT_WEIGHT_DECAY="$WEIGHT_DECAY"
export MAT_AGENT_GRAD_CLIP="$GRAD_CLIP"
export MAT_AGENT_ENERGY_WEIGHT="$ENERGY_WEIGHT"
export MAT_AGENT_FORCES_WEIGHT="$FORCES_WEIGHT"
export MAT_AGENT_STRESS_WEIGHT="$STRESS_WEIGHT"
export MAT_AGENT_REPLAY_WEIGHT="$REPLAY_WEIGHT"
export MAT_AGENT_NUM_WORKERS="$NUM_WORKERS"
export MAT_AGENT_SEED="$SEED"
export MAT_AGENT_SAVE_EVERY="$SAVE_EVERY"
export MAT_AGENT_LOG_EVERY="$LOG_EVERY"

run_train_in_container \
    "mat-agent/deepmd:latest" \
    "$FOUNDATION_MODEL" "$TRAIN_DATA" "$VAL_DATA" "$OUTPUT_DIR" "$DEVICE" \
    "$HEAD" "$MAX_STEPS" "$BATCH_SIZE" ""
