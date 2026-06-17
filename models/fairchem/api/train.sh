#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

usage() {
    cat >&2 << 'USAGE'
Usage: bash models/fairchem/api/train.sh \
    --foundation-model <path> Pretrained checkpoint
    [--config <path>]       FairChem training YAML
    --train-data <path>     Train extxyz
    --val-data <path>       Validation extxyz
    --output-dir <path>     Output directory
    [--replay-data <path>]  Replay extxyz for multihead/multitask
    [--head <string>]       Task name (oc20/omat/omol/odac)
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

BACKEND_CONFIG="$CONFIG"
[ -n "$BACKEND_CONFIG" ] || BACKEND_CONFIG="$FOUNDATION_MODEL"

foundation_path=$(realpath "$FOUNDATION_MODEL" 2>/dev/null || echo "$FOUNDATION_MODEL")
config_path=$(realpath "$BACKEND_CONFIG" 2>/dev/null || echo "$BACKEND_CONFIG")
train_path=$(realpath "$TRAIN_DATA" 2>/dev/null || echo "$TRAIN_DATA")
val_path=$(realpath "$VAL_DATA" 2>/dev/null || echo "$VAL_DATA")
output_dir=$(realpath "$OUTPUT_DIR" 2>/dev/null || echo "$OUTPUT_DIR")
foundation_name=$(basename "$foundation_path")
config_name=$(basename "$config_path")
train_name=$(basename "$train_path")
val_name=$(basename "$val_path")
mkdir -p "$output_dir"

gpu_flags=""
[ "$DEVICE" = "cuda" ] && gpu_flags="--gpus all"

extra_mount=""
replay_name=""
if [ -n "$REPLAY_DATA" ]; then
    replay_path=$(realpath "$REPLAY_DATA" 2>/dev/null || echo "$REPLAY_DATA")
    replay_name=$(basename "$replay_path")
    extra_mount="-v $replay_path:/work/data/$replay_name:ro"
fi

docker run --rm \
    $gpu_flags \
    -v "$foundation_path:/work/model/$foundation_name:ro" \
    -v "$config_path:/work/config/$config_name:ro" \
    -v "$train_path:/work/data/$train_name:ro" \
    -v "$val_path:/work/data/$val_name:ro" \
    -v "$output_dir:/work/output" \
    $extra_mount \
    -v "$ROOT/tools:/opt/mat-agent-tools:ro" \
    -e PYTHONPATH="/opt/mat-agent-tools:/opt/internal" \
    -e MAT_AGENT_FOUNDATION_MODEL="/work/model/$foundation_name" \
    -e MAT_AGENT_CONFIG="${CONFIG:+/work/config/$config_name}" \
    -e MAT_AGENT_TRAIN_DATA="/work/data/$train_name" \
    -e MAT_AGENT_VAL_DATA="/work/data/$val_name" \
    -e MAT_AGENT_REPLAY_DATA="${replay_name:+/work/data/$replay_name}" \
    -e MAT_AGENT_OUTPUT_DIR="/work/output" \
    -e MAT_AGENT_DEVICE="$DEVICE" \
    -e MAT_AGENT_HEAD="$HEAD" \
    -e MAT_AGENT_FINETUNE_MODE="$FINETUNE_MODE" \
    -e MAT_AGENT_MAX_STEPS="$MAX_STEPS" \
    -e MAT_AGENT_BATCH_SIZE="$BATCH_SIZE" \
    -e MAT_AGENT_LR="$LR" \
    -e MAT_AGENT_WEIGHT_DECAY="$WEIGHT_DECAY" \
    -e MAT_AGENT_GRAD_CLIP="$GRAD_CLIP" \
    -e MAT_AGENT_ENERGY_WEIGHT="$ENERGY_WEIGHT" \
    -e MAT_AGENT_FORCES_WEIGHT="$FORCES_WEIGHT" \
    -e MAT_AGENT_STRESS_WEIGHT="$STRESS_WEIGHT" \
    -e MAT_AGENT_REPLAY_WEIGHT="$REPLAY_WEIGHT" \
    -e MAT_AGENT_NUM_WORKERS="$NUM_WORKERS" \
    -e MAT_AGENT_SEED="$SEED" \
    -e MAT_AGENT_SAVE_EVERY="$SAVE_EVERY" \
    -e MAT_AGENT_LOG_EVERY="$LOG_EVERY" \
    -w /work \
    "mat-agent/fairchem:latest" \
    python3 /opt/internal/run_train.py \
        --config "/work/config/$config_name" \
        --train-data "/work/data/$train_name" \
        --val-data "/work/data/$val_name" \
        --output-dir /work/output \
        --device "$DEVICE" \
        --head "$HEAD" \
        --max-steps "$MAX_STEPS" \
        --batch-size "$BATCH_SIZE" \
        --test-data ""
