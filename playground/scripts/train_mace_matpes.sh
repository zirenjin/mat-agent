#!/usr/bin/env bash
# Fine-tune MACE on a MATPES slice through the unified mat-agent API.
#
# Defaults are intentionally small enough for an end-to-end smoke run. Scale
# TRAIN_N/VAL_N/TEST_N/EPOCHS after the pipeline is verified.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-pbe}"
FOUNDATION_MODEL="${FOUNDATION_MODEL:-/root/.cache/mace/20231203mace128L1_epoch199model}"
DEVICE="${DEVICE:-cuda}"

TRAIN_N="${TRAIN_N:-800}"
VAL_N="${VAL_N:-100}"
TEST_N="${TEST_N:-100}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-0.005}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-7}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1}"
FORCES_WEIGHT="${FORCES_WEIGHT:-1000}"
SEED="${SEED:-1234}"
STREAMING="${STREAMING:-0}"
SAMPLE_MODE="${SAMPLE_MODE:-head}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
USE_VPN_PROXY="${USE_VPN_PROXY:-1}"
VPN_PROXY_URL="${VPN_PROXY_URL:-socks5h://127.0.0.1:1080}"
DOCKER_NETWORK_ARGS="${DOCKER_NETWORK_ARGS:-}"
PREP_ONLY="${PREP_ONLY:-0}"

RUN_NAME="${RUN_NAME:-mace_matpes_${CONFIG}_smoke}"
DATA_DIR="data/matpes/${CONFIG}"
RUN_DIR="playground/runs/${RUN_NAME}"

mkdir -p "$RUN_DIR"

if [ ! -f "$DATA_DIR/train.extxyz" ] || [ ! -f "$DATA_DIR/val.extxyz" ]; then
  echo "[matpes] preparing data: config=$CONFIG train=$TRAIN_N val=$VAL_N test=$TEST_N"
  STAGE_DATA="$(mktemp -d /tmp/mat-agent-matpes.XXXXXX)"
  chmod a+rwx "$STAGE_DATA"
  if [ -d data/matpes/_cache ]; then
    cp -a data/matpes/_cache "$STAGE_DATA/_cache"
    chmod -R a+rwX "$STAGE_DATA/_cache"
  fi
  trap 'rm -rf "$STAGE_DATA"' EXIT
  set +e
  PREP_ARGS=(
    python3 /opt/mat-agent-datasets/prep_scripts/prepare_matpes.py
    --configs "$CONFIG"
    --train-n "$TRAIN_N"
    --val-n "$VAL_N"
    --test-n "$TEST_N"
    --out-root /out
    --sample-mode "$SAMPLE_MODE"
    --seed "$SEED"
  )
  if [ "$STREAMING" = "1" ]; then
    PREP_ARGS+=(--streaming)
  fi
  DOCKER_ENV=(
    -e "HF_ENDPOINT=$HF_ENDPOINT"
    -e HF_HUB_ENABLE_HF_TRANSFER=0
    -e MATPES_CACHE_DIR=/out/_cache
  )
  if [ "$USE_VPN_PROXY" = "1" ]; then
    DOCKER_ENV+=(
      -e "HTTPS_PROXY=$VPN_PROXY_URL"
      -e "HTTP_PROXY=$VPN_PROXY_URL"
      -e "ALL_PROXY=$VPN_PROXY_URL"
      -e "https_proxy=$VPN_PROXY_URL"
      -e "http_proxy=$VPN_PROXY_URL"
      -e "all_proxy=$VPN_PROXY_URL"
    )
  fi
  {
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    printf '%q ' "${PREP_ARGS[@]}"
    echo
  } > "$STAGE_DATA/prepare_cmd.sh"
  chmod a+rx "$STAGE_DATA/prepare_cmd.sh"
  docker run --rm \
    ${DOCKER_NETWORK_ARGS:+$DOCKER_NETWORK_ARGS} \
    -v "$STAGE_DATA:/out" \
    "${DOCKER_ENV[@]}" \
    -w /work \
    mat-agent/base:latest \
    /bin/bash -lc 'bash /out/prepare_cmd.sh > /out/prepare.log 2>&1'
  PREP_CODE=$?
  set -e
  cp "$STAGE_DATA/prepare.log" "$RUN_DIR/prepare_matpes.log" 2>/dev/null || true
  if [ "$PREP_CODE" -ne 0 ]; then
    echo "[matpes] preparation failed with exit $PREP_CODE"
    [ ! -f "$RUN_DIR/prepare_matpes.log" ] || tail -120 "$RUN_DIR/prepare_matpes.log"
    exit "$PREP_CODE"
  fi
  mkdir -p "$(dirname "$DATA_DIR")"
  if [ -d "$STAGE_DATA/_cache" ]; then
    rm -rf data/matpes/_cache
    cp -a "$STAGE_DATA/_cache" data/matpes/_cache
  fi
  rm -rf "$DATA_DIR"
  cp -a "$STAGE_DATA/$CONFIG" "$DATA_DIR"
else
  echo "[matpes] using existing $DATA_DIR"
fi

if [ "$PREP_ONLY" = "1" ]; then
  echo "[matpes] prep-only requested; stopping before MACE training"
  exit 0
fi

export MAT_AGENT_RUN_NAME="$RUN_NAME"
export MAT_AGENT_LR="$LR"
export MAT_AGENT_WEIGHT_DECAY="$WEIGHT_DECAY"
export MAT_AGENT_ENERGY_WEIGHT="$ENERGY_WEIGHT"
export MAT_AGENT_FORCES_WEIGHT="$FORCES_WEIGHT"
export MAT_AGENT_SEED="$SEED"
export MAT_AGENT_ENABLE_EMA="${MAT_AGENT_ENABLE_EMA:-1}"
export MAT_AGENT_ENABLE_SWA="${MAT_AGENT_ENABLE_SWA:-1}"

bash models/mace/api/train.sh \
  --foundation-model "$FOUNDATION_MODEL" \
  --train-data "$DATA_DIR/train.extxyz" \
  --val-data "$DATA_DIR/val.extxyz" \
  --output-dir "$RUN_DIR" \
  --max-steps "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --energy-weight "$ENERGY_WEIGHT" \
  --forces-weight "$FORCES_WEIGHT" \
  --device "$DEVICE" \
  --seed "$SEED"

if [ -f "$RUN_DIR/${RUN_NAME}.model" ] && [ -f "$DATA_DIR/test.extxyz" ]; then
  bash models/mace/api/evaluate.sh \
    --model "$RUN_DIR/${RUN_NAME}.model" \
    --data "$DATA_DIR/test.extxyz" \
    --output "$RUN_DIR/test_verdict.json" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE"
fi

echo "[matpes] run dir: $RUN_DIR"
