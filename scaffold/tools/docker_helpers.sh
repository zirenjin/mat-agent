#!/usr/bin/env bash
# docker_helpers.sh — shared Docker run wrappers for all mat-agent models.
# Sourced by models/<model>/api/*.sh. Requires $ROOT (repo root) to be set.

# ---------------------------------------------------------------------------
# run_in_container — inference
# ---------------------------------------------------------------------------
run_in_container() {
    local image="$1" model_path="$2" data_path="$3" output_path="$4" device="$5"
    local head="${6:-}" batch_size="${7:-4}" extra_args="${8:-}"
    local model_key="${image#mat-agent/}"
    model_key="${model_key%%:*}"
    local internal_dir="$ROOT/models/$model_key/internal"

    model_path=$(realpath "$model_path" 2>/dev/null || echo "$model_path")
    data_path=$(realpath "$data_path" 2>/dev/null || echo "$data_path")
    output_dir=$(dirname "$(realpath "$output_path" 2>/dev/null || echo "$output_path")")
    output_name=$(basename "$output_path")
    model_name=$(basename "$model_path")
    data_name=$(basename "$data_path")
    local stage_dir
    stage_dir=$(mktemp -d /tmp/mat-agent-docker-infer.XXXXXX)
    trap 'rm -rf "$stage_dir"' RETURN
    mkdir -p "$stage_dir/models" "$stage_dir/data" "$stage_dir/output" "$stage_dir/tools"
    cp "$model_path" "$stage_dir/models/$model_name"
    cp "$data_path" "$stage_dir/data/$data_name"
    cp -a "$ROOT/tools/." "$stage_dir/tools/"
    chmod -R a+rwX "$stage_dir"

    local gpu_flags=""
    [ "$device" = "cuda" ] && gpu_flags="--gpus all"

    local container_cmd="python3 /opt/internal/run_inference.py \
        --model-path /work/models/$model_name \
        --data /work/data/$data_name \
        --output /work/output/$output_name \
        --device $device"
    [ -n "$head" ] && container_cmd="$container_cmd --head $head"
    [ -n "$batch_size" ] && container_cmd="$container_cmd --batch-size $batch_size"
    [ -n "$extra_args" ] && container_cmd="$container_cmd $extra_args"

    mkdir -p "$output_dir"

    echo "[docker] image=$image device=$device" >&2
    echo "[docker] stage=$stage_dir" >&2
    echo "[docker] cmd=$container_cmd" >&2

    set +e
    docker run --rm \
        $gpu_flags \
        -v "$stage_dir/models:/work/models:ro" \
        -v "$stage_dir/data:/work/data:ro" \
        -v "$stage_dir/output:/work/output" \
        -v "$stage_dir/tools:/opt/mat-agent-tools:ro" \
        -v "$internal_dir:/opt/internal:ro" \
        -e PYTHONPATH="/opt/mat-agent-tools" \
        -e EXTRA_CONFIG_JSON="${EXTRA_CONFIG_JSON:-}" \
        -e MAT_AGENT_CONFIG="${MAT_AGENT_CONFIG:-}" \
        -e MAT_AGENT_REPLAY_DATA="${MAT_AGENT_REPLAY_DATA:-}" \
        -e MAT_AGENT_FINETUNE_MODE="${MAT_AGENT_FINETUNE_MODE:-}" \
        -e MAT_AGENT_LR="${MAT_AGENT_LR:-}" \
        -e MAT_AGENT_WEIGHT_DECAY="${MAT_AGENT_WEIGHT_DECAY:-}" \
        -e MAT_AGENT_GRAD_CLIP="${MAT_AGENT_GRAD_CLIP:-}" \
        -e MAT_AGENT_ENERGY_WEIGHT="${MAT_AGENT_ENERGY_WEIGHT:-}" \
        -e MAT_AGENT_FORCES_WEIGHT="${MAT_AGENT_FORCES_WEIGHT:-}" \
        -e MAT_AGENT_STRESS_WEIGHT="${MAT_AGENT_STRESS_WEIGHT:-}" \
        -e MAT_AGENT_REPLAY_WEIGHT="${MAT_AGENT_REPLAY_WEIGHT:-}" \
        -e MAT_AGENT_NUM_WORKERS="${MAT_AGENT_NUM_WORKERS:-}" \
        -e MAT_AGENT_SEED="${MAT_AGENT_SEED:-}" \
        -e MAT_AGENT_COMPUTE_STRESS="${MAT_AGENT_COMPUTE_STRESS:-}" \
        -e MAT_AGENT_DTYPE="${MAT_AGENT_DTYPE:-}" \
        -e MAT_AGENT_UNCERTAINTY="${MAT_AGENT_UNCERTAINTY:-}" \
        -e MAT_AGENT_SAVE_EMBEDDINGS="${MAT_AGENT_SAVE_EMBEDDINGS:-}" \
        -e MAT_AGENT_SAVE_EVERY="${MAT_AGENT_SAVE_EVERY:-}" \
        -e MAT_AGENT_LOG_EVERY="${MAT_AGENT_LOG_EVERY:-}" \
        -e MAT_AGENT_RUN_NAME="${MAT_AGENT_RUN_NAME:-}" \
        -e MAT_AGENT_ENABLE_EMA="${MAT_AGENT_ENABLE_EMA:-}" \
        -e MAT_AGENT_EMA_DECAY="${MAT_AGENT_EMA_DECAY:-}" \
        -e MAT_AGENT_ENABLE_SWA="${MAT_AGENT_ENABLE_SWA:-}" \
        -e MAT_AGENT_START_SWA="${MAT_AGENT_START_SWA:-}" \
        -e MAT_AGENT_SWA_ENERGY_WEIGHT="${MAT_AGENT_SWA_ENERGY_WEIGHT:-}" \
        -e MAT_AGENT_SWA_FORCES_WEIGHT="${MAT_AGENT_SWA_FORCES_WEIGHT:-}" \
        -w /work \
        "$image" \
        bash -lc "$container_cmd" > "$stage_dir/output/docker.log" 2>&1
    local status=$?
    set -e
    cp "$stage_dir/output/docker.log" "$output_dir/docker.log" 2>/dev/null || true
    if [ "$status" -eq 0 ] && [ -f "$stage_dir/output/$output_name" ]; then
        cp "$stage_dir/output/$output_name" "$output_dir/$output_name"
        grep -E "^INFERENCE OK [0-9]+ structures" "$stage_dir/output/docker.log" | tail -n 1 || true
    elif [ -f "$stage_dir/output/docker.log" ]; then
        tail -n 80 "$stage_dir/output/docker.log" >&2 || true
    fi
    return "$status"
}

# ---------------------------------------------------------------------------
# run_train_in_container — training
# ---------------------------------------------------------------------------
run_train_in_container() {
    local image="$1" model_path="$2" train_data="$3" val_data="$4" output_dir="$5" device="$6"
    local head="${7:-}" epochs="${8:-4}" batch_size="${9:-4}" test_data="${10:-}"
    local model_key="${image#mat-agent/}"
    model_key="${model_key%%:*}"
    local internal_dir="$ROOT/models/$model_key/internal"

    model_path=$(realpath "$model_path" 2>/dev/null || echo "$model_path")
    train_data=$(realpath "$train_data" 2>/dev/null || echo "$train_data")
    val_data=$(realpath "$val_data" 2>/dev/null || echo "$val_data")
    output_dir=$(realpath "$output_dir" 2>/dev/null || echo "$output_dir")
    model_name=$(basename "$model_path")
    train_name=$(basename "$train_data")
    val_name=$(basename "$val_data")
    local stage_dir
    stage_dir=$(mktemp -d /tmp/mat-agent-docker-train.XXXXXX)
    trap 'rm -rf "$stage_dir"' RETURN
    mkdir -p "$stage_dir/models" "$stage_dir/data" "$stage_dir/output" "$stage_dir/tools"
    cp "$model_path" "$stage_dir/models/$model_name"
    cp "$train_data" "$stage_dir/data/$train_name"
    cp "$val_data" "$stage_dir/data/$val_name"
    cp -a "$ROOT/tools/." "$stage_dir/tools/"
    chmod -R a+rwX "$stage_dir"

    local gpu_flags=""
    [ "$device" = "cuda" ] && gpu_flags="--gpus all"

    local container_cmd="python3 /opt/internal/run_train.py \
        --model-path /work/models/$model_name \
        --train-data /work/data/$train_name \
        --val-data /work/data/$val_name \
        --output-dir /work/output \
        --device $device \
        --epochs $epochs"
    [ -n "$head" ] && container_cmd="$container_cmd --head $head"
    [ -n "$batch_size" ] && container_cmd="$container_cmd --batch-size $batch_size"
    if [ -n "$test_data" ] && [ -f "$test_data" ]; then
        test_path=$(realpath "$test_data" 2>/dev/null || echo "$test_data")
        test_name=$(basename "$test_path")
        container_cmd="$container_cmd --test-data /work/data/$test_name"
    fi

    mkdir -p "$output_dir"

    echo "[docker] image=$image device=$device" >&2
    echo "[docker] stage=$stage_dir" >&2
    echo "[docker] cmd=$container_cmd" >&2

    local extra_mount=""
    if [ -n "$test_data" ] && [ -f "$test_data" ]; then
        test_path=$(realpath "$test_data" 2>/dev/null || echo "$test_data")
        test_name=$(basename "$test_path")
        cp "$test_path" "$stage_dir/data/$test_name"
        chmod a+r "$stage_dir/data/$test_name"
    fi

    set +e
    docker run --rm \
        $gpu_flags \
        -v "$stage_dir/models:/work/models:ro" \
        -v "$stage_dir/data:/work/data:ro" \
        -v "$stage_dir/output:/work/output" \
        -v "$stage_dir/tools:/opt/mat-agent-tools:ro" \
        -v "$internal_dir:/opt/internal:ro" \
        -e PYTHONPATH="/opt/mat-agent-tools" \
        -e EXTRA_CONFIG_JSON="${EXTRA_CONFIG_JSON:-}" \
        -e MAT_AGENT_CONFIG="${MAT_AGENT_CONFIG:-}" \
        -e MAT_AGENT_REPLAY_DATA="${MAT_AGENT_REPLAY_DATA:-}" \
        -e MAT_AGENT_FINETUNE_MODE="${MAT_AGENT_FINETUNE_MODE:-}" \
        -e MAT_AGENT_LR="${MAT_AGENT_LR:-}" \
        -e MAT_AGENT_WEIGHT_DECAY="${MAT_AGENT_WEIGHT_DECAY:-}" \
        -e MAT_AGENT_GRAD_CLIP="${MAT_AGENT_GRAD_CLIP:-}" \
        -e MAT_AGENT_ENERGY_WEIGHT="${MAT_AGENT_ENERGY_WEIGHT:-}" \
        -e MAT_AGENT_FORCES_WEIGHT="${MAT_AGENT_FORCES_WEIGHT:-}" \
        -e MAT_AGENT_STRESS_WEIGHT="${MAT_AGENT_STRESS_WEIGHT:-}" \
        -e MAT_AGENT_REPLAY_WEIGHT="${MAT_AGENT_REPLAY_WEIGHT:-}" \
        -e MAT_AGENT_NUM_WORKERS="${MAT_AGENT_NUM_WORKERS:-}" \
        -e MAT_AGENT_SEED="${MAT_AGENT_SEED:-}" \
        -e MAT_AGENT_COMPUTE_STRESS="${MAT_AGENT_COMPUTE_STRESS:-}" \
        -e MAT_AGENT_DTYPE="${MAT_AGENT_DTYPE:-}" \
        -e MAT_AGENT_UNCERTAINTY="${MAT_AGENT_UNCERTAINTY:-}" \
        -e MAT_AGENT_SAVE_EMBEDDINGS="${MAT_AGENT_SAVE_EMBEDDINGS:-}" \
        -e MAT_AGENT_SAVE_EVERY="${MAT_AGENT_SAVE_EVERY:-}" \
        -e MAT_AGENT_LOG_EVERY="${MAT_AGENT_LOG_EVERY:-}" \
        -e MAT_AGENT_RUN_NAME="${MAT_AGENT_RUN_NAME:-}" \
        -e MAT_AGENT_ENABLE_EMA="${MAT_AGENT_ENABLE_EMA:-}" \
        -e MAT_AGENT_EMA_DECAY="${MAT_AGENT_EMA_DECAY:-}" \
        -e MAT_AGENT_ENABLE_SWA="${MAT_AGENT_ENABLE_SWA:-}" \
        -e MAT_AGENT_START_SWA="${MAT_AGENT_START_SWA:-}" \
        -e MAT_AGENT_SWA_ENERGY_WEIGHT="${MAT_AGENT_SWA_ENERGY_WEIGHT:-}" \
        -e MAT_AGENT_SWA_FORCES_WEIGHT="${MAT_AGENT_SWA_FORCES_WEIGHT:-}" \
        -w /work \
        "$image" \
        bash -lc "$container_cmd" > "$stage_dir/output/docker.log" 2>&1
    local status=$?
    set -e
    if [ -d "$stage_dir/output" ]; then
        cp -a "$stage_dir/output/." "$output_dir/"
    fi
    return "$status"
}

# ---------------------------------------------------------------------------
# parse_config_json <file>
# Load flags from JSON config. CLI values (already set) take precedence.
# Unknown keys are exported as EXTRA_CONFIG_JSON for internal/run_*.py
# ---------------------------------------------------------------------------
parse_config_json() {
    local config_file="$1"
    [ -f "$config_file" ] || { echo "ERROR: config file not found: $config_file" >&2; exit 1; }
    which jq >/dev/null 2>&1 || { echo "ERROR: jq required for --config parsing" >&2; exit 2; }
    local json
    json=$(cat "$config_file")
    [ -z "$MODEL_PATH" ] && MODEL_PATH=$(echo "$json" | jq -r '."model-path" // empty')
    [ -z "$DATA" ]       && DATA=$(echo "$json" | jq -r '.data // empty')
    [ -z "$OUTPUT" ]     && OUTPUT=$(echo "$json" | jq -r '.output // empty')
    [ -z "$DEVICE" ]     && DEVICE=$(echo "$json" | jq -r '.device // empty')
    [ -z "$HEAD" ]       && HEAD=$(echo "$json" | jq -r '.head // empty')
    [ -z "$BATCH_SIZE" ] && BATCH_SIZE=$(echo "$json" | jq -r '."batch-size" // empty')
    EXTRA=$(echo "$json" | jq -c 'del(."model-path", .data, .output, .device, .head, ."batch-size")')
    export EXTRA_CONFIG_JSON="$EXTRA"
}
