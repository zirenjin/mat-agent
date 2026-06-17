#!/usr/bin/env bash
# docker_helpers.sh — shared Docker run wrappers for all mat-agent models.
# Sourced by models/<model>/api/*.sh. Requires $ROOT (repo root) to be set.

# ---------------------------------------------------------------------------
# run_in_container — inference
# ---------------------------------------------------------------------------
run_in_container() {
    local image="$1" model_path="$2" data_path="$3" output_path="$4" device="$5"
    local head="${6:-}" batch_size="${7:-4}" extra_args="${8:-}"

    model_path=$(realpath "$model_path" 2>/dev/null || echo "$model_path")
    data_path=$(realpath "$data_path" 2>/dev/null || echo "$data_path")
    output_dir=$(dirname "$(realpath "$output_path" 2>/dev/null || echo "$output_path")")
    output_name=$(basename "$output_path")
    model_name=$(basename "$model_path")
    data_name=$(basename "$data_path")

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
    echo "[docker] cmd=$container_cmd" >&2

    docker run --rm \
        $gpu_flags \
        -v "$model_path:/work/models/$model_name:ro" \
        -v "$data_path:/work/data/$data_name:ro" \
        -v "$output_dir:/work/output" \
        -v "$ROOT/tools:/opt/mat-agent-tools:ro" \
        -e PYTHONPATH="/opt/mat-agent-tools" \
        -e EXTRA_CONFIG_JSON="${EXTRA_CONFIG_JSON:-}" \
        -w /work \
        "$image" \
        /bin/sh -c "$container_cmd"
    return $?
}

# ---------------------------------------------------------------------------
# run_train_in_container — training
# ---------------------------------------------------------------------------
run_train_in_container() {
    local image="$1" model_path="$2" train_data="$3" val_data="$4" output_dir="$5" device="$6"
    local head="${7:-}" epochs="${8:-4}" batch_size="${9:-4}" test_data="${10:-}"

    model_path=$(realpath "$model_path" 2>/dev/null || echo "$model_path")
    train_data=$(realpath "$train_data" 2>/dev/null || echo "$train_data")
    val_data=$(realpath "$val_data" 2>/dev/null || echo "$val_data")
    output_dir=$(realpath "$output_dir" 2>/dev/null || echo "$output_dir")
    model_name=$(basename "$model_path")
    train_name=$(basename "$train_data")
    val_name=$(basename "$val_data")

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
    echo "[docker] cmd=$container_cmd" >&2

    local extra_mount=""
    if [ -n "$test_data" ] && [ -f "$test_data" ]; then
        test_path=$(realpath "$test_data" 2>/dev/null || echo "$test_data")
        test_name=$(basename "$test_path")
        extra_mount="-v $test_path:/work/data/$test_name:ro"
    fi

    docker run --rm \
        $gpu_flags \
        -v "$model_path:/work/models/$model_name:ro" \
        -v "$train_data:/work/data/$train_name:ro" \
        -v "$val_data:/work/data/$val_name:ro" \
        -v "$output_dir:/work/output" \
        $extra_mount \
        -v "$ROOT/tools:/opt/mat-agent-tools:ro" \
        -e PYTHONPATH="/opt/mat-agent-tools" \
        -e EXTRA_CONFIG_JSON="${EXTRA_CONFIG_JSON:-}" \
        -w /work \
        "$image" \
        /bin/sh -c "$container_cmd"
    return $?
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
