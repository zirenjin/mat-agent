#!/usr/bin/env bash
set -u

usage() {
  cat >&2 <<'USAGE'
Usage: launch_mbd_official_relax_range.sh MODEL_NAME MODEL_PATH N_CHUNKS START_CHUNK END_CHUNK PARALLEL

Launch a specific chunk range for official-compatible MBD WBM IS2RE-SR MACE relaxation.
Must be run on thu-GenSi from /GenSIvePFS/users/public/mat-agent.
USAGE
  exit 2
}

[ "$#" -eq 6 ] || usage

MODEL_NAME="$1"
MODEL_PATH="$2"
N_CHUNKS="$3"
START_CHUNK="$4"
END_CHUNK="$5"
PARALLEL="$6"

ROOT="/GenSIvePFS/users/public/mat-agent"
IMAGE="mat-agent/mace-matbench:latest"
OUT_DIR="/work/playground/runs/mbd_official_rerun/${MODEL_NAME}"
HOST_OUT_DIR="${ROOT}/playground/runs/mbd_official_rerun/${MODEL_NAME}"
ATOMS_ZIP="/work/2024-08-04-wbm-initial-atoms.extxyz.zip"

mkdir -p "${HOST_OUT_DIR}/logs"

run_chunk() {
  local chunk="$1"
  local chunk_pad
  local total_pad
  chunk_pad="$(printf "%03d" "${chunk}")"
  total_pad="$(printf "%03d" "${N_CHUNKS}")"
  local out_file="${HOST_OUT_DIR}/${MODEL_NAME}-wbm-IS2RE-FIRE-${chunk_pad}-of-${total_pad}.json.gz"
  local status_file="${HOST_OUT_DIR}/logs/chunk-${chunk_pad}.status"
  if [ -s "${out_file}" ] && [ -s "${status_file}" ] && [ "$(cat "${status_file}" 2>/dev/null)" = "0" ]; then
    echo "[skip] ${MODEL_NAME} chunk ${chunk}/${N_CHUNKS}"
    return 0
  fi

  "${ROOT}/tools/run_logged_docker.sh" \
    --name "mbd-${MODEL_NAME}-${chunk_pad}" \
    --image "${IMAGE}" \
    --log "${HOST_OUT_DIR}/logs/chunk-${chunk_pad}.log" \
    --gpu \
    --mount "${ROOT}:/work" \
    --workdir /work \
    --env "PYTHONPATH=/work/third_party/matbench-discovery:/work" \
    -- python3 playground/scripts/run_mbd_official_mace_relax.py \
      --model-name "${MODEL_NAME}" \
      --model "${MODEL_PATH}" \
      --atoms-zip "${ATOMS_ZIP}" \
      --out-dir "${OUT_DIR}" \
      --chunk-id "${chunk}" \
      --n-chunks "${N_CHUNKS}" \
      --device cuda \
      --dtype float64
}

export -f run_chunk
export MODEL_NAME MODEL_PATH N_CHUNKS ROOT IMAGE OUT_DIR HOST_OUT_DIR ATOMS_ZIP

seq "${START_CHUNK}" "${END_CHUNK}" | xargs -P "${PARALLEL}" -I {} bash -lc 'run_chunk "$@"' _ {}
