#!/usr/bin/env bash
set -u

ROOT="/GenSIvePFS/users/public/mat-agent"
IMAGE="mat-agent/mace-matbench:latest"
ATOMS_ZIP="/work/2024-08-04-wbm-initial-atoms.extxyz.zip"
N_CHUNKS=200

TARGET_ACTIVE="${TARGET_ACTIVE:-40}"
MEM_LIMIT_MIB="${MEM_LIMIT_MIB:-70000}"
BATCH_MAX="${BATCH_MAX:-6}"
SLEEP_SEC="${SLEEP_SEC:-600}"
DEADLINE="${DEADLINE:-}"
EMAIL_TO="${EMAIL_TO:-zirenjin@umich.edu}"
EMAIL_INTERVAL_SEC="${EMAIL_INTERVAL_SEC:-7200}"
EMAIL_SUBJECT_PREFIX="${EMAIL_SUBJECT_PREFIX:-[mat-agent mbd]}"
state_dir="${ROOT}/playground/runs/mbd_official_rerun/autoscale"
mkdir -p "${state_dir}"
state_file="${state_dir}/launched.tsv"
last_email_file="${state_dir}/last_email_epoch"
touch "${state_file}"

model_path() {
  case "$1" in
    pbe_50k) echo "/work/playground/runs/mace_matpes_pbe_50k/mace_matpes_pbe_50k.model" ;;
    r2scan_50k) echo "/work/playground/runs/mace_matpes_r2scan_50k/mace_matpes_r2scan_50k.model" ;;
    mace_mpa_0) echo "/work/models/checkpoints/mace/mace-mpa-0-medium.model" ;;
    *) return 1 ;;
  esac
}

active_count() {
  docker ps --format '{{.Names}}' | grep -c '^mbd-' || true
}

gpu_mem_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1
}

is_active() {
  docker ps --format '{{.Names}}' | grep -qx "$1"
}

is_done() {
  local model="$1"
  local chunk="$2"
  local pad total out
  pad="$(printf "%03d" "${chunk}")"
  total="$(printf "%03d" "${N_CHUNKS}")"
  out="${ROOT}/playground/runs/mbd_official_rerun/${model}/${model}-wbm-IS2RE-FIRE-${pad}-of-${total}.json.gz"
  [ -s "${out}" ]
}

was_launched() {
  grep -qx "$1	$2" "${state_file}"
}

mark_launched() {
  printf "%s\t%s\n" "$1" "$2" >> "${state_file}"
}

launch_one() {
  local model="$1"
  local chunk="$2"
  local pad cname path out_dir host_log host_status container_log container_status
  pad="$(printf "%03d" "${chunk}")"
  cname="mbd-${model}-${pad}"
  path="$(model_path "${model}")"
  out_dir="/work/playground/runs/mbd_official_rerun/${model}"
  host_log="${ROOT}/playground/runs/mbd_official_rerun/${model}/logs/chunk-${pad}.log"
  host_status="${ROOT}/playground/runs/mbd_official_rerun/${model}/logs/chunk-${pad}.status"
  container_log="/work/playground/runs/mbd_official_rerun/${model}/logs/chunk-${pad}.log"
  container_status="/work/playground/runs/mbd_official_rerun/${model}/logs/chunk-${pad}.status"
  mkdir -p "$(dirname "${host_log}")"

  if is_active "${cname}" || is_done "${model}" "${chunk}"; then
    return 1
  fi

  echo "[autoscale] launch ${model} chunk ${chunk}/${N_CHUNKS} at $(date -Is)" | tee -a "${state_dir}/autoscale.log"
  docker run -d --rm \
    --name "${cname}" \
    --gpus all \
    -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH=/work/scoring/matbench-discovery:/work \
    -v "${ROOT}:/work" \
    -w /work \
    "${IMAGE}" \
    bash -lc "echo '[container] start '\"\$(date -Is)\" >> '${container_log}'; python3 playground/scripts/run_mbd_official_mace_relax.py --model-name '${model}' --model '${path}' --atoms-zip '${ATOMS_ZIP}' --out-dir '${out_dir}' --chunk-id '${chunk}' --n-chunks '${N_CHUNKS}' --device cuda --dtype float64 >> '${container_log}' 2>&1; s=\$?; echo \$s > '${container_status}'; echo '[container] exit='\$s' '\"\$(date -Is)\" >> '${container_log}'; exit \$s" \
    >> "${state_dir}/docker-launch.log" 2>&1
  local rc=$?
  if [ "${rc}" -eq 0 ]; then
    mark_launched "${model}" "${chunk}"
    return 0
  fi
  echo "[autoscale] failed launch ${model} ${chunk} rc=${rc}" | tee -a "${state_dir}/autoscale.log"
  return 1
}

next_launches() {
  local launched=0
  for chunk in $(seq 1 "${N_CHUNKS}"); do
    for model in pbe_50k r2scan_50k mace_mpa_0; do
      [ "${launched}" -ge "${BATCH_MAX}" ] && return 0
      local active mem
      active="$(active_count)"
      mem="$(gpu_mem_mib)"
      [ "${active}" -ge "${TARGET_ACTIVE}" ] && return 0
      [ "${mem}" -ge "${MEM_LIMIT_MIB}" ] && return 0
      if launch_one "${model}" "${chunk}"; then
        launched=$((launched + 1))
        sleep 10
      fi
    done
  done
}

completed_chunks() {
  local model="$1"
  local count=0
  local chunk pad total out
  total="$(printf "%03d" "${N_CHUNKS}")"
  for chunk in $(seq 1 "${N_CHUNKS}"); do
    pad="$(printf "%03d" "${chunk}")"
    out="${ROOT}/playground/runs/mbd_official_rerun/${model}/${model}-wbm-IS2RE-FIRE-${pad}-of-${total}.json.gz"
    if [ -s "${out}" ]; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

progress_summary() {
  local active mem pbe r2scan mpa total
  active="$(active_count)"
  mem="$(gpu_mem_mib)"
  pbe="$(completed_chunks pbe_50k)"
  r2scan="$(completed_chunks r2scan_50k)"
  mpa="$(completed_chunks mace_mpa_0)"
  total=$((pbe + r2scan + mpa))
  cat <<EOF
time: $(date -Is)
host: $(hostname)
root: ${ROOT}
active_containers: ${active}
gpu_memory_mib: ${mem}
target_active: ${TARGET_ACTIVE}
memory_limit_mib: ${MEM_LIMIT_MIB}
completed_total: ${total}/$((N_CHUNKS * 3))
pbe_50k: ${pbe}/${N_CHUNKS}
r2scan_50k: ${r2scan}/${N_CHUNKS}
mace_mpa_0: ${mpa}/${N_CHUNKS}
autoscale_log: ${state_dir}/autoscale.log
EOF
}

send_progress_email() {
  local subject="$1"
  local body="$2"
  [ -n "${EMAIL_TO}" ] || return 0

  if command -v mail >/dev/null 2>&1; then
    printf "%s\n" "${body}" | mail -s "${subject}" "${EMAIL_TO}"
  elif command -v mailx >/dev/null 2>&1; then
    printf "%s\n" "${body}" | mailx -s "${subject}" "${EMAIL_TO}"
  elif command -v sendmail >/dev/null 2>&1; then
    {
      printf "To: %s\n" "${EMAIL_TO}"
      printf "Subject: %s\n" "${subject}"
      printf "\n%s\n" "${body}"
    } | sendmail -t
  else
    echo "[autoscale] email unavailable: no mail/mailx/sendmail found; wanted to send to ${EMAIL_TO}" | tee -a "${state_dir}/autoscale.log"
    return 1
  fi
}

maybe_send_progress_email() {
  [ -n "${EMAIL_TO}" ] || return 0
  [ "${EMAIL_INTERVAL_SEC}" -gt 0 ] || return 0

  local now last elapsed subject body
  now="$(date +%s)"
  last="0"
  if [ -s "${last_email_file}" ]; then
    last="$(cat "${last_email_file}" 2>/dev/null || echo 0)"
  fi
  elapsed=$((now - last))
  [ "${elapsed}" -ge "${EMAIL_INTERVAL_SEC}" ] || return 0

  body="$(progress_summary)"
  subject="${EMAIL_SUBJECT_PREFIX} progress $(date -Is)"
  if send_progress_email "${subject}" "${body}"; then
    echo "[autoscale] emailed progress to ${EMAIL_TO}" | tee -a "${state_dir}/autoscale.log"
  fi
  echo "${now}" > "${last_email_file}"
}

all_models_done() {
  [ "$(completed_chunks pbe_50k)" -eq "${N_CHUNKS}" ] \
    && [ "$(completed_chunks r2scan_50k)" -eq "${N_CHUNKS}" ] \
    && [ "$(completed_chunks mace_mpa_0)" -eq "${N_CHUNKS}" ]
}

deadline_reached() {
  [ -n "${DEADLINE}" ] || return 1
  local deadline_epoch
  deadline_epoch="$(TZ=Asia/Shanghai date -d "${DEADLINE}" +%s)"
  [ "$(date +%s)" -ge "${deadline_epoch}" ]
}

echo "[autoscale] start target=${TARGET_ACTIVE} mem_limit=${MEM_LIMIT_MIB} deadline=${DEADLINE:-none}" | tee -a "${state_dir}/autoscale.log"
while true; do
  if all_models_done; then
    echo "[autoscale] all model chunks complete $(date -Is)" | tee -a "${state_dir}/autoscale.log"
    send_progress_email "${EMAIL_SUBJECT_PREFIX} complete $(date -Is)" "$(progress_summary)"
    exit 0
  fi
  if deadline_reached; then
    echo "[autoscale] deadline reached $(date -Is)" | tee -a "${state_dir}/autoscale.log"
    send_progress_email "${EMAIL_SUBJECT_PREFIX} deadline reached $(date -Is)" "$(progress_summary)"
    exit 0
  fi
  active="$(active_count)"
  mem="$(gpu_mem_mib)"
  echo "[autoscale] tick $(date -Is) active=${active} mem=${mem}MiB done pbe=$(completed_chunks pbe_50k) r2scan=$(completed_chunks r2scan_50k) mpa=$(completed_chunks mace_mpa_0)" | tee -a "${state_dir}/autoscale.log"
  maybe_send_progress_email
  if [ "${active}" -lt "${TARGET_ACTIVE}" ] && [ "${mem}" -lt "${MEM_LIMIT_MIB}" ]; then
    next_launches
  fi
  sleep "${SLEEP_SEC}"
done
