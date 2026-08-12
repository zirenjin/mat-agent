#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: run_logged_docker.sh --name NAME --image IMAGE --log LOG_PATH [--workdir DIR] [--gpu]
                            [--mount SRC:DST[:MODE]]... [--env KEY=VALUE]... -- CMD [ARG...]

Runs Docker with robust host + container-side logging. The log directory is
mounted at /matagent-log and the container appends stdout/stderr directly to the
same host log file, avoiding images whose Docker stdout is unreliable.
USAGE
  exit 1
}

NAME=""; IMAGE=""; LOG=""; WORKDIR="/work"; GPU=0
MOUNTS=(); ENVS=(); CMD=()
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --log) LOG="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --gpu) GPU=1; shift ;;
    --mount) MOUNTS+=("$2"); shift 2 ;;
    --env) ENVS+=("$2"); shift 2 ;;
    --) shift; CMD=("$@"); break ;;
    -h|--help) usage ;;
    *) echo "Unknown flag: $1" >&2; usage ;;
  esac
done
[ -n "$NAME" ] || { echo "ERROR: --name required" >&2; usage; }
[ -n "$IMAGE" ] || { echo "ERROR: --image required" >&2; usage; }
[ -n "$LOG" ] || { echo "ERROR: --log required" >&2; usage; }
[ "${#CMD[@]}" -gt 0 ] || { echo "ERROR: command required after --" >&2; usage; }

LOG_ABS=$(realpath -m "$LOG")
LOG_DIR=$(dirname "$LOG_ABS")
LOG_BASE=$(basename "$LOG_ABS")
META="${LOG_ABS%.log}.meta.json"
STATUS_FILE="${LOG_ABS%.log}.status"
CONTAINER_STATUS_FILE="${LOG_ABS%.log}.container_status"
DOCKER_STATUS_FILE="${LOG_ABS%.log}.docker_status"
mkdir -p "$LOG_DIR"
chmod -R a+rwX "$LOG_DIR" || true
rm -f "$STATUS_FILE" "$CONTAINER_STATUS_FILE" "$DOCKER_STATUS_FILE"
: > "$LOG_ABS"
chmod a+rw "$LOG_ABS" || true

MOUNTS_JSON=$(printf '%s\n' "${MOUNTS[@]}" | python3 -c 'import json,sys; print(json.dumps([l.rstrip("\n") for l in sys.stdin if l.rstrip("\n")]))')
ENVS_JSON=$(printf '%s\n' "${ENVS[@]}" | python3 -c 'import json,sys; print(json.dumps([l.rstrip("\n") for l in sys.stdin if l.rstrip("\n")]))')
CMD_JSON=$(printf '%s\n' "${CMD[@]}" | python3 -c 'import json,sys; print(json.dumps([l.rstrip("\n") for l in sys.stdin if l.rstrip("\n")]))')
export NAME IMAGE LOG_ABS WORKDIR GPU MOUNTS_JSON ENVS_JSON CMD_JSON STATUS_FILE CONTAINER_STATUS_FILE DOCKER_STATUS_FILE
python3 - <<'PY' > "$META"
import json, os, socket, datetime
payload = {
  "name": os.environ["NAME"],
  "image": os.environ["IMAGE"],
  "log": os.environ["LOG_ABS"],
  "workdir": os.environ["WORKDIR"],
  "gpu": os.environ.get("GPU") == "1",
  "mounts": json.loads(os.environ.get("MOUNTS_JSON", "[]")),
  "env": json.loads(os.environ.get("ENVS_JSON", "[]")),
  "cmd": json.loads(os.environ.get("CMD_JSON", "[]")),
  "host": socket.gethostname(),
  "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
  "status_file": os.environ.get("STATUS_FILE"),
  "container_status_file": os.environ.get("CONTAINER_STATUS_FILE"),
  "docker_status_file": os.environ.get("DOCKER_STATUS_FILE"),
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY

DOCKER_ARGS=(
  run --rm --name "$NAME"
  -e PYTHONUNBUFFERED=1
  -e MAT_AGENT_CONTAINER_LOG="/matagent-log/$LOG_BASE"
  -e MAT_AGENT_CONTAINER_STATUS="/matagent-log/$(basename "$CONTAINER_STATUS_FILE")"
)
[ "$GPU" -eq 0 ] || DOCKER_ARGS+=(--gpus all)
DOCKER_ARGS+=(-v "$LOG_DIR:/matagent-log")
for m in "${MOUNTS[@]}"; do
  src="${m%%:*}"
  if [ -e "$src" ]; then chmod -R a+rwX "$src" 2>/dev/null || true; fi
  DOCKER_ARGS+=(-v "$m")
done
for e in "${ENVS[@]}"; do DOCKER_ARGS+=(-e "$e"); done
DOCKER_ARGS+=(-w "$WORKDIR" "$IMAGE")

echo "[logged-docker] name=$NAME image=$IMAGE" | tee -a "$LOG_ABS"
echo "[logged-docker] log=$LOG_ABS" | tee -a "$LOG_ABS"
echo "[logged-docker] meta=$META" | tee -a "$LOG_ABS"
echo "[logged-docker] cmd=${CMD[*]}" | tee -a "$LOG_ABS"
set +e
docker "${DOCKER_ARGS[@]}" bash -lc '
  echo "[container] start $(date -Is)" >> "$MAT_AGENT_CONTAINER_LOG"
  "$@" >> "$MAT_AGENT_CONTAINER_LOG" 2>&1
  status=$?
  echo "$status" > "$MAT_AGENT_CONTAINER_STATUS"
  echo "[container] exit=$status $(date -Is)" >> "$MAT_AGENT_CONTAINER_LOG"
  exit "$status"
' _ "${CMD[@]}" >> "$LOG_ABS" 2>&1
docker_status=$?
set -e
echo "$docker_status" > "$DOCKER_STATUS_FILE"
status="$docker_status"
if [ -s "$CONTAINER_STATUS_FILE" ]; then
  status="$(tr -dc '0-9' < "$CONTAINER_STATUS_FILE" | head -c 8)"
  [ -n "$status" ] || status="$docker_status"
fi
echo "$status" > "$STATUS_FILE"
echo "[logged-docker] docker_exit=$docker_status container_exit=$(cat "$CONTAINER_STATUS_FILE" 2>/dev/null || echo missing) final_exit=$status" | tee -a "$LOG_ABS"
if [ "$status" -ne 0 ]; then
  echo "[logged-docker] failure tail ($LOG_ABS):" >&2
  tail -n 160 "$LOG_ABS" >&2 || true
fi
exit "$status"
