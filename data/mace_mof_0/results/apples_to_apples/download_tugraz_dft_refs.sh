#!/usr/bin/env bash
set -euo pipefail

# TU Graz record wyc7s-8en40 currently returns 200 OK, not 206 Partial Content,
# for Range probes on /content. That means aria2/wget/curl cannot do reliable
# byte-range resume. Fastest tested path from this environment is direct
# no-proxy curl, with one full-file connection per archive and the four required
# archives running in parallel. Each archive writes to .part, verifies byte size
# and MD5, then atomically publishes the final .tar.gz.

ROOT="${ROOT:-/GenSIvePFS/users/public/mat-agent/data/mace_mof_0/external/tugraz_wyc7s-8en40}"
BASE="https://repository.tugraz.at/api/records/wyc7s-8en40/files"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-300}"
STALL_SPEED_LIMIT_BPS="${STALL_SPEED_LIMIT_BPS:-2048}"
STALL_TIME_SECONDS="${STALL_TIME_SECONDS:-900}"
CONNECT_TIMEOUT_SECONDS="${CONNECT_TIMEOUT_SECONDS:-60}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
USE_PROXY="${USE_PROXY:-0}"

mkdir -p "$ROOT"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: download_tugraz_dft_refs.sh [archive ...]

Archives:
  MOF-5.tar.gz
  UiO-66.tar.gz
  MIL-53_lp.tar.gz
  MOF-74.tar.gz
  MIL-53_np.tar.gz

Environment:
  PARALLEL_JOBS=4             Number of archives to download concurrently.
  USE_PROXY=0                 Default bypasses proxy with curl --noproxy '*'.
  RETRY_SLEEP_SECONDS=300     Sleep between full-file attempts.

If no archive is given, downloads the four archives needed for paper Table 5:
MOF-5, UiO-66, MIL-53_lp, and MOF-74.
EOF
}

manifest_line() {
  case "$1" in
    MOF-5.tar.gz)      echo "15c24738051921d1fe474fde3fa7e5f9 4273170662" ;;
    UiO-66.tar.gz)     echo "4b859edcbe072c9c5cc269ebca71d8d4 3795640162" ;;
    MIL-53_lp.tar.gz)  echo "4304c39b5343706a30ecdb14d6c9f148 6801436320" ;;
    MOF-74.tar.gz)     echo "20c202e107e4f842b591a69248632dd0 17560852217" ;;
    MIL-53_np.tar.gz)  echo "5bc138e85455cb00dda3eded932e456e 4648965376" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown archive: $1" >&2; usage >&2; exit 2 ;;
  esac
}

verify_file() {
  local file="$1"
  local md5="$2"
  local bytes="$3"
  local actual_bytes
  actual_bytes="$(stat -c '%s' "$file")"
  if [[ "$actual_bytes" != "$bytes" ]]; then
    echo "[verify] $file byte-size mismatch: got $actual_bytes expected $bytes" >&2
    return 1
  fi
  echo "$md5  $file" | md5sum -c -
}

download_one() {
  local name="$1"
  read -r md5 bytes < <(manifest_line "$name")
  local url="$BASE/$name/content"
  local part="$name.part"
  local attempt=1

  if [[ -f "$name" ]]; then
    echo "[check] $name"
    verify_file "$name" "$md5" "$bytes"
    return 0
  fi

  while true; do
    echo "[start] $name attempt=$attempt expected_bytes=$bytes md5=$md5 use_proxy=$USE_PROXY time=$(date -Is)"
    rm -f "$part"

    local curl_args=(
      --location
      --fail
      --show-error
      --connect-timeout "$CONNECT_TIMEOUT_SECONDS"
      --speed-limit "$STALL_SPEED_LIMIT_BPS"
      --speed-time "$STALL_TIME_SECONDS"
      --keepalive-time 60
      --retry 20
      --retry-all-errors
      --retry-delay 60
      --retry-max-time 0
      --output "$part"
      --write-out "[curl] $name http_code=%{http_code} size_download=%{size_download} speed_download=%{speed_download} time_total=%{time_total}\\n"
    )
    if [[ "$USE_PROXY" == "0" ]]; then
      curl_args=(--noproxy '*' "${curl_args[@]}")
    fi

    set +e
    curl "${curl_args[@]}" "$url"
    local curl_rc=$?
    set -e

    if [[ "$curl_rc" -eq 0 ]] && verify_file "$part" "$md5" "$bytes"; then
      mv "$part" "$name"
      echo "[done] $name time=$(date -Is)"
      return 0
    fi

    echo "[retry] $name curl_rc=$curl_rc sleeping=${RETRY_SLEEP_SECONDS}s time=$(date -Is)" >&2
    rm -f "$part"
    sleep "$RETRY_SLEEP_SECONDS"
    attempt=$((attempt + 1))
  done
}

if [[ "$#" -eq 0 ]]; then
  set -- MOF-5.tar.gz UiO-66.tar.gz MIL-53_lp.tar.gz MOF-74.tar.gz
fi

if [[ "$PARALLEL_JOBS" -le 1 || "$#" -le 1 ]]; then
  for archive in "$@"; do
    download_one "$archive"
  done
  exit 0
fi

status=0
for archive in "$@"; do
  while [[ "$(jobs -rp | wc -l)" -ge "$PARALLEL_JOBS" ]]; do
    wait -n || status=1
  done
  download_one "$archive" &
done

while [[ "$(jobs -rp | wc -l)" -gt 0 ]]; do
  wait -n || status=1
done

exit "$status"
