#!/usr/bin/env bash
set -euo pipefail

image="${1:-mat-agent:full}"
smoke_root="$(mktemp -d /tmp/mat-agent-docker-smoke.XXXXXX)"
smoke_name="$(basename "${smoke_root}")"
chmod 777 "${smoke_root}"
trap 'rm -rf "${smoke_root}"' EXIT

printf 'checkpoint\n' > "${smoke_root}/model.pt"
printf '1\nH 0 0 0\n' > "${smoke_root}/train.extxyz"
chmod 666 "${smoke_root}/model.pt" "${smoke_root}/train.extxyz"

container_cmd=$(cat <<'EOS'
set -euo pipefail
python -c "import mattertune, nshconfig"
train.sh mace \
  --trust-checkpoint \
  --head-mode single_head \
  --checkpoint /host_tmp/${SMOKE_NAME}/model.pt \
  --train-data /host_tmp/${SMOKE_NAME}/train.extxyz \
  --run-dir playground/runs/docker-smoke \
  --properties energy,forces \
  --learning-rate 1e-4 \
  --dry-run \
  --json > /host_tmp/${SMOKE_NAME}/plan.json
EOS
)

docker run --rm \
  -v /tmp:/host_tmp \
  -e SMOKE_NAME="${smoke_name}" \
  "${image}" \
  /bin/bash -lc "${container_cmd}"

python - "${smoke_root}/plan.json" <<'PY'
import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert plan["model_type"] == "mace", plan
assert plan["mattertune_config_valid"] is True, plan
assert plan["parity_status"]["tier_c_verified"] is False, plan
print("docker_mattertune_dry_run_smoke=ok")
PY
