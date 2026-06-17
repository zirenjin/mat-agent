#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEVICE="${MAT_AGENT_DEVICE:-cuda}"

gpu_flags=()
if [ "$DEVICE" = "cuda" ]; then
    gpu_flags=(--gpus all)
fi

section() {
    printf '\n== %s ==\n' "$1"
}

run_help() {
    local script="$1"
    local status
    set +e
    bash "$ROOT/$script" --help >/dev/null 2>&1
    status=$?
    set -e
    if [ "$status" -ne 0 ] && [ "$status" -ne 1 ]; then
        echo "ERROR: $script --help failed with exit $status" >&2
        exit "$status"
    fi
    echo "ok: $script --help"
}

check_image() {
    local image="$1"
    docker image inspect "$image" >/dev/null
    echo "ok: image $image"
}

section "API help"
run_help models/deepmd/api/inference.sh
run_help models/deepmd/api/train.sh
run_help models/deepmd/api/evaluate.sh
run_help models/mace/api/inference.sh
run_help models/mace/api/train.sh
run_help models/mace/api/evaluate.sh
run_help models/fairchem/api/inference.sh
run_help models/fairchem/api/train.sh
run_help models/fairchem/api/evaluate.sh

section "Docker images"
check_image mat-agent/base:latest
check_image mat-agent/deepmd:latest
check_image mat-agent/mace:latest
check_image mat-agent/fairchem:latest

section "Canonical ASE fields"
docker run --rm -i \
    -v "$ROOT/tools:/opt/mat-agent-tools:ro" \
    -e PYTHONPATH=/opt/mat-agent-tools \
    mat-agent/base:latest \
    python3 - <<'PY'
import numpy as np
from ase import Atoms
from ase_conventions import set_pred, get_pred_energy, get_pred_forces

atoms = Atoms("H", positions=[[0, 0, 0]])
set_pred(atoms, 1.25, np.zeros((1, 3)))
assert get_pred_energy(atoms) == 1.25
assert get_pred_forces(atoms).shape == (1, 3)
assert "pred_energy" in atoms.info
assert "pred_forces" in atoms.arrays
print("ok: pred_energy/pred_forces")
PY
echo "ok: pred_energy/pred_forces"

section "Runtime imports"
docker run --rm -i "${gpu_flags[@]}" mat-agent/deepmd:latest python3 - <<PY
import torch
import deepmd
if "$DEVICE" == "cuda":
    assert torch.cuda.is_available(), "cuda requested but torch.cuda.is_available() is false"
print("ok: deepmd runtime")
PY
echo "ok: deepmd runtime"
docker run --rm -i "${gpu_flags[@]}" mat-agent/mace:latest python3 - <<PY
import torch
import mace
if "$DEVICE" == "cuda":
    assert torch.cuda.is_available(), "cuda requested but torch.cuda.is_available() is false"
print("ok: mace runtime")
PY
echo "ok: mace runtime"
docker run --rm -i "${gpu_flags[@]}" mat-agent/fairchem:latest python3 - <<PY
import torch
import fairchem.core
if "$DEVICE" == "cuda":
    assert torch.cuda.is_available(), "cuda requested but torch.cuda.is_available() is false"
print("ok: fairchem runtime")
PY
echo "ok: fairchem runtime"

section "Unsupported FairChem training guard"
if bash "$ROOT/models/fairchem/api/train.sh" >/tmp/mat-agent-fairchem-train-smoke.log 2>&1; then
    echo "ERROR: FairChem train unexpectedly succeeded" >&2
    exit 1
fi
grep -q "unsupported" /tmp/mat-agent-fairchem-train-smoke.log
echo "ok: FairChem train reports unsupported"

section "Done"
echo "mat-agent smoke test passed with MAT_AGENT_DEVICE=$DEVICE"
