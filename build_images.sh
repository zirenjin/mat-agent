#!/usr/bin/env bash
# build_images.sh — One-command rebuild of all mat-agent Docker images.
#
# Usage:
#   bash build_images.sh                  # build all
#   bash build_images.sh --no-cache       # clean rebuild
#   bash build_images.sh --skip-base --skip-deepmd  # only mace + fairchem
#   bash build_images.sh --help           # print help
#
# Build order: base → deepmd → mace → fairchem
# (deepmd does not FROM base, but builds first because it is special —
#  it uses the official deepmodeling/deepmd-kit image directly.)

set -euo pipefail

# ---------------------------------------------------------------------------
# Globals & defaults
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_BUILD_OPTS=""

SKIP_BASE=false
SKIP_DEEPMD=false
SKIP_MACE=false
SKIP_FAIRCHEM=false
NO_CACHE=false
SHOW_HELP=false

# ---------------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --skip-base)    SKIP_BASE=true ;;
        --skip-deepmd)  SKIP_DEEPMD=true ;;
        --skip-mace)    SKIP_MACE=true ;;
        --skip-fairchem) SKIP_FAIRCHEM=true ;;
        --no-cache)     NO_CACHE=true ;;
        --help|-h)      SHOW_HELP=true ;;
        *)
            echo "[ERROR] Unknown flag: $arg"
            echo "  Run: bash build_images.sh --help"
            exit 1
            ;;
    esac
done

if $SHOW_HELP; then
    echo "build_images.sh — One-command rebuild of all mat-agent Docker images."
    echo ""
    echo "Usage:"
    echo "  bash build_images.sh                  # build all images"
    echo "  bash build_images.sh --no-cache       # clean rebuild (no docker cache)"
    echo "  bash build_images.sh --skip-base      # skip mat-agent/base"
    echo "  bash build_images.sh --skip-deepmd    # skip mat-agent/deepmd"
    echo "  bash build_images.sh --skip-mace      # skip mat-agent/mace"
    echo "  bash build_images.sh --skip-fairchem  # skip mat-agent/fairchem"
    echo "  bash build_images.sh --help           # this message"
    echo ""
    echo "Flags can be combined:"
    echo "  bash build_images.sh --skip-base --skip-deepmd --no-cache"
    echo ""
    echo "Build order (fixed): base -> deepmd -> mace -> fairchem"
    echo "  - deepmd does NOT FROM base (uses official deepmodeling/deepmd-kit image)"
    echo "  - mace and fairchem both FROM mat-agent/base"
    echo ""
    echo "Prerequisites checked before any build:"
    echo "  - docker daemon running"
    echo "  - nvidia-container-runtime available"
    echo "  - >30 GB disk space"
    echo "  - deepmodeling/deepmd-kit:3.1.2_cuda129 present locally"
    exit 0
fi

if $NO_CACHE; then
    DOCKER_BUILD_OPTS="--no-cache"
fi

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

banner() {
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  $*${NC}"
    echo -e "${GREEN}============================================${NC}"
}

fail() {
    echo -e "${RED}[FATAL] $*${NC}" >&2
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARN] $*${NC}" >&2
}

ok() {
    echo -e "${GREEN}[OK] $*${NC}"
}

# ---------------------------------------------------------------------------
# Phase 0: Prerequisites
# ---------------------------------------------------------------------------
banner "Phase 0: Prerequisites check"

check_docker_version() {
    if ! command -v docker &>/dev/null; then
        fail "docker is not installed or not in PATH"
    fi
    local ver
    ver=$(docker version --format '{{.Server.Version}}' 2>/dev/null || true)
    if [ -z "$ver" ]; then
        fail "docker daemon is not running or not accessible"
    fi
    ok "docker version: $ver"
}

needs_gpu_runtime() {
    ! $SKIP_DEEPMD || ! $SKIP_MACE || ! $SKIP_FAIRCHEM
}

check_nvidia_runtime() {
    if ! needs_gpu_runtime; then
        ok "nvidia runtime: skipped (base-only build)"
        return
    fi
    if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
        fail "nvidia-container-runtime not configured. Install nvidia-container-toolkit, or skip GPU model images."
    fi
    ok "nvidia runtime detected"
}

check_disk_space() {
    local avail_gb
    avail_gb=$(df -BG --output=avail "$REPO_ROOT" 2>/dev/null | tail -1 | sed 's/G//' | tr -d ' ')
    if [ -z "$avail_gb" ] || [ "$avail_gb" -lt 30 ]; then
        fail "Insufficient disk space: ${avail_gb:-unknown} GB available (need > 30 GB)"
    fi
    ok "disk space: ${avail_gb} GB available"
}

check_deepmd_official_image_present() {
    if $SKIP_DEEPMD; then
        ok "deepmd: skipped, no official image check needed"
        return
    fi
    if docker image inspect deepmodeling/deepmd-kit:3.1.2_cuda129 &>/dev/null; then
        ok "deepmodeling/deepmd-kit:3.1.2_cuda129 is present locally"
    else
        echo ""
        echo -e "${RED}============================================${NC}"
        echo -e "${RED}  MISSING: deepmodeling/deepmd-kit:3.1.2_cuda129${NC}"
        echo -e "${RED}============================================${NC}"
        echo ""
        echo "  This image cannot be pulled directly from China (Docker Hub is blocked)."
        echo ""
        echo "  HOW TO OBTAIN IT:"
        echo "  -----------------"
        echo ""
        echo "  Option A — scp from a machine that has it (recommended):"
        echo "    1. On a machine with VPN/Docker Hub access:"
        echo "       docker pull deepmodeling/deepmd-kit:3.1.2_cuda129"
        echo "       docker save deepmodeling/deepmd-kit:3.1.2_cuda129 -o /tmp/deepmd-kit-3.1.2.tar"
        echo ""
        echo "    2. Transfer to this server:"
        echo "       scp -P 14990 /tmp/deepmd-kit-3.1.2.tar root@115.190.6.177:/tmp/"
        echo ""
        echo "    3. Load on this server:"
        echo "       docker load -i /tmp/deepmd-kit-3.1.2.tar"
        echo ""
        echo "  Option B — daocloud proxy (may not work for this image):"
        echo "    docker pull m.daocloud.io/deepmodeling/deepmd-kit:3.1.2_cuda129"
        echo "    docker tag m.daocloud.io/deepmodeling/deepmd-kit:3.1.2_cuda129 \\"
        echo "        deepmodeling/deepmd-kit:3.1.2_cuda129"
        echo ""
        echo "  See BUILD_NOTES.md section 'China Network Workaround Guide' for more details."
        echo ""
        fail "deepmodeling/deepmd-kit:3.1.2_cuda129 not found locally"
    fi
}

check_docker_version
check_nvidia_runtime
check_disk_space
check_deepmd_official_image_present

# ---------------------------------------------------------------------------
# Phase 1: base
# ---------------------------------------------------------------------------
build_base() {
    banner "Phase 1/4: Building mat-agent/base"
    cd "$REPO_ROOT"
    docker build $DOCKER_BUILD_OPTS \
        -f models/_common/Dockerfile.base \
        -t mat-agent/base:latest \
        .
    ok "mat-agent/base:latest built"
}

verify_base() {
    echo "  -> Sanity check mat-agent/base..."
    docker run --rm mat-agent/base:latest python3 -c "
from ase_conventions import set_ref, get_ref_energy, REF_ENERGY_KEY, REF_FORCES_KEY
import ase; import numpy; import scipy; import datasets; import pymatgen; import matbench; import matminer
print('base OK: tools + ase/numpy/scipy + datasets/pymatgen/matbench/matminer importable')
" || fail "mat-agent/base verification FAILED"
    ok "mat-agent/base: verification OK"
}

# ---------------------------------------------------------------------------
# Phase 2: deepmd
# ---------------------------------------------------------------------------
build_deepmd() {
    banner "Phase 2/4: Building mat-agent/deepmd"
    cd "$REPO_ROOT"
    docker build $DOCKER_BUILD_OPTS \
        -f models/deepmd/Dockerfile \
        -t mat-agent/deepmd:latest \
        .
    ok "mat-agent/deepmd:latest built"
}

verify_deepmd() {
    echo "  -> Sanity check mat-agent/deepmd..."
    docker run --rm mat-agent/deepmd:latest python3 -c "
import torch
print('torch ' + str(torch.__version__) + ' cxx11_abi=' + str(torch._C._GLIBCXX_USE_CXX11_ABI))
assert torch._C._GLIBCXX_USE_CXX11_ABI == 0, \
    'ABI mismatch: expected cxx11-abi=0 for deepmd-kit 3.1.2'

import deepmd
print('deepmd-kit version: ' + str(deepmd.__version__))

from deepmd.infer.deep_eval import DeepEval
print('deepmd.infer.deep_eval.DeepEval import OK')
" || fail "mat-agent/deepmd verification FAILED"
    ok "mat-agent/deepmd: verification OK"
}

# ---------------------------------------------------------------------------
# Phase 3: mace
# ---------------------------------------------------------------------------
build_mace() {
    banner "Phase 3/4: Building mat-agent/mace"
    cd "$REPO_ROOT"
    docker build $DOCKER_BUILD_OPTS \
        -f models/mace/Dockerfile \
        -t mat-agent/mace:latest \
        .
    ok "mat-agent/mace:latest built"
}

verify_mace() {
    echo "  -> Sanity check mat-agent/mace..."
    docker run --rm mat-agent/mace:latest python3 -c "
import torch
print('torch ' + str(torch.__version__) + ' cxx11_abi=' + str(torch._C._GLIBCXX_USE_CXX11_ABI))
assert torch._C._GLIBCXX_USE_CXX11_ABI == 1, \
    'ABI mismatch: expected cxx11-abi=1 for mace-torch'

from mace.calculators import mace_mp
print('mace_mp imported OK')

import ase_conventions
print('ase_conventions OK')
" || fail "mat-agent/mace verification FAILED"
    ok "mat-agent/mace: verification OK"
}

# ---------------------------------------------------------------------------
# Phase 4: fairchem
# ---------------------------------------------------------------------------
build_fairchem() {
    banner "Phase 4/4: Building mat-agent/fairchem"
    cd "$REPO_ROOT"
    docker build $DOCKER_BUILD_OPTS \
        -f models/fairchem/Dockerfile \
        -t mat-agent/fairchem:latest \
        .
    ok "mat-agent/fairchem:latest built"
}

verify_fairchem() {
    echo "  -> Sanity check mat-agent/fairchem..."
    docker run --rm mat-agent/fairchem:latest python3 -c "
from fairchem.core import pretrained_mlip
from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
from ase_conventions import set_ref, set_pred
print('fairchem-core OK: pretrained_mlip + FAIRChemCalculator + ase_conventions')
" || fail "mat-agent/fairchem verification FAILED"
    ok "mat-agent/fairchem: verification OK"
}

# ---------------------------------------------------------------------------
# Main build sequence
# ---------------------------------------------------------------------------

# Phase 1: base
if $SKIP_BASE; then
    warn "Skipping mat-agent/base (--skip-base)"
else
    build_base
    verify_base
fi

# Phase 2: deepmd
if $SKIP_DEEPMD; then
    warn "Skipping mat-agent/deepmd (--skip-deepmd)"
else
    build_deepmd
    verify_deepmd
fi

# Phase 3: mace
if $SKIP_MACE; then
    warn "Skipping mat-agent/mace (--skip-mace)"
else
    build_mace
    verify_mace
fi

# Phase 4: fairchem
if $SKIP_FAIRCHEM; then
    warn "Skipping mat-agent/fairchem (--skip-fairchem)"
else
    build_fairchem
    verify_fairchem
fi

# ---------------------------------------------------------------------------
# Phase 5: Final listing
# ---------------------------------------------------------------------------
banner "Final images"
echo ""
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" | grep -E "REPOSITORY|mat-agent" || true
echo ""
ok "All images built successfully"
echo ""
echo "Next steps:"
echo "  cd $REPO_ROOT"
echo "  bash infer_runner.sh <model> <data> <output>"
echo "  See BUILD_NOTES.md for architecture decisions and known issues."
