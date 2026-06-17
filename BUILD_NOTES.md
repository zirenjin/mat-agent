# mat-agent Docker Build Notes

> Written 2026-06-15 — engineering log for containerizing three MLIP models.
> For future maintainers: read this first — it's faster than reverse-engineering the code.

## System Architecture Overview

```
                         ┌─────────────────────────────┐
                         │     tools/                   │
                         │  ase_conventions.py          │
                         │  docker_helpers.sh           │
                         │  eval_ft.py                  │
                         │  pearson_ft.py               │
                         │  verdict.py                  │
                         └────────────┬────────────────┘
                                      │ copied into every image
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐   ┌────────────────────────┐   ┌──────────────────────┐
│ mat-agent/    │   │ mat-agent/             │   │ mat-agent/           │
│ deepmd        │   │ mace                   │   │ fairchem             │
│ 15.5 GB       │   │ 8.94 GB                │   │ 11.2 GB              │
│               │   │                        │   │                      │
│ FROM deep-    │   │ FROM mat-agent/base    │   │ FROM mat-agent/base  │
│ modeling/     │   │ + torch 2.6.0 cu126    │   │ + fairchem-core 2.20 │
│ deepmd-kit:   │   │ + mace-torch 0.3.10    │   │ + PyTorch (trans.)   │
│ 3.1.2_cuda129 │   │                        │   │                      │
│               │   │ internal/              │   │ internal/            │
│ internal/     │   │   run_inference.py     │   │   run_inference.py   │
│   run_inference.py │   run_train.py        │   │   run_train.py (!)   │
│   run_train.py│   │                        │   │                      │
└───────┬───────┘   └───────────┬────────────┘   └──────────┬───────────┘
        │                       │                           │
        │                       │    ┌──────────────────┐   │
        │                       └────┤ mat-agent/base   │◄──┘
        │                            │ 3.65 GB          │
        │ (does NOT FROM base)       │                  │
        │                            │ FROM ubuntu:22.04│
        │                            │ + python3.11     │
        │                            │ + CUDA 12.1      │
        │                            │ + ase>=3.22      │
        │                            │ + numpy, scipy   │
        │                            └──────────────────┘
        │
        │  deepmd-kit official image from deepmodeling (Docker Hub)
        │  ~7.2 GB tar, NOT FROM mat-agent/base
        └──────────────────────────────────────────────────────

    Container API layer (per model):
        models/<model>/api/
            inference.sh    — run_inference end-to-end wrapper
            eval_official.sh — official benchmark evaluation
            ft_*.sh         — fine-tune wrappers
```

## Key Decision Records (ADR-style)

### ADR-1: Field Naming — ref_*/pred_*

- **Context**: ASE extxyz reserved-key trap. ASE treats `energy` and `forces` as special keys in extxyz format — on write, they get moved into a calculator object instead of atoms.arrays / atoms.info. On re-read, these keys silently disappear, causing data loss.
- **Decision**: All ground-truth labels use `ref_energy` / `ref_forces`. All model predictions use `pred_energy` / `pred_forces`. This is the only correct way to get uniform output across all three models.
- **Alternatives considered**: Use ASE standard keys `energy` / `forces`. **Rejected because**: `SinglePointCalculator` consumes them during extxyz round-trips, confirmed in OC20 runs causing silent data loss.
- **Implementation**: `tools/ase_conventions.py` provides `set_ref()` / `get_ref_energy()` / `set_pred()` / `get_pred_energy()` helpers. All `models/*/internal/run_inference.py` and `models/*/internal/run_train.py` import this module uniformly.
- **Impact**: All downstream scripts (`tools/eval_ft.py`, `tools/verdict.py`) hard-depend on `ref_*` / `pred_*` key names.

### ADR-2: DeePMD Image Deviates from Base Contract

- **Context**: The other three images (base → mace, base → fairchem) follow a clean two-layer inheritance. deepmd-kit 3.1.x has a PyTorch + CUDA + ABI three-way alignment problem that failed 6 times inside containers.
- **Decision**: Use `FROM deepmodeling/deepmd-kit:3.1.2_cuda129` (the official prebuilt Docker image) directly, bypassing `mat-agent/base`.
- **6 failed attempts (in chronological order)**:
  1. deepmd-kit 3.0.0 + torch 2.5.1+cu121 → does not support DPA-3 descriptor
  2. GHCR (ghcr.io) pull of official image → unreachable from China
  3. conda-forge install of deepmd-kit → conda solver conflict, dependency tree unresolvable
  4. torch 2.7+cu126 (ABI=1) + PyPI wheel → deepmd-kit wheel requires ABI=0, version mismatch
  5. torch 2.10+cu126 + PyPI wheel → "inconsistent with PyTorch Runtime" at runtime
  6. Source-compile deepmd-kit inside container → CUDA compiler (nvcc) missing; installing CUDA toolkit exceeds 10 GB
- **Implementation**: Local VPN docker pull → docker save -o (7.2 GB tar) → scp -P 14990 → docker load -i. `build_images.sh` Phase 0 checks for this image and prints full acquisition guide if missing.
- **Impact**: Image is 15.5 GB (vs mat-agent/base 3.65 GB), 4× larger. But DPA-3.1-3M actually loads — all 6 failed routes cannot.

### ADR-3: Base Image — Manual CUDA Install Instead of nvidia/cuda Official

- **Context**: `nvidia/cuda:12.1.0-base-ubuntu22.04` is ~1.5 GB but lacks essential packages (no python3.11). `nvidia/cuda:12.1.0-devel-ubuntu22.04` is ~3.5 GB and includes unnecessary build toolchain.
- **Decision**: Start from `ubuntu:22.04`, manually install `cuda-libraries-12-1` + `libcudnn8` via NVIDIA China CDN (`developer.download.nvidia.cn`).
- **Alternatives considered**: Use `nvidia/cuda` official image as base. **Rejected because**: devel image too large (3.5 GB base + python layer > 5 GB); base image has outdated Python (3.10); and official images download slowly from China.
- **Implementation**: `models/_common/Dockerfile.base` uses `developer.download.nvidia.cn` CDN; apt mirrors use aliyun.
- **Impact**: Base image is only 3.65 GB, containing Ubuntu 22.04 + CUDA 12.1 + Python 3.11 + ASE. Mace and FairChem build on this base.

### ADR-4: Multi-Task Head Handling Strategy

- **Context**: All three models support multi-task checkpoints (one ckpt with head weights for multiple tasks), but API behavior is inconsistent across models.
- **Decision**: Per-model differentiated handling:
  - **DeePMD**: `--head` **is mandatory** on multi-task ckpts, otherwise error "multiple heads found".
  - **MACE**: `--head` is **explicitly rejected** — MACE uses single-task `.model` files (foundation model format).
  - **FairChem**: `--head` **is mandatory and non-empty**, value must be a task_name: `oc20` / `omat` / `omol` / `odac`.
- **Implementation**: Each `models/*/internal/run_inference.py` has its own argparse and head validation logic. `tools/docker_helpers.sh`'s `run_in_container()` passes through `--head` transparently.
- **Impact**: Users need to know per-model head requirements. `infer_runner.sh` and `prep_runner.sh` encapsulate these differences.

### ADR-5: Foundation Models Not Downloaded at Build Time

- **Context**: Foundation model checkpoints for all three models are NOT automatically downloaded during Dockerfile build.
- **Decision**: Require users to pass local absolute paths via `--model-path` at runtime.
- **Rationale**:
  1. GitHub Releases / HuggingFace Hub / Docker Hub are all unreliable from China; build-time download failure rate > 50%.
  2. Model files are large (DPA-3.1-3M.pt ~300 MB, equivariant MACE model ~200 MB, UMA-s-1p1.pt ~200 MB); embedding them would bloat images by another 0.5–1 GB.
  3. Users may need different checkpoint versions; embedding loses flexibility.
  4. `HF_ENDPOINT=https://hf-mirror.com` is set as env var for hf-mirror, but MACE's built-in HF download logic still fails intermittently.
- **Implementation**: `run_inference.py` rejects HF shorthand model names (`small`/`medium`/`large`), only accepting local absolute paths. Users are responsible for downloading models before running tasks.
- **Impact**: One extra step for onboarding (acquiring model files first), but avoids massive build failures and image bloat.

## Known Issues

### K-1: Docker Root Not Moved to GPFS

- **Current state**: Docker data directory is at `/ebs/docker` (69 GB EBS volume).
- **Problem**: Attempted to move to `/GenSIvePFS/users/public/docker-data` (GPFS shared storage), but `systemctl restart docker` rolled back after `daemon.json` modification.
- **Root cause**: GPFS overlay2 storage driver incompatibility — Docker daemon detects unsupported FS at startup and falls back to default path.
- **Impact**: 69 GB EBS volume currently ~40 GB used (4 images + build cache + container layers). More model images may strain space.
- **Fix options**: (1) Grow EBS volume; (2) Use `devicemapper` or `vfs` storage driver; (3) Mount larger EBS volume to `/ebs/docker`. Retrying GPFS direct is not recommended.

### K-2: FairChem Train Not Yet Implemented

- **Current state**: `models/fairchem/internal/run_train.py` currently raises `NotImplementedError`.
- **Rationale**: fairchem-core v2.20 fine-tune config is unstable inside containers (energy/forces key mapping issues). Upstream fairchem-core team rewrote the trainer API in v2.21.
- **Impact**: FairChem in mat-agent supports inference only, not fine-tuning.
- **TODO**: Re-evaluate and implement `run_train.py` after upgrading to fairchem-core ≥ 2.21.

### K-3: MACE Foundation Model Downloads

- **Current state**: `mace_eval_configs` built-in HF download logic frequently fails from China (connection timeout or DNS unreachable).
- **Problem**: `mace.calculators.mace_mp()` attempts to download the default model from HuggingFace Hub, timing out without a proxy.
- **Workarounds**:
  1. User downloads `.model` file locally via gh-proxy.
  2. Pass `--model-path /absolute/path/to/model` when calling `run_inference.py`.
  3. Set `HF_ENDPOINT=https://hf-mirror.com` (already set in Dockerfile).
- **Note**: `run_inference.py` rejects HF shorthand names (`small`/`medium`/`large`); only local paths are accepted.

### K-4: DeepMD Image Size 15.5 GB

- **Cause**: `deepmodeling/deepmd-kit:3.1.2_cuda129` official image includes full CUDA 12.9 toolkit, PyTorch build environment, LAMMPS, etc. The image is inherently large (7.2 GB compressed tar).
- **Impact**: docker save/load takes ~3–5 minutes; scp transfer ~70 seconds at 100 MB/s.
- **No fix available**: Unless deepmodeling publishes a slim image variant, this size must be accepted.

## China Network Workaround Guide

### Mirror Registry Summary

| Registry | URL | Purpose |
|---|---|---|
| PyPI | `https://pypi.tuna.tsinghua.edu.cn/simple` | Python package installs |
| PyTorch | `https://download.pytorch.org/whl/cu126` | PyTorch GPU wheels (official, reachable from China) |
| APT | `https://mirrors.aliyun.com` | Ubuntu system packages |
| CUDA | `https://developer.download.nvidia.cn` | CUDA 12.1 toolkit (NVIDIA China CDN) |
| HF Hub | `https://hf-mirror.com` | HuggingFace model downloads (set as HF_ENDPOINT) |
| Docker Hub | `m.daocloud.io` | Docker image proxy (unreliable) |
| GitHub | Direct or gh-proxy | Source/release downloads |

### How to Obtain deepmodeling/deepmd-kit:3.1.2_cuda129

This is the prerequisite for DeePMD containerization. The official image is ~7.2 GB compressed as tar.

**Recommended — Local docker pull + scp tar:**

On a machine with Docker Hub access (e.g., a dev box with VPN):
```bash
# Step 1: pull the official image
docker pull deepmodeling/deepmd-kit:3.1.2_cuda129

# Step 2: export as tar
docker save deepmodeling/deepmd-kit:3.1.2_cuda129 \
    -o /tmp/deepmd-kit-3.1.2.tar

# Step 3: transfer to mat-agent server
scp -P 14990 /tmp/deepmd-kit-3.1.2.tar \
    root@115.190.6.177:/tmp/
```

On the mat-agent server:
```bash
# Step 4: load into local Docker
docker load -i /tmp/deepmd-kit-3.1.2.tar

# Step 5: verify
docker image inspect deepmodeling/deepmd-kit:3.1.2_cuda129 | head -5

# Step 6: clean up (optional)
rm /tmp/deepmd-kit-3.1.2.tar
```

**Fallback — daocloud proxy:**

```bash
docker pull m.daocloud.io/deepmodeling/deepmd-kit:3.1.2_cuda129
docker tag m.daocloud.io/deepmodeling/deepmd-kit:3.1.2_cuda129 \
    deepmodeling/deepmd-kit:3.1.2_cuda129
```

Note: The daocloud proxy is unreliable; the image may not be cached or the tag may be missing. Prefer Option A above.

### cxx11 ABI Alignment Reference

| Image | torch ABI | CUDA | Reason |
|---|---|---|---|
| mat-agent/deepmd | **0** (pre-cxx11) | 12.9 | deepmd-kit official image compiled with old ABI |
| mat-agent/mace | **1** (cxx11) | 12.6 | torch 2.6.0 cu126 wheel uses new ABI |
| mat-agent/fairchem | 1 (cxx11) | 12.1 | fairchem-core depends on new-ABI torch |
| mat-agent/base | N/A | 12.1 | No torch installed, no ABI concept |

## Acceptance Test Matrix

| # | Test | Data | Foundation model | Expected last line |
|---|---|---|---|---|
| 13.1 | DeePMD inference | water tutorial / oc20_1k | DPA-3.1-3M.pt + `--head H2O_H2O_PD` | `INFERENCE OK 1000` |
| 13.2 | MACE inference | rmd17_aspirin_1k | medium.model (local absolute path) | `INFERENCE OK 1000` |
| 13.3 | FairChem inference | oc20_1k | uma-s-1p1.pt + `--head oc20` | `INFERENCE OK 1000` |

Example test commands:
```bash
# DeePMD (note: --head is mandatory)
bash models/deepmd/api/inference.sh \
    /path/to/DPA-3.1-3M.pt \
    /GenSIvePFS/users/public/mat-agent/data/oc20_1k.extxyz \
    /tmp/dpmd_out.extxyz \
    cuda H2O_H2O_PD

# MACE (note: --head must NOT be passed)
bash models/mace/api/inference.sh \
    /path/to/medium.model \
    /GenSIvePFS/users/public/mat-agent/data/rmd17_aspirin_1k.extxyz \
    /tmp/mace_out.extxyz \
    cuda

# FairChem (note: --head is mandatory)
bash models/fairchem/api/inference.sh \
    /path/to/uma-s-1p1.pt \
    /GenSIvePFS/users/public/mat-agent/data/oc20_1k.extxyz \
    /tmp/fc_out.extxyz \
    cuda oc20
```

## Image Size Reference

| Image | Size | Notes |
|---|---|---|
| `mat-agent/base` | 3.65 GB | Ubuntu 22.04 + CUDA 12.1 + Python 3.11 + ASE |
| `mat-agent/deepmd` | 15.5 GB | FROM deepmodeling/deepmd-kit:3.1.2_cuda129, full build environment |
| `mat-agent/mace` | 8.94 GB | FROM mat-agent/base + torch 2.6.0 cu126 + mace-torch 0.3.10 |
| `mat-agent/fairchem` | 11.2 GB | FROM mat-agent/base + fairchem-core 2.20 + PyTorch (transitive) |
| **Total** | **~39 GB** | ~50 GB with build cache and deepmd official image |

## Build Process

```bash
# Full rebuild
bash build_images.sh

# Rebuild only mace (e.g., after modifying mace Dockerfile)
bash build_images.sh --skip-base --skip-deepmd --skip-fairchem

# Clean rebuild of everything (no docker build cache)
bash build_images.sh --no-cache

# Rebuild base + downstream (e.g., after modifying base Dockerfile)
bash build_images.sh --skip-deepmd
```

Build order is fixed: base → deepmd → mace → fairchem.
- deepmd does not depend on base, but is placed before mace/fairchem to build the heaviest image earlier.
- mace and fairchem depend on base, so base must be built first.

## Future TODO

- [ ] **Milestone 9: Skills repository** — based on 13.X test results across all three models, write reusable skill scripts
- [ ] **DPMD finetune SOP rerun via Docker** — full rerun of DeePMD fine-tune pipeline with new images, verify train → inference loop
- [ ] **MACE-OFF23 image** — organic chemistry SOP requires dedicated MACE-OFF23 model, possibly a new image
- [ ] **Docker root moved to GPFS** — resolve overlay2/GPFS compatibility, or use alternative storage driver
- [ ] **CI/CD: GitHub Actions auto build + push** — trigger on models/*/Dockerfile changes, auto build and push to container registry
- [ ] **FairChem fine-tune support** — re-evaluate after fairchem-core ≥ 2.21 upgrade, implement `run_train.py`
- [ ] **Image upstream tracking** — monitor new deepmodeling/deepmd-kit releases, evaluate upgrade path
- [ ] **Multi-GPU testing** — currently validated on single A100 only; need multi-GPU inference correctness tests
