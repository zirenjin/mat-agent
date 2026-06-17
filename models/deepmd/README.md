# DeePMD-kit (DPA) Model Wrapper

## Image

`mat-agent/deepmd:latest` — based on official `deepmodeling/deepmd-kit:3.1.2_cuda129`.

## Contract Deviation

Per `docs/unified_api_spec.md`, all model images should derive from `mat-agent/base:latest`.
The DeePMD-kit image deviates from this contract:

- **Base**: `deepmodeling/deepmd-kit:3.1.2_cuda129` (official DeepModeling image)
- **Why**: deepmd-kit 3.1.x requires exact PyTorch+CUDA+CXX11_ABI alignment that proved
  impossible to achieve with a self-built pip/conda image after 4+ hours of iteration.
  The official image bundles a conda environment with PyTorch 2.8.0, CUDA 12.9, and
  deepmd-kit 3.1.2 all pre-compiled with matching ABIs.
- **Impact**: ~15.5 GB base image (vs ~3.6 GB for mat-agent/base). Acceptable because
  deepmd-kit already ships large CUDA/MPI dependencies that would be installed anyway.
- **MACE and FairChem**: Continue to use `mat-agent/base:latest` as specified.
  This deviation applies to deepmd only.

## Runtime

Python 3.12 conda environment at `/opt/deepmd-kit/`. Key paths:
- `python3` → `/opt/deepmd-kit/bin/python3`
- `dp` → `/opt/deepmd-kit/bin/dp`
- Site-packages: `/opt/deepmd-kit/lib/python3.12/site-packages/`

mat-agent tools:
- `/opt/mat-agent-tools/ase_conventions.py`
- `/opt/internal/run_inference.py`, `run_train.py`
- PYTHONPATH includes both

## Verified

| Check | Result |
|---|---|
| torch version | 2.8.0 |
| cxx11_abi | True |
| CUDA available | True (A100-SXM4-80GB) |
| DPA-3.1-3M.pt load | OK |
| Inference smoke test | 1000 structures, 123.5s |
