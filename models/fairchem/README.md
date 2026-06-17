# FairChem/UMA Model Wrapper

## Image

`mat-agent/fairchem:latest` — based on `mat-agent/base:latest` (no contract deviation).

## Runtime

- Python 3.11, fairchem-core >= 2.20
- `/opt/mat-agent-tools/ase_conventions.py`
- `/opt/internal/run_inference.py`, `run_train.py`
- PYTHONPATH includes both

## --head Requirement

FairChem/UMA is multitask. `--head` is REQUIRED. Valid values: `oc20`, `omat`, `omol`, `odac`.
Missing `--head` causes exit 1 with error message.

## Local Model Loading

Uses `fairchem.core.pretrained_mlip.load_predict_unit(path, device=...)` — the LOCAL
ckpt API. Explicitly NOT `get_predict_unit` which goes through HuggingFace gating.
Model must be an absolute path to a `.pt` checkpoint.

UMA checkpoints (host paths):
- `/GenSIvePFS/users/public/mat-agent/uma_ckpts/uma_ckpts/uma-s-1p1.pt` (small, 1.1 GB)
- `/GenSIvePFS/users/public/mat-agent/uma_ckpts/uma_ckpts/uma-m-1p1.pt` (medium, 11 GB)

## Field Mapping

`run_inference.py` uses `FAIRChemCalculator` for inference, then calls
`ase_conventions.set_pred()` to set `pred_energy`/`pred_forces`. Any remaining
`fairchem_*` or `FAIRChem*` keys are stripped before writing.

## Training

`run_train.py` raises NotImplementedError. FairChem fine-tuning via this wrapper
is not yet supported; use fairchem CLI directly.

## Verified

| Check | Result |
|---|---|
| fairchem-core import | OK |
| load_predict_unit (local ckpt) | OK |
| Smoke test 13.3 | oc20_1k, 1000 structures |
