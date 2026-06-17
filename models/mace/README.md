# MACE Model Wrapper

## Image

`mat-agent/mace:latest` — based on `mat-agent/base:latest` (no contract deviation).

## Runtime

- Python 3.11, mace-torch 0.3.16, torch 2.6.0+cu126
- `/opt/mat-agent-tools/ase_conventions.py`
- `/opt/internal/run_inference.py`, `run_train.py`
- PYTHONPATH includes both

## Foundation Model Loading

Foundation model short names (`small`, `medium`, `large`, `medium-mpa-0`) are
explicitly rejected. User must pass an absolute path to a local `.model` file.
Download logic lives on the host, not inside the container — this avoids silent
network downloads during inference.

## --head Handling

MACE is single-task. Passing `--head` causes exit 2 with an error message.
Use DeePMD or FairChem if you need multi-task heads.

## Field Mapping

`run_inference.py` calls `mace_eval_configs` CLI and then renames:
- `MACE_energy` → `pred_energy`
- `MACE_forces` → `pred_forces`

All `MACE_*` keys are stripped before writing the final output.

## Verified

| Check | Result |
|---|---|
| torch version | 2.6.0+cu126 |
| cxx11_abi | True |
| CUDA available | True (A100-SXM4-80GB) |
| Inference smoke test | 1000 structures (rmd17_aspirin_1k) |
| MACE_energy stripped | 0 occurrences in output |
| pred_energy present | 1000/1000 |
