# mat-agent Unified API Specification

> Status: canonical public contract.
> Version: 2.0.0
> Last revised: 2026-06-15

This document defines the single bash API exposed to agents. Model-specific
Python, Docker, Hydra, and framework details stay behind these shell wrappers.

## Principles

1. One public API only. Wrappers do not accept legacy aliases.
2. `--config` is the model's main runtime entry:
   - Inference: checkpoint/model file.
   - DeePMD/MACE train: pretrained checkpoint/model file used for fine-tuning.
   - FairChem train: FairChem YAML config consumed by `fairchem -c`.
3. Prediction output is always extxyz with `pred_energy` and `pred_forces`.
4. Ground truth is read from extxyz with `ref_energy` and `ref_forces`.
5. `evaluate.sh` uses the short flag `--truth`.

## Directory Layout

```text
models/<model>/api/inference.sh
models/<model>/api/train.sh
models/<model>/api/evaluate.sh
models/<model>/internal/run_inference.py
models/<model>/internal/run_train.py
tools/ase_conventions.py
tools/verdict.py
tools/docker_helpers.sh
```

Supported model names are `deepmd`, `mace`, and `fairchem`.

## Inference

All models expose:

```bash
bash models/<model>/api/inference.sh \
    --config <path> \
    --data <path> \
    --output <path> \
    --device cpu|cuda \
    [--head <name>] \
    [--batch-size <int>]
```

Arguments:

| Flag | Required | Meaning |
|---|---:|---|
| `--config` | yes | Model checkpoint/config entry. DeePMD/FairChem use `.pt`; MACE uses `.model`. |
| `--data` | yes | Input extxyz. |
| `--output` | yes | Output extxyz. |
| `--device` | yes | `cpu` or `cuda`. |
| `--head` | model-specific | Required for FairChem; optional for DeePMD; rejected/not used by MACE. |
| `--batch-size` | no | Batch size, default `4`. |

Successful output must end with an extxyz where every predicted structure uses:

```python
atoms.info["pred_energy"]
atoms.arrays["pred_forces"]
```

## Training

All models expose:

```bash
bash models/<model>/api/train.sh \
    --config <path> \
    --train-data <path> \
    --val-data <path> \
    --output-dir <path> \
    --device cpu|cuda \
    [--head <name>] \
    [--max-steps <int>] \
    [--batch-size <int>] \
    [--test-data <path>]
```

Arguments:

| Flag | Required | Meaning |
|---|---:|---|
| `--config` | yes | DeePMD/MACE: pretrained checkpoint/model for fine-tuning. FairChem: training YAML. |
| `--train-data` | yes | Train extxyz. |
| `--val-data` | yes | Validation extxyz. |
| `--output-dir` | yes | Directory for `train.log`, checkpoints, and optional predictions. |
| `--device` | yes | `cpu` or `cuda`. |
| `--head` | model-specific | Task/head selector where applicable. |
| `--max-steps` | no | Smoke-test training step budget, default `4`. |
| `--batch-size` | no | Batch size, default `4`. |
| `--test-data` | no | Optional extxyz for post-train inference. |

Training output:

- Every wrapper writes `<output-dir>/train.log` when the backend supports logging.
- DeePMD/MACE write model artifacts according to their backend conventions.
- FairChem copies the latest checkpoint under `<output-dir>` to `<output-dir>/frozen.pt`.

### FairChem Training Config

FairChem training is implemented through the official FairChem v2 CLI:

```bash
fairchem -c <yaml>
```

The wrapper mounts `--config`, `--train-data`, `--val-data`, and `--output-dir`
into the container and exports these variables for use in YAML:

| Environment variable | Meaning |
|---|---|
| `MAT_AGENT_TRAIN_DATA` | Mounted train extxyz path. |
| `MAT_AGENT_VAL_DATA` | Mounted validation extxyz path. |
| `MAT_AGENT_TEST_DATA` | Mounted optional test extxyz path. |
| `MAT_AGENT_OUTPUT_DIR` | Mounted output directory. |
| `MAT_AGENT_DEVICE` | `cpu` or `cuda`. |
| `MAT_AGENT_HEAD` | Task/head selector. |
| `MAT_AGENT_MAX_STEPS` | Unified max-step budget. |
| `MAT_AGENT_BATCH_SIZE` | Unified batch size. |

FairChem YAML should reference these with OmegaConf environment interpolation,
for example `${oc.env:MAT_AGENT_TRAIN_DATA}`. This keeps the public mat-agent API
fixed while leaving the FairChem runner schema in the YAML where it belongs.

## Evaluation

All models expose:

```bash
bash models/<model>/api/evaluate.sh \
    --predictions <path> \
    --truth <path> \
    --output <path> \
    [--metrics force_rmse,energy_rmse,spearman,mean_centered_mae]
```

Arguments:

| Flag | Required | Meaning |
|---|---:|---|
| `--predictions` | yes | extxyz from `inference.sh`, with `pred_energy` and `pred_forces`. |
| `--truth` | yes | Ground-truth extxyz, preferably with `ref_energy` and `ref_forces`. |
| `--output` | yes | Verdict JSON path. |
| `--metrics` | no | Reserved selector for future metric subsets. |

The model-specific evaluate wrappers all call the shared `tools/verdict.py`.

## Field Conventions

Use these canonical keys:

| Kind | Energy key | Force key |
|---|---|---|
| Ground truth | `ref_energy` | `ref_forces` |
| Prediction | `pred_energy` | `pred_forces` |

`tools/ase_conventions.py` is the single helper module for reading and writing
these fields. Legacy keys such as `energy`, `forces`, `MACE_energy`, and
`MACE_forces` may be read during migration, but wrappers must write canonical
`pred_*` fields.

## Exit Codes

| Code | Meaning |
|---:|---|
| `0` | Success. |
| `1` | Usage or input path error. |
| `2` | Backend runtime error. |
| `3` | Data or output artifact error. |

## Examples

```bash
bash models/deepmd/api/inference.sh \
    --config datasets/raw/pretrained/DPA-3.1-3M.pt \
    --data datasets/prepared/water_tutorial/data_0.extxyz \
    --output experiments/predictions/deepmd_water.xyz \
    --device cuda \
    --head H2O_H2O_PD
```

```bash
bash models/mace/api/train.sh \
    --config datasets/raw/pretrained/mace.model \
    --train-data datasets/prepared/rmd17_aspirin/train.extxyz \
    --val-data datasets/prepared/rmd17_aspirin/val.extxyz \
    --output-dir experiments/checkpoints/mace_rmd17 \
    --device cuda \
    --max-steps 100
```

```bash
bash models/fairchem/api/evaluate.sh \
    --predictions experiments/predictions/uma_oc20.xyz \
    --truth datasets/prepared/oc20_1k/test.extxyz \
    --output experiments/verdicts/uma_oc20.json
```
