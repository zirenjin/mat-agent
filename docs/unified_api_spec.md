# mat-agent Unified API Specification

> Status: canonical public contract.
> Version: 3.0.0
> Last revised: 2026-06-17

This document defines the single bash API exposed to agents. Model-specific
Python, Docker, Hydra, and framework details stay behind these shell wrappers.

## Principles

1. One public API only. Wrappers do not accept legacy aliases.
2. Training uses `--foundation-model` for the pretrained checkpoint and
   optional `--config` for training hyperparameters.
3. Inference and evaluation use `--model` for the checkpoint to run.
4. Prediction output is always extxyz with `pred_energy` and `pred_forces`.
5. Ground truth is read from extxyz with `ref_energy` and `ref_forces`.
6. `evaluate.sh` runs inference internally and writes metrics JSON.

## Directory Layout

```text
models/<model>/api/train.sh
models/<model>/api/evaluate.sh
models/<model>/api/inference.sh
models/<model>/internal/run_inference.py
models/<model>/internal/run_train.py
tools/ase_conventions.py
tools/verdict.py
tools/docker_helpers.sh
```

Supported model names are `deepmd`, `mace`, and `fairchem`.

## Training

All models expose:

```bash
bash models/<model>/api/train.sh \
  --foundation-model <checkpoint> \
  --config <yaml> \
  --train-data <train.extxyz> \
  --val-data <val.extxyz> \
  --output-dir <dir> \
  --replay-data <replay.extxyz> \
  --head <task-or-head> \
  --finetune-mode <single|multihead|multitask|lora> \
  --max-steps <int> \
  --batch-size <int> \
  --lr <float> \
  --weight-decay <float> \
  --grad-clip <float> \
  --energy-weight <float> \
  --forces-weight <float> \
  --stress-weight <float> \
  --replay-weight <float> \
  --device <cpu|cuda> \
  --num-workers <int> \
  --seed <int> \
  --save-every <int> \
  --log-every <int>
```

Arguments:

| Flag | Required | Meaning |
|---|---:|---|
| `--foundation-model` | yes | Pretrained checkpoint/model used for fine-tuning. |
| `--config` | no | Training hyperparameter YAML. Wrappers provide defaults where backend support exists. |
| `--train-data` | yes | Train extxyz. |
| `--val-data` | yes | Validation extxyz. |
| `--output-dir` | yes | Directory for `train.log`, checkpoints, and optional predictions. |
| `--replay-data` | no | Replay extxyz for multihead/multitask training. |
| `--head` | model-specific | Task/head selector where applicable. |
| `--finetune-mode` | no | `single`, `multihead`, `multitask`, or `lora`; default `single`. |
| `--max-steps` | no | Step budget, default `4` for smoke tests. |
| `--batch-size` | no | Batch size, default `4`. |
| `--lr` | no | Learning rate. |
| `--weight-decay` | no | Weight decay. |
| `--grad-clip` | no | Gradient clipping threshold. |
| `--energy-weight` | no | Energy loss weight. |
| `--forces-weight` | no | Force loss weight. |
| `--stress-weight` | no | Stress loss weight. |
| `--replay-weight` | no | Replay-head loss weight. |
| `--device` | no | `cpu` or `cuda`; default `cpu`. |
| `--num-workers` | no | Data loader worker count; default `0`. |
| `--seed` | no | Reproducibility seed; default `0`. |
| `--save-every` | no | Checkpoint save interval. |
| `--log-every` | no | Logging interval. |

## Evaluation

All models expose:

```bash
bash models/<model>/api/evaluate.sh \
  --model <checkpoint> \
  --data <test.extxyz> \
  --output <metrics.json> \
  --head <task-or-head> \
  --compute-stress \
  --metrics <mae|rmse|both> \
  --device <cpu|cuda> \
  --batch-size <int> \
  --num-workers <int> \
  --dtype <float32|float64> \
  --seed <int>
```

Arguments:

| Flag | Required | Meaning |
|---|---:|---|
| `--model` | yes | Checkpoint/model to evaluate. |
| `--data` | yes | Ground-truth extxyz with `ref_energy` and `ref_forces`. |
| `--output` | yes | Metrics JSON path. |
| `--head` | model-specific | Task/head selector where applicable. |
| `--compute-stress` | no | Request stress metrics/prediction where supported. |
| `--metrics` | no | `mae`, `rmse`, or `both`; default `both`. |
| `--device` | no | `cpu` or `cuda`; default `cpu`. |
| `--batch-size` | no | Batch size, default `4`. |
| `--num-workers` | no | Data loader worker count; default `0`. |
| `--dtype` | no | `float32` or `float64`; default `float32`. |
| `--seed` | no | Reproducibility seed; default `0`. |

`evaluate.sh` calls the matching `inference.sh` internally, then compares the
temporary predictions against `--data` through `tools/verdict.py`.

## Inference

All models expose:

```bash
bash models/<model>/api/inference.sh \
  --model <checkpoint> \
  --data <input.extxyz> \
  --output <pred.extxyz> \
  --head <task-or-head> \
  --compute-stress \
  --device <cpu|cuda> \
  --batch-size <int> \
  --num-workers <int> \
  --dtype <float32|float64> \
  --seed <int>
```

Arguments:

| Flag | Required | Meaning |
|---|---:|---|
| `--model` | yes | Checkpoint/model to run. |
| `--data` | yes | Input extxyz. |
| `--output` | yes | Output extxyz. |
| `--head` | model-specific | Required for FairChem; optional for DeePMD; accepted for API uniformity by MACE. |
| `--compute-stress` | no | Request stress prediction where supported. |
| `--device` | no | `cpu` or `cuda`; default `cpu`. |
| `--batch-size` | no | Batch size, default `4`. |
| `--num-workers` | no | Data loader worker count; default `0`. |
| `--dtype` | no | `float32` or `float64`; default `float32`. |
| `--seed` | no | Reproducibility seed; default `0`. |

Successful output must end with an extxyz where every predicted structure uses:

```python
atoms.info["pred_energy"]
atoms.arrays["pred_forces"]
```

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
bash models/deepmd/api/train.sh \
  --foundation-model datasets/raw/pretrained/DPA-3.1-3M.pt \
  --train-data datasets/prepared/water_tutorial/train.extxyz \
  --val-data datasets/prepared/water_tutorial/val.extxyz \
  --output-dir experiments/checkpoints/deepmd_water \
  --head H2O_H2O_PD \
  --finetune-mode single \
  --device cuda
```

```bash
bash models/mace/api/evaluate.sh \
  --model experiments/checkpoints/mace_rmd17/model.model \
  --data datasets/prepared/rmd17_aspirin/test.extxyz \
  --output experiments/verdicts/mace_rmd17.json \
  --metrics both \
  --device cuda
```

```bash
bash models/fairchem/api/inference.sh \
  --model datasets/raw/pretrained/uma-s-1.pt \
  --data datasets/prepared/oc20_1k/test.extxyz \
  --output experiments/predictions/uma_oc20.xyz \
  --head oc20 \
  --device cuda
```
