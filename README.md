# mat-agent

`mat-agent` is a Docker-first workspace for running MLIP experiments with
DeePMD, MACE, and FairChem/UMA in a GPU-capable environment. The maintained
public surface is the bash API under `models/*/api`.

## Repository Layout

```text
/GenSIvePFS/users/public/mat-agent
├── build_images.sh                 # Build maintained Docker images
├── smoke_test.sh                   # Lightweight interface/runtime check
├── docs/
│   └── unified_api_spec.md         # Canonical public API contract
├── tools/
│   ├── docker_helpers.sh           # Shared Docker wrapper utilities
│   ├── ase_conventions.py          # Canonical ref_/pred_ ASE field helpers
│   └── verdict.py                  # Shared prediction-vs-truth evaluator
└── models/
    ├── deepmd/
    │   ├── api/                    # DeePMD host-side bash interface
    │   └── internal/               # Container-side implementation
    ├── mace/
    │   ├── api/                    # MACE host-side bash interface
    │   └── internal/               # Container-side implementation
    └── fairchem/
        ├── api/                    # FairChem/UMA host-side bash interface
        └── internal/               # Container-side implementation
```

Large datasets, checkpoints, experiment runs, result files, and one-off data
preparation scripts are intentionally kept out of Git.


## Playground

`playground/` is the agent-facing orchestration layer above the stable
`models/*/api` commands. It keeps lightweight registries for datasets, models,
tools, and tasks, plus scripts for repeatable preparation, fine-tuning, and
benchmark probes.

```text
playground/
├── README.md                 # Playground overview
├── registry/                 # Agent-visible dataset/model/task/tool manifests
└── scripts/                  # Reusable experiment adapters
```

Generated runs live under `playground/runs/` and are ignored by Git. Keep large
artifacts, downloaded data, logs, checkpoints, and leaderboard outputs out of
source control; commit only reusable scripts, registries, and documentation.

See `docs/agent_infra.md` and `docs/data_ingestion.md` for the playground
contracts.

## Docker Images

The maintained images are:

```text
mat-agent/base:latest      shared Python/ASE/evaluation utilities
mat-agent/deepmd:latest    DeePMD runtime
mat-agent/mace:latest      MACE runtime
mat-agent/fairchem:latest  FairChem/UMA runtime
```

Rebuild images from the repository root:

```bash
cd /GenSIvePFS/users/public/mat-agent
bash build_images.sh
```

## Public API

Run commands from the host. The API scripts start Docker containers internally.
Pass absolute host paths to model, data, and output arguments.

Training uses `--foundation-model` for the pretrained checkpoint and optional
`--config` for training hyperparameters. Inference and evaluation use `--model`
for the checkpoint to run.

Prediction outputs are canonicalized to:

```text
pred_energy
pred_forces
```

Reference/truth files should use:

```text
ref_energy
ref_forces
```

The full contract is in `docs/unified_api_spec.md`.

## Training

```bash
bash models/<model>/api/train.sh \
  --foundation-model /abs/path/to/checkpoint \
  --config /abs/path/to/train.yaml \
  --train-data /abs/path/to/train.extxyz \
  --val-data /abs/path/to/val.extxyz \
  --output-dir /abs/path/to/output_dir \
  --replay-data /abs/path/to/replay.extxyz \
  --head <task-or-head> \
  --finetune-mode single \
  --max-steps 1000 \
  --batch-size 4 \
  --lr 1e-4 \
  --weight-decay 0.0 \
  --grad-clip 10.0 \
  --energy-weight 1.0 \
  --forces-weight 100.0 \
  --stress-weight 0.0 \
  --replay-weight 1.0 \
  --device cuda \
  --num-workers 4 \
  --seed 0 \
  --save-every 100 \
  --log-every 10
```

`--config`, `--replay-data`, and most hyperparameters are optional. Use `--head`
for multi-head or task-routed models such as DeePMD and FairChem/UMA.

## Evaluation

```bash
bash models/<model>/api/evaluate.sh \
  --model /abs/path/to/checkpoint \
  --data /abs/path/to/test.extxyz \
  --output /abs/path/to/metrics.json \
  --head <task-or-head> \
  --compute-stress \
  --metrics both \
  --device cuda \
  --batch-size 4 \
  --num-workers 4 \
  --dtype float32 \
  --seed 0
```

`evaluate.sh` runs the matching `inference.sh` internally, then compares the
temporary predictions against `--data` through `tools/verdict.py`.

## Inference

```bash
bash models/<model>/api/inference.sh \
  --model /abs/path/to/checkpoint \
  --data /abs/path/to/input.extxyz \
  --output /abs/path/to/pred.extxyz \
  --head <task-or-head> \
  --compute-stress \
  --device cuda \
  --batch-size 4 \
  --num-workers 4 \
  --dtype float32 \
  --seed 0
```

FairChem/UMA requires `--head` values such as `oc20`, `omat`, `omol`, or `odac`.
MACE accepts `--head` for API uniformity but does not normally use it.

## Verdict Utility

`tools/verdict.py` is part of the maintained API. It computes energy/force
metrics from prediction extxyz files (`pred_energy`, `pred_forces`) and
ground-truth extxyz files (`ref_energy`, `ref_forces`). The model-specific
`evaluate.sh` wrappers call it after running inference.

## Smoke Test

Run a lightweight interface/runtime check:

```bash
cd /GenSIvePFS/users/public/mat-agent
bash smoke_test.sh
```

By default the smoke test expects CUDA. To run the same checks without GPU
assertions:

```bash
MAT_AGENT_DEVICE=cpu bash smoke_test.sh
```
