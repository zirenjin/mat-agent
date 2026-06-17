# mat-agent

`mat-agent` is a Docker-first workspace for running MLIP experiments with DeePMD, MACE, and FairChem/UMA in a GPU-capable environment. The maintained public surface is the bash API under `models/*/api`.

## Repository layout

```text
/GenSIvePFS/users/public/mat-agent
├── build_images.sh                 # Build maintained Docker images
├── smoke_test.sh                   # Lightweight interface/runtime check
├── tools/
│   ├── docker_helpers.sh           # Shared Docker wrapper utilities
│   ├── ase_conventions.py          # Canonical ref_/pred_ ASE field helpers
│   └── verdict.py                  # Shared prediction-vs-truth evaluator
├── models/
│   ├── deepmd/
│   │   ├── api/                    # DeePMD host-side bash interface
│   │   ├── docker/                 # DeePMD image definition
│   │   └── internal/               # Container-side implementation
│   ├── mace/
│   │   ├── api/                    # MACE host-side bash interface
│   │   ├── docker/                 # MACE image definition
│   │   └── internal/               # Container-side implementation
│   └── fairchem/
│       ├── api/                    # FairChem/UMA host-side bash interface
│       ├── docker/                 # FairChem image definition
│       └── internal/               # Container-side implementation
├── data/, datasets/                # Input data and prepared datasets
├── uma_ckpts/                      # UMA checkpoints
├── results/, runs/                 # Experiment outputs
├── sops/, docs/                    # Maintained operational docs
└── time_logger.py                  # Optional runtime logging helper
```

## Docker images

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

Docker layer storage is on `/ebs/docker`, which is smaller than the PFS workspace. For large image sharing, prefer a registry such as GHCR over `docker save` on this host.

## Bash API conventions

Run commands from the host. The API scripts start Docker containers internally.

Pass absolute host paths to model, data, and output arguments. The wrappers mount those paths into the container.

Use `--device cuda` for GPU execution; the wrapper then adds `--gpus all` to `docker run`.

Use `--model-path` for model/checkpoint paths. `--config` remains accepted as a backward-compatible alias.

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

The evaluator keeps legacy fallbacks, but new data and predictions should use the canonical field names above.

## DeePMD interface

Inference:

```bash
bash models/deepmd/api/inference.sh \
  --model-path /abs/path/to/model.pt \
  --data /abs/path/to/input.extxyz \
  --output /abs/path/to/pred.extxyz \
  --device cuda \
  --head H2O_H2O_PD \
  --batch-size 4
```

Training or fine-tuning:

```bash
bash models/deepmd/api/train.sh \
  --model-path /abs/path/to/model.pt \
  --train-data /abs/path/to/train.extxyz \
  --val-data /abs/path/to/val.extxyz \
  --output-dir /abs/path/to/output_dir \
  --device cuda \
  --head H2O_H2O_PD \
  --max-steps 1000 \
  --batch-size 4
```

Evaluation:

```bash
bash models/deepmd/api/evaluate.sh \
  --predictions /abs/path/to/pred.extxyz \
  --truth /abs/path/to/truth.extxyz \
  --output /abs/path/to/verdict.json
```

DeePMD multi-task checkpoints usually require `--head`.

## MACE interface

Inference:

```bash
bash models/mace/api/inference.sh \
  --model-path /abs/path/to/model.model \
  --data /abs/path/to/input.extxyz \
  --output /abs/path/to/pred.extxyz \
  --device cuda \
  --batch-size 4
```

Training:

```bash
bash models/mace/api/train.sh \
  --model-path /abs/path/to/model.model \
  --train-data /abs/path/to/train.extxyz \
  --val-data /abs/path/to/val.extxyz \
  --output-dir /abs/path/to/output_dir \
  --device cuda \
  --max-steps 1000 \
  --batch-size 4
```

Evaluation:

```bash
bash models/mace/api/evaluate.sh \
  --predictions /abs/path/to/pred.extxyz \
  --truth /abs/path/to/truth.extxyz \
  --output /abs/path/to/verdict.json
```

MACE does not use `--head`.

## FairChem / UMA interface

Inference:

```bash
bash models/fairchem/api/inference.sh \
  --model-path /abs/path/to/uma-s-1p1.pt \
  --data /abs/path/to/input.extxyz \
  --output /abs/path/to/pred.extxyz \
  --head oc20 \
  --device cuda \
  --batch-size 4
```

Evaluation:

```bash
bash models/fairchem/api/evaluate.sh \
  --predictions /abs/path/to/pred.extxyz \
  --truth /abs/path/to/truth.extxyz \
  --output /abs/path/to/verdict.json
```

Valid UMA heads include `oc20`, `omat`, `omol`, and `odac`.

FairChem/UMA training is currently unsupported in `mat-agent`; `models/fairchem/api/train.sh` exits with a clear unsupported message.

## Smoke test

Run a lightweight interface/runtime check:

```bash
cd /GenSIvePFS/users/public/mat-agent
bash smoke_test.sh
```

By default the smoke test expects CUDA. To run the same checks without GPU assertions:

```bash
MAT_AGENT_DEVICE=cpu bash smoke_test.sh
```
