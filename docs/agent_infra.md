# Agent Infrastructure

`mat-agent` is organized around a stable model API plus a lightweight
playground for agent-facing orchestration.

## Layers

1. `models/*/api`: stable host-side commands for training, inference, and
   evaluation.
2. `tools/` and `datasets/prep_scripts/`: shared conversion, scoring, and data
   preparation utilities.
3. `playground/`: registries and reusable scripts that let an agent discover
   tasks and run repeatable experiments.

The model API is the public execution boundary. Playground scripts may compose
API calls, but they should not bypass the model-specific wrappers for training
or evaluation.

## Base Runtime Contract

`mat-agent/base:latest` should provide shared, model-agnostic functionality:

- ASE and canonical `ref_*` / `pred_*` helpers
- `tools/verdict.py`
- data preparation dependencies such as `datasets`, `pymatgen`, `matminer`, and
  `matbench`
- dataset preparation scripts under `/opt/mat-agent-datasets/prep_scripts`

Model images add backend-specific runtimes such as MACE, DeePMD, or FairChem.
If a playground adapter needs benchmark/data dependencies inside a model image,
provide an explicit image variant or Dockerfile and document it in the
playground registry.

## Playground Contract

An agent should interact with the playground through structured files:

- `playground/registry/datasets.yaml`
- `playground/registry/models.yaml`
- `playground/registry/tasks.yaml`
- `playground/registry/tools.yaml`
- reusable scripts in `playground/scripts/`

Generated state must stay out of Git:

- `playground/runs/`
- `logs/`
- downloaded data under `data/`
- checkpoints and model outputs

## Extension Points

Add a model by implementing:

```text
models/<model>/api/train.sh
models/<model>/api/evaluate.sh
models/<model>/api/inference.sh
models/<model>/internal/
```

Add a dataset by implementing a prep script that writes a manifest plus split
artifacts under `data/<dataset>/`.

Add a task by registering it in `playground/registry/tasks.yaml` and pointing it
to a reusable script in `playground/scripts/`.
