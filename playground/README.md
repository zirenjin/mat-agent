# Mat Agent Playground

The playground is the agent-facing layer above the stable model API in
`models/*/api`. It should contain only reusable orchestration assets:
registries, adapters, submission scripts, and documentation.

## Layout

```text
playground/
├── registry/
│   ├── datasets.yaml
│   ├── models.yaml
│   ├── tasks.yaml
│   └── tools.yaml
├── scripts/
└── runs/        # generated, ignored by Git
```

## Contracts

- Model execution must go through `models/<model>/api/{train,inference,evaluate}.sh`.
- Dataset preparation should live in `datasets/prep_scripts/` and be registered
  in `playground/registry/tools.yaml` when agent-visible.
- Task definitions belong in `playground/registry/tasks.yaml` and should point
  to reusable scripts rather than one-off commands.
- Runs, logs, checkpoints, downloaded data, and benchmark outputs belong under
  ignored paths such as `playground/runs/`, `logs/`, and `data/`.

## Adding A Playground Task

1. Add or update the dataset entry in `playground/registry/datasets.yaml`.
2. Add any model/tool capability in `playground/registry/models.yaml` or
   `playground/registry/tools.yaml`.
3. Add a task entry in `playground/registry/tasks.yaml` with defaults that are
   small enough for a smoke run.
4. Put reusable orchestration code in `playground/scripts/`.
5. Verify with a small smoke run before scaling the task outside Git-tracked
   paths.

## Adaptive Agent Primitives

The playground exposes three reusable primitives for autonomous training loops:

- `tools/anomaly_sniffer.py`: converts crash logs and latest trajectories into
  structured physical diagnostics.
- `playground/scripts/data_bridge_converter.py`: maps scout-model `pred_*`
  pseudo-labels into worker-model `ref_*` training datasets.
- `playground/scripts/runner_controller.py`: wraps model training APIs with
  telemetry, retry, and deterministic recovery hooks.

These primitives are registered in `playground/registry/tools.yaml` so planners
can discover and compose them without hard-coding paths.
