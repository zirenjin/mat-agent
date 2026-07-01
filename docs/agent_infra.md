# Agent Infrastructure

`mat-agent` is organized around stable model APIs plus a reusable Materials
CodeAct scaffold.

## Layers

1. `models/*/api`: stable host-side commands for training, inference, and
   evaluation.
2. `tools/`: essential shared utilities such as canonical ASE field handling,
   verdict scoring, Docker helpers, and runtime diagnostics.
3. `mat_agent/`: reusable data, compute, MLIP, and research planes for
   code-acting agents.
4. `playground/`: agent-facing registries and ignored local run artifacts.
5. `skills/`: operational instructions for common framework workflows.

The model API is the public execution boundary. Scaffold helpers may compose API
calls, but they should not bypass the model-specific wrappers for training,
inference, or evaluation.

## Boundary

This repository is framework-only. Do not commit benchmark tasks, hidden labels,
paper baselines, scaffold ablation configs, or task-specific scoring.

## Extension Points

Add a model by implementing:

```text
models/<model>/api/train.sh
models/<model>/api/evaluate.sh
models/<model>/api/inference.sh
models/<model>/internal/
```

Add reusable agent affordances under `mat_agent/planes/<plane>/`. Add operational
instructions under `skills/` when a workflow should be discoverable by agents.
