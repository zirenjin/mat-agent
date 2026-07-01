# mat-agent Architecture

`mat-agent` is a reusable Materials CodeAct scaffold. It provides structured
affordances for code-acting agents working with machine-learning interatomic
potential workflows.

The repository is intentionally framework-only. It does not contain paper
benchmarks, hidden tests, scaffold ablation configs, or task-specific scoring.

## Layers

1. `models/*/api`: stable model wrappers for train, inference, and evaluation.
2. `tools/`: essential host utilities shared by wrappers and agents.
3. `mat_agent/`: reusable scaffold package organized into data, compute, MLIP,
   and research planes.
4. `playground/`: agent-facing registries and ignored local run artifacts.
5. `skills/`: operational instructions for common framework workflows.
