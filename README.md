# mat-agent

`mat-agent` is a reusable Materials CodeAct agent scaffold for machine-learning
interatomic potential workflows. It provides structured data, compute, MLIP, and
research affordances without prescribing benchmark-specific workflows.

This repository is framework-only. It intentionally does not include paper
benchmarks, hidden tests, scaffold ablation configs, or task-specific scoring.

## Stable Model APIs

Model execution goes through:

```text
models/<model>/api/train.sh
models/<model>/api/inference.sh
models/<model>/api/evaluate.sh
```

Supported model families are MACE, DeePMD/DPA, and FairChem/UMA.

## Scaffold Planes

```text
mat_agent/planes/data       dataset inspection, validation, conversion, splitting
mat_agent/planes/compute    smoke-run, failure parsing, GPU/checkpoint helpers
mat_agent/planes/mlip       command builders around stable model APIs
mat_agent/planes/research   generic experiment logs, comparisons, reports
```

## Skills

Operational instructions live under `skills/`:

- `skills/inference/SKILL.md`
- `skills/evaluation/SKILL.md`
- `skills/logging/SKILL.md`
- `skills/matbench_discovery/SKILL.md`

## Boundary

Keep downstream benchmark tasks, hidden labels, paper baselines, and specific
experiment scoring outside this repository.
