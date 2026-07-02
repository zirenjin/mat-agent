# mat-agent

`mat-agent` is a reusable Materials CodeAct scaffold for machine-learning
interatomic potential workflows. It provides stable model APIs, factual
registries, lightweight utilities, and operational instructions without
prescribing benchmark-specific workflows.

This repository is framework-only. It intentionally does not include paper
benchmarks, private tests, scaffold ablation configs, or task-specific scoring.

## Stable Model APIs

Model execution goes through:

```text
models/<model>/api/train.sh
models/<model>/api/inference.sh
models/<model>/api/evaluate.sh
```

Supported model families are MACE, DeePMD/DPA, and FairChem/UMA.

## Agent Utilities

```text
mat_agent/data.py              extxyz inspection and validation
mat_agent/mlip.py              command builders for model APIs
mat_agent/failures.py          runtime log classification
mat_agent/logging.py           JSONL experiment logging
mat_agent/registry.py          registry loading and repo path helpers
mat_agent/matbench_discovery.py path helpers for vendored Matbench Discovery
mat_agent/scaffolds/           CodeAct prompts, affordances, and guardrails
```

## Skills

Operational instructions live under `skills/`:

- `skills/training.md`
- `skills/inference.md`
- `skills/evaluation.md`
- `skills/logging.md`
- `skills/matbench_discovery.md`

## Boundary

Keep downstream benchmark tasks, private labels, paper baselines, and specific
experiment scoring outside this repository.
