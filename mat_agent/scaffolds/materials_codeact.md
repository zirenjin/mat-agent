# Materials CodeAct Scaffold

Use this scaffold to act through code while keeping scientific reasoning
auditable.

## Loop

1. Inspect registries and current artifacts.
2. State the immediate hypothesis or operational goal.
3. Execute one concrete code/API step.
4. Observe logs, metrics, files, and failure signals.
5. Revise based on evidence.
6. Record the command, artifacts, and conclusion.

## Affordances

- `playground/registry/*.yaml`: factual model, dataset, and tool inventory.
- `models/<model>/api/*.sh`: stable training, inference, and evaluation APIs.
- `mat_agent.data`: extxyz inspection and validation.
- `mat_agent.mlip`: model API command builders.
- `mat_agent.failures`: runtime log classification.
- `mat_agent.logging`: experiment JSONL records.

## Boundary

Do not encode paper tasks, scoring rules, private labels, or fixed scientific
SOPs in this scaffold. Put those in downstream experiment repositories.
