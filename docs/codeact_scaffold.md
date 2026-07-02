# Materials CodeAct Scaffold

The scaffold is an operating contract for code-acting materials agents. It is
not a ReAct runner, benchmark harness, or planner engine.

## Loop

1. Inspect available registries, data, models, and previous artifacts.
2. State a compact hypothesis or operational goal.
3. Execute a concrete command or write code against a stable API.
4. Observe logs, metrics, files, and failure signals.
5. Revise the plan when evidence contradicts the expectation.
6. Record the final command, artifacts, metrics, and interpretation.

## Affordances

- Data: `mat_agent.data` for extxyz inspection and validation.
- Model APIs: `models/<model>/api/{train,inference,evaluate}.sh`.
- MLIP commands: `mat_agent.mlip` for constructing API calls.
- Runtime diagnosis: `mat_agent.failures` for log classification.
- Evidence logging: `mat_agent.logging` for JSONL experiment records.
- Registries: `playground/registry/*.yaml` for factual model, dataset, and tool inventory.

## Boundary

Downstream experiment repositories define tasks, private tests, scoring, and
paper-specific baselines. This repository provides the reusable scaffold only.
