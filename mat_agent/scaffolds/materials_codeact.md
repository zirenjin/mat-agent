# Materials CodeAct Scaffold

This scaffold gives a code-acting agent reusable affordances for materials
modeling without prescribing a benchmark-specific workflow.

## Planes

- Data Plane: inspect, validate, convert, split, and summarize MLIP datasets.
- Compute Plane: inspect runtime state, run smoke jobs, classify failures, and
  find checkpoints.
- MLIP Plane: build canonical train, inference, and evaluation commands for the
  maintained model APIs.
- Research Plane: log experiment intent, evidence, metrics, and reports.

## Boundary

This scaffold does not include hidden tests, paper benchmarks, task scoring, or
fixed scientific SOPs. Those belong in downstream experiment repositories.
