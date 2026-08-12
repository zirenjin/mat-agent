# Matbench Discovery Skill

Use this skill when a task needs Matbench Discovery as a scoring or benchmark
source.

## Source

The vendored upstream source snapshot lives at:

```text
scoring/adapters/matbench_discovery/upstream
```

Shared scoring adapters live under `scoring/adapters/`, and registered scorers
are listed in `scoring/registry.yaml`.

## Boundary

Matbench Discovery code is scoring infrastructure. Task definitions, warm starts, hidden labels, and split manifests should stay run-local until a benchmark format is reintroduced.

Do not copy protected WBM test/OOD data into training runs. Evaluation should go
through `docker/entrypoints/evaluate.sh` once that lifecycle command is
implemented.
