# Matbench Discovery Skill

Use this skill when a task needs Matbench Discovery as a scoring or benchmark
source.

## Source

The vendored upstream source snapshot lives at:

```text
scoring/matbench-discovery
```

Shared scoring adapters live under `scoring/adapters/`, and registered scorers
are listed in `scoring/registry.yaml`.

## Boundary

Matbench Discovery code is scoring infrastructure. Benchmark task definitions,
warm starts, hidden labels, and task-specific split manifests belong under
`benchmark_tasks/`.

Do not copy protected WBM test/OOD data into training runs. Evaluation should go
through `docker/entrypoints/evaluate.sh` once that lifecycle command is
implemented.
