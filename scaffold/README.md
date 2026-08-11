# Scaffold

`scaffold/` contains reusable infrastructure for agent experiments.

```text
tools/       bash and Python utilities for logging, budgets, dataset guards, diagnostics
policies/    human-readable sandbox and MLIP rules
schemas/     JSON schemas for run records, feedback, and budgets
python/      importable helper package (`mat_agent`)
skills/      operating instructions for agents
```

This directory should stay task-agnostic. Task-specific scoring, labels, and warm starts belong in `benchmark_tasks/`.
