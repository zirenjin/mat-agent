# Logging Skill

Use this skill to keep agent experiments auditable.

## Event Logs

Append JSONL records for actions and experiments. The framework helper is:

```python
from mat_agent.planes.research.experiment import log_experiment

log_experiment(
    "runs/example/experiment_log.jsonl",
    name="baseline_eval",
    intent="Evaluate the foundation model before changing data or config.",
    command=["bash", "models/mace/api/evaluate.sh", "..."],
    metrics={"force_mae": 0.12},
)
```

## Failure Logs

Classify runtime logs with:

```bash
python3 -m mat_agent.cli.main classify-log /path/to/run.log
```

Logging captures what happened, why it was attempted, and what artifacts were
produced. It does not implement benchmark-specific scoring.
