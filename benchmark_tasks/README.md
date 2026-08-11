# Benchmark Tasks

Benchmark tasks define what the agent sees, what it can modify, and how it is scored.

```text
tasks/<task_id>/      concrete benchmark package
baselines/            reusable baseline agents or policies
templates/            task templates and schemas
examples/             example task code or old examples retained for reference
```

A normal task package should contain:

```text
task.md
task_config.yaml
warm_start/
public_data_manifest.yaml
```

Protected test/OOD labels must not be placed in agent-writeable paths.
