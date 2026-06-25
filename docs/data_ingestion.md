# Data Ingestion

Dataset ingestion is model-agnostic. It lives in `datasets/prep_scripts/` and
feeds the stable model API through explicit artifacts under `data/`.

## Responsibilities

A preparation script should:

- download or read source data
- convert it into the repository convention
- write split files and a `manifest.json`
- keep hidden targets out of agent-visible paths when needed

For MLIP datasets, prefer extxyz with:

```text
atoms.info["ref_energy"]
atoms.arrays["ref_forces"]
atoms.info["ref_stress"]  # optional
```

For property-prediction datasets, prefer JSONL records with stable IDs and a
manifest that describes folds, targets, and visibility.

## Playground Registration

Prepared datasets should be discoverable through
`playground/registry/datasets.yaml`. Preparation tools that are safe for agents
to call should be registered in `playground/registry/tools.yaml`.

## Git Hygiene

Commit reusable ingestion code and manifests. Do not commit generated datasets,
large downloads, hidden targets, logs, checkpoints, or experiment outputs.
Those belong under ignored paths such as `data/`, `logs/`, or
`playground/runs/`.
