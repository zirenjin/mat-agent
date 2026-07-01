# Data Ingestion Boundary

Dataset ingestion for specific projects should live in downstream repositories,
not in this framework repository.

`mat-agent` provides generic data-plane affordances that downstream projects can
reuse:

- inspect extxyz files
- validate canonical `ref_energy` and `ref_forces` labels
- rewrite extxyz files through ASE
- create small deterministic splits
- summarize simple geometry signals

For MLIP datasets, prefer extxyz with:

```text
atoms.info["ref_energy"]
atoms.arrays["ref_forces"]
atoms.info["ref_stress"]  # optional
```

Keep generated datasets, large downloads, logs, checkpoints, hidden labels, and
experiment outputs out of this repository.
