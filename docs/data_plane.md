# Data Plane

The Data Plane exposes generic MLIP dataset affordances:

- inspect extxyz metadata and array keys
- validate `ref_energy` and `ref_forces`
- rewrite extxyz files through ASE
- create small deterministic splits
- summarize simple geometric signals

Dataset- or benchmark-specific preparation scripts should live outside this
framework repository.
