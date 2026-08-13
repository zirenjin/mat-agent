# Physics Metrics

Physics metrics are grouped by paper/benchmark. Each subfolder owns its README,
DOI/source attribution, and Bash-first metric scripts.

Current benchmark folders:

- `mace_mof_0/`: MACE-MOF-0 MOF physics tests, DOI `10.1038/s41524-025-01611-8`.

Shared helpers live in `_common.py`; benchmark folders should keep one metric per
Python file and emit stable JSON to stdout.
