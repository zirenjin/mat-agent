# Task Loop Database

Each meaningful test or reproduction loop should leave two commits:

1. Architecture commit: code/config/schema changes that make the loop runnable.
2. Result commit: test output, run manifest, audit note, or result summary after the loop is executed.

Use `tools/task_loop_db.py` to maintain a small SQLite history of loops, commits, artifacts, and metrics. The default database path is:

`playground/loop_db/task_loops.sqlite`

Example:

```bash
python tools/task_loop_db.py start \
  --name mace-mof0-phonon-rmsd \
  --objective "Run apples-to-apples phonon RMSD for MACE-MP-MOF0"

python tools/task_loop_db.py event \
  --loop-id 1 \
  --kind architecture_commit \
  --summary "Implemented paper-aligned phonon runner"

python tools/task_loop_db.py finish \
  --loop-id 1 \
  --summary "Phonon runner validated; DFT refs blocked by TUGraz download" \
  --status blocked \
  --artifact data/mace_mof_0/results/apples_to_apples/manifest.json
```

The database intentionally records paths and metrics, not large artifacts. Large checkpoints, model files, and raw data stay outside git unless a task explicitly requires a small derived summary.
