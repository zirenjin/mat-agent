# MatterTune Checkpoint Defaults Loop Result

Date: 2026-08-20
Loop DB ID: 1
Architecture commit: e26b45f
MatterTune subrepo commit: b1d8966

## Result

Implemented default checkpoint preservation for MatterTune adapter runs:

- periodic archive callback: `<run-dir>/checkpoints/periodic-epoch={epoch:04d}-step={step}`, `save_last=True`, `save_top_k=-1`, `every_n_epochs=100` by default;
- validation best callback: `<run-dir>/checkpoints/best-val_loss-epoch={epoch:04d}-step={step}`, `monitor=val_loss`, `mode=min`, `save_top_k=1`, `every_n_epochs=1` when `--val-data` is present;
- loop tracking CLI: `tools/task_loop_db.py`, default DB `playground/loop_db/task_loops.sqlite`.

## Verification

- `python3 -m py_compile tools/mattertune_train.py tools/task_loop_db.py mattertune/src/mattertune/main.py`: passed
- `python3 -m unittest tests.test_mattertune_bash_dry_run_matrix -v`: passed, 8 tests
- `python3 tools/task_loop_db.py --db /tmp/task_loop_test.sqlite ...`: passed init/start/event/finish/list
