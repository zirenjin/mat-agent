# Training Skill

Use this skill to fine-tune an MLIP through the Bash-first MatterTune API.

## Boundary

Use `docker/entrypoints/train.sh MODEL ...`. Do not write MatterTune YAML/JSON
configs as the v1 agent-facing interface, and do not call model-internal training
scripts directly.

Training may read train and validation splits only. It must not read test or OOD
splits.

## Preconditions

1. Choose a model from `playground/registry/models.yaml`.
2. Resolve train and optional validation data from `playground/registry/datasets.yaml` or a task manifest.
3. Run `train.sh MODEL --help` and use only flags shown for that model.
4. Start with `--dry-run` and inspect `<run-dir>/execution_plan.json`.

## Example

```bash
docker/entrypoints/train.sh mace \
  --checkpoint /path/to/mace.model \
  --trust-checkpoint \
  --head-mode single_head \
  --train-data /path/to/train.extxyz \
  --val-data /path/to/val.extxyz \
  --run-dir playground/runs/<run_id> \
  --properties energy,forces \
  --learning-rate 1e-4 \
  --max-steps 100 \
  --dry-run
```

## Required Record

Keep run artifacts under `playground/runs/<run_id>/`, including the dry-run
execution plan, command argv, data identities, checkpoint identity, selected
protocol, capability requirements, logs, and final checkpoint path when training
is implemented.
