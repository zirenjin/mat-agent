# Training Skill

Use this skill to train or fine-tune an MLIP through a maintained model API.

## Boundary

Training must use `models/<model>/api/train.sh`. Do not call model-internal
scripts directly unless you are developing the wrapper itself.

This skill does not define paper-specific tasks, private splits, or fixed
training recipes. Resolve datasets from `playground/registry/datasets.yaml` or a
run-local config, then record the exact train/validation paths used.

## Preconditions

1. Choose a model API from `playground/registry/models.yaml`.
2. Resolve a train dataset and a validation dataset.
3. Confirm labels are present before launching training:

```bash
python3 -m mat_agent.cli validate-extxyz /path/to/train.extxyz
python3 -m mat_agent.cli validate-extxyz /path/to/val.extxyz
```

For full datasets that do not provide framework-defined validation splits,
create the split in the downstream run workspace and log the split manifest.
Do not write derived splits back into the full-dataset registry.

## Smoke Run

Start with a short CPU or GPU smoke run:

```bash
bash models/mace/api/train.sh \
  --foundation-model /path/to/foundation.model \
  --train-data /path/to/train.extxyz \
  --val-data /path/to/val.extxyz \
  --output-dir playground/runs/<run_id>/train_smoke \
  --max-steps 4 \
  --batch-size 4 \
  --device cpu
```

Inspect logs and output artifacts before increasing steps or switching to CUDA.

## Full Run

Use the same API contract for longer training:

```bash
bash models/<model>/api/train.sh \
  --foundation-model /path/to/foundation.model \
  --config /path/to/train_config.yaml \
  --train-data /path/to/train.extxyz \
  --val-data /path/to/val.extxyz \
  --output-dir playground/runs/<run_id>/train \
  --max-steps <steps> \
  --batch-size <batch_size> \
  --device cuda
```

Optional flags include `--finetune-mode`, `--head`, `--replay-data`, `--lr`,
`--energy-weight`, `--forces-weight`, `--stress-weight`, `--seed`, `--save-every`,
and `--log-every` when supported by the wrapper.

## Required Record

Write a run-local training record, for example:

```text
playground/runs/<run_id>/training_record.yaml
```

Include:

- selected model API and foundation model path
- train and validation dataset paths
- split provenance and random seed, if a split was created
- config path and key hyperparameters
- compute device and container image
- output checkpoint path
- training log path
- validation metrics or the evaluation command to produce them

Use `skills/logging.md` to append the final command, artifacts, metrics,
and failures to the experiment log.
