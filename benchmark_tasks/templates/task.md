# Task Title

## Goal

Describe the MLIP adaptation target in physical terms.

## Visible Inputs

- Train split:
- Validation split:
- Foundation MLIP/backbone:
- Warm-start code:

## Allowed Actions

- Fine-tune with MatterTune-supported configuration changes.
- Adjust replay, multi-head, layer-freezing, and optimization settings if listed in the task config.
- Write derived artifacts under `playground/runs/<run_id>/`.

## Forbidden Actions

- Do not modify canonical `test` or `ood` splits.
- Do not patch MLIP architecture source.
- Do not use hidden labels during optimization.

## Submission

List the files the agent must produce: checkpoint/config, predictions, run manifest, and report.

