# Inference Skill

Use this skill to run a trained checkpoint on unlabeled structures.

## Boundary

The public lifecycle command is `docker/entrypoints/inference.sh MODEL ...`.
Inference writes predictions and does not compute benchmark metrics.

## Contract

Required arguments are `MODEL`, `--checkpoint`, `--data`, `--output`, and
`--properties`. Optional runtime arguments include `--batch-size`,
`--num-workers`, `--device`, and `--manifest`.

As of API v1, the command exists as a lifecycle placeholder and returns exit code
64 until the installed runtime implements inference.
