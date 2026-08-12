# Evaluation Skill

Use this skill to evaluate a checkpoint on labeled benchmark data.

## Boundary

The public lifecycle command is `docker/entrypoints/evaluate.sh MODEL ...`.
Protected test and OOD splits may be read only through authorized evaluation
flows, never through `train.sh`.

## Contract

Required arguments are `MODEL`, `--checkpoint`, `--eval-data`, `--split`,
`--scorer`, and `--output`. Scorers are registered in `scoring/registry.yaml`.

As of API v1, the command exists as a lifecycle placeholder and returns exit code
64 until the installed runtime implements evaluation.
