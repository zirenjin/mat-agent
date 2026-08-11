# Softening LiZnInIO Minimal Environment

This is the first minimal dynamic adaptation environment scaffold for a reduced WBM systematic-softening smoke test:

`dataset -> replay oracle -> multi-round adaptation episode -> hidden test -> score-vs-budget curve`

The source study is Deng et al. (2025), "Systematic softening in universal machine learning interatomic potentials". The review identified this as the lowest-risk first pilot because it supports tiny oracle budgets, including a one-label correction and a 10-label fine-tuning case.

## Current Status

- Environment selected: `BENCHMARK-SOFT-LIZNINIO-004`
- Repo path: `benchmark/envs/softening_liznino`
- Data source: Figshare article `10.6084/m9.figshare.27307776`
- Data status: raw JSON downloaded locally, MD5 verified, reduced extxyz prepared
- License status: Figshare metadata reports MIT; still record exact license text before redistribution
- Model status: MACE Docker backend and local `mace-mpa-0-medium.model` were used; default Python still lacks `mace` and `chgnet`

## Minimal Episode

- `budget = 0`: zero-shot evaluation, if a usable foundation-model runner is available.
- `budget = 1, 3, 10`: random oracle selection plus naive fine-tune, once data and MLIP dependencies are verified.
- `max_rounds = 3`
- `max_train_minutes_per_round = 10`
- Primary reduced metric: force MAE.
- Secondary reduced metric: energy MAE per atom.

## Files

- `data_manifest.yaml`: source, license, data-path, and label metadata.
- `env_config.yaml`: episode, action, feedback, and score settings.
- `split_manifest.yaml`: deterministic split plan. It remains pending until frame IDs are verified.
- `run_zero_shot.py`: runs zero-shot scoring from a prediction file, or records a clear blocker when no model backend is available.
- `run_budget_curve.py`: aggregates available score JSON files into `results/score_curve.csv`, `results/score_summary.json`, and `results/score_vs_budget.png`.
- `score.py`: scores predictions against labels and split IDs.

## Next Command Sequence

After data download and conversion to an ASE-readable labeled file:

```bash
python benchmark/envs/softening_liznino/score.py \
  --labels data/prepared/benchmark/softening_liznino/labels.extxyz \
  --predictions benchmark/envs/softening_liznino/results/predictions.extxyz \
  --split-manifest benchmark/envs/softening_liznino/split_manifest.yaml \
  --split hidden_test \
  --model zero_shot \
  --baseline zero_shot \
  --budget 0 \
  --output benchmark/envs/softening_liznino/results/zero_shot_hidden.json
```

Then aggregate:

```bash
python benchmark/envs/softening_liznino/run_budget_curve.py
```

## First Curve Result

The first reduced curve is complete for hidden force MAE:

| budget | baseline | n | force MAE mean | force MAE std | energy MAE/atom mean |
|---:|---|---:|---:|---:|---:|
| 0 | zero-shot | 1 | 0.063968 | 0.000000 | 0.005407 |
| 1 | random oracle + naive FT | 3 | 0.065788 | 0.004045 | 1.246950 |
| 3 | random oracle + naive FT | 3 | 0.067855 | 0.004879 | 2.156475 |
| 10 | random oracle + naive FT | 3 | 0.069362 | 0.001929 | 3.414191 |

This smoke test demonstrates a non-monotonic/adverse naive adaptation curve: average force MAE worsens relative to zero-shot, and energy MAE per atom degrades sharply without reference-alignment handling.
