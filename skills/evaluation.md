# Evaluation Skill

Use this skill to evaluate predictions against labeled `ref_*` data.

## Steps

1. Validate labels:

```bash
python3 -m mat_agent.cli validate-extxyz /path/to/test.extxyz
```

2. Run evaluation through the model API:

```bash
bash models/mace/api/evaluate.sh \
  --model /path/to/checkpoint.model \
  --data /path/to/test.extxyz \
  --output /path/to/metrics.json \
  --device cuda
```

3. Read the metrics JSON and record it as an experiment artifact.

`evaluate.sh` is responsible for running inference and then comparing canonical
`pred_*` fields to `ref_*` labels.
