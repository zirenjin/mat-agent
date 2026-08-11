# Inference Skill

Use this skill to run a maintained model API for prediction.

## Steps

1. Inspect the input structure file:

```bash
python3 -m mat_agent.cli inspect-extxyz /path/to/input.extxyz
```

2. Run the model API directly:

```bash
bash models/mace/api/inference.sh \
  --model /path/to/checkpoint.model \
  --data /path/to/input.extxyz \
  --output /path/to/pred.extxyz \
  --device cuda
```

3. Confirm the output contains `pred_energy` and `pred_forces`.

Use `models/<model>/api/inference.sh` for model execution. Avoid model-specific
internal scripts unless developing the wrapper itself.
