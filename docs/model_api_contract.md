# Model API Contract

Model execution should go through:

```text
models/<model>/api/train.sh
models/<model>/api/inference.sh
models/<model>/api/evaluate.sh
```

Training uses `--foundation-model`. Inference and evaluation use `--model`.
Prediction files should write `pred_energy` and `pred_forces`; labeled files
should use `ref_energy` and `ref_forces`.

See `docs/unified_api_spec.md` for the canonical detailed contract.
