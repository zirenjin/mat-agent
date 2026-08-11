# Dataset Policy

Dataset files are split by role:

- `train`: visible and usable for fitting.
- `val`: visible and usable for feedback and model selection.
- `test`: protected; used only by scorers or benchmark harnesses.
- `ood`: protected; used only by scorers or benchmark harnesses.
- `raw`: protected canonical source material.

Agents should write derived data under `playground/runs/<run_id>/data/` and record the derivation in the run manifest.

