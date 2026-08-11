# Data

`data/` is the only top-level place for datasets and data preparation.

```text
registry.yaml       canonical dataset inventory
prep_scripts/       reusable scripts that create or inspect datasets
<dataset_id>/       local dataset files and manifests
checkpoints/        local model/checkpoint assets mounted or referenced by tasks
raw/                protected raw source data
prepared/           prepared canonical splits
```

Large data files are gitignored. Derived agent data should be written under `playground/runs/<run_id>/data/`, not back into canonical dataset folders.
