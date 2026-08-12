# Scoring

`scoring/` has exactly three active folders:

```text
machine_learning/  one Python file per generic ML metric: MAE, RMSE, R2
physics/           one Python file per physics metric inspired by MOF phonon benchmarks
adapters/          thin integrations with external benchmark projects
```

Metric files are Bash-callable and emit stable JSON. Prefer adding a new metric
as `scoring/<family>/<metric_name>.py` instead of growing a shared monolith.
