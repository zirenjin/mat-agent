# Machine Learning Metrics

Each metric is a standalone Bash-callable Python file.

```bash
python scoring/machine_learning/mae.py --y-true 1,2,3 --y-pred 1,2,4
python scoring/machine_learning/rmse.py --true-file labels.csv --true-column y --pred-file preds.csv --pred-column y
python scoring/machine_learning/r2.py --y-true 1,2,3 --y-pred 1,2,3 --output playground/runs/run_id/r2.json
```
