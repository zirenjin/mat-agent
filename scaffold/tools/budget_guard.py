#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description='Validate a requested action against a budget JSON file.')
    p.add_argument('--budget', required=True); p.add_argument('--epochs', type=int, default=0)
    p.add_argument('--train-minutes', type=float, default=0.0); p.add_argument('--oracle-labels', type=int, default=0); p.add_argument('--gpus', type=int, default=0)
    a = p.parse_args(); b = json.loads(Path(a.budget).read_text()); violations = []
    for key, requested in [('max_epochs', a.epochs), ('max_train_minutes_per_round', a.train_minutes), ('max_oracle_labels', a.oracle_labels), ('max_gpus', a.gpus)]:
        limit = b.get(key)
        if limit is not None and requested > limit: violations.append(f'{key}: requested {requested} > limit {limit}')
    result = {'ok': not violations, 'violations': violations}
    print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result['ok'] else 2
if __name__ == '__main__': raise SystemExit(main())

