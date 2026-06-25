#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TMP_DIR="/tmp/mat_agent_smoke14"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"
POOL="$TMP_DIR/pool.extxyz"
HIGH="$TMP_DIR/high.extxyz"
REMAIN="$TMP_DIR/remaining.extxyz"
SUMMARY="$TMP_DIR/summary.json"
LABELED="$TMP_DIR/labeled.extxyz"
VAL="$TMP_DIR/val.extxyz"
METRICS="$TMP_DIR/metrics.json"
REPORT="$TMP_DIR/eval_report.json"

PYTHONPATH="$ROOT/tools" python3 - <<PY
from ase import Atoms
from ase.io import write
import numpy as np
rng = np.random.default_rng(14)
elements = ['Li', 'O', 'Fe', 'Mo', 'S']
atoms = []
for i in range(30):
    n = int(rng.integers(3, 9))
    symbols = ''.join(rng.choice(elements, size=n))
    a = Atoms(symbols, positions=rng.random((n, 3)) * 4.0, cell=[8, 8, 8], pbc=True)
    peratom = rng.uniform(0.1, 1.0, size=n)
    a.arrays['pred_force_variance_peratom'] = peratom
    a.info['pred_force_variance_max'] = float(np.max(peratom))
    a.info['pred_force_variance_p90'] = float(np.percentile(peratom, 90))
    a.info['pred_force_variance_mean'] = float(np.mean(peratom))
    a.arrays['mace_node_embedding'] = rng.normal(size=(n, 16))
    a.info['pred_energy'] = float(rng.normal())
    a.arrays['pred_forces'] = rng.normal(size=(n, 3))
    a.info['ref_energy'] = a.info['pred_energy']
    a.arrays['ref_forces'] = a.arrays['pred_forces'].copy()
    atoms.append(a)
write('$POOL', atoms, format='extxyz')
write('$VAL', atoms[:5], format='extxyz')
PY

PYTHONPATH="$ROOT/tools" python3 "$ROOT/tools/active_learning.py" \
  --pool "$POOL" --output-high "$HIGH" --output-remaining "$REMAIN" --summary "$SUMMARY" \
  --strategy combined --uncertainty-field max --max-select 10
python3 -m json.tool "$SUMMARY" >/dev/null
PYTHONPATH="$ROOT/tools" python3 - <<PY
import json
from ase.io import read
s = json.load(open('$SUMMARY'))
high = read('$HIGH', ':')
remain = read('$REMAIN', ':')
assert len(high) <= 10
assert len(high) + len(remain) == 30
assert s['diversity_method_used'] == 'embedding'
for key in ['n_pool_input', 'n_selected', 'uncertainty_field_used', 'element_variance_breakdown']:
    assert key in s
PY

PYTHONPATH="$ROOT/tools" python3 - <<PY
from ase.io import read, write
from ase_conventions import get_pred_energy, get_pred_forces, set_ref
from datetime import datetime, timezone
atoms = read('$HIGH', ':')
ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
for atom in atoms:
    set_ref(atom, get_pred_energy(atom), get_pred_forces(atom))
    atom.info['oracle_model'] = 'MACE-MP-0'
    atom.info['oracle_timestamp'] = ts
    for key in list(atom.info):
        if key == 'pred_energy' or key.startswith('pred_force_variance_') or key == 'pred_energy_variance':
            del atom.info[key]
    for key in list(atom.arrays):
        if key == 'pred_forces' or key.startswith('pred_force_variance_') or key == 'mace_node_embedding':
            del atom.arrays[key]
write('$LABELED', atoms, format='extxyz')
PY
PYTHONPATH="$ROOT/tools" python3 - <<PY
from ase.io import read
atoms = read('$LABELED', ':')
assert atoms
for atom in atoms:
    assert 'ref_energy' in atom.info
    assert 'ref_forces' in atom.arrays
    assert 'pred_energy' not in atom.info
    assert 'pred_forces' not in atom.arrays
    assert 'oracle_model' in atom.info
PY

PYTHONPATH="$ROOT/tools" python3 "$ROOT/tools/eval_report.py" \
  --predictions "$VAL" --truth "$VAL" --output "$REPORT" --model mace --checkpoint-path smoke.model \
  --task-id smoke14 --uncertainty-threshold 0.3
python3 -m json.tool "$REPORT" >/dev/null
PYTHONPATH="$ROOT/tools" python3 - <<PY
import json
r = json.load(open('$REPORT'))
assert r['status'] == 'ok'
assert r['uncertainty_stats'] is not None
PY

echo "SMOKE TEST 14 OK"
