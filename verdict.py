"""
verdict.py — SOP §7: compute force/energy RMSE for MACE zero-shot on rMD17 aspirin.

Ground truth: ref_energy / ref_forces (v2 convention).
Predictions:  MACE_energy / MACE_forces (mace_eval_configs default info_prefix=MACE_).
Primary metric: force RMSE (PASS < 300, GOOD < 100 meV/A).

Usage:
    python verdict.py [predictions.xyz] [out_verdict.json] [model_label]
Defaults reproduce the MACE-MP-0 run.
"""
import json
import sys
import subprocess
import numpy as np
from ase.io import read

PRED_PATH = sys.argv[1] if len(sys.argv) > 1 else "runs/mace_rmd17/predictions.xyz"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "results/mace_rmd17_verdict.json"
MODEL_LABEL = sys.argv[3] if len(sys.argv) > 3 else "MACE-MP-0 medium"
TRUTH_PATH = "data/rmd17_aspirin_1k/test.extxyz"

preds = read(PRED_PATH, ":")
print(f"Loaded {len(preds)} predicted structures from {PRED_PATH}")

truth = read(TRUTH_PATH, ":")
assert len(preds) == len(truth), f"count mismatch: {len(preds)} vs {len(truth)}"

def get_pred_energy(a):
    for k in ("MACE_energy", "mace_energy", "pred_energy", "energy"):
        if k in a.info:
            return a.info[k]
    if a.calc is not None:
        try:
            return a.get_potential_energy()
        except Exception:
            pass
    return None

def get_pred_forces(a):
    for k in ("MACE_forces", "mace_forces", "pred_forces", "forces"):
        if k in a.arrays:
            return a.arrays[k]
    if a.calc is not None:
        try:
            return a.get_forces()
        except Exception:
            pass
    return None

e_pred, e_true, f_errs = [], [], []
n_skipped = 0
for p, t in zip(preds, truth):
    ep = get_pred_energy(p)
    et = t.info.get("ref_energy")
    fp = get_pred_forces(p)
    ft = t.arrays.get("ref_forces")
    if ep is None or et is None or fp is None or ft is None:
        n_skipped += 1
        continue
    e_pred.append(ep / len(p))
    e_true.append(et / len(t))
    f_errs.append((np.asarray(fp) - np.asarray(ft)).flatten())

if n_skipped > 0:
    print(f"WARN: skipped {n_skipped} structures (missing fields)")

e_pred = np.array(e_pred)
e_true = np.array(e_true)
de = e_pred - e_true

mae_e_raw = float(np.mean(np.abs(de))) * 1000
rmse_e_raw = float(np.sqrt(np.mean(de**2))) * 1000
mae_e_ctr = float(np.mean(np.abs(de - de.mean()))) * 1000
rmse_e_ctr = float(np.sqrt(np.mean((de - de.mean())**2))) * 1000

f_all = np.concatenate(f_errs)
mae_f = float(np.mean(np.abs(f_all))) * 1000
rmse_f = float(np.sqrt(np.mean(f_all**2))) * 1000

if rmse_f < 100:
    force_verdict = "GOOD"
elif rmse_f < 300:
    force_verdict = "PASS"
else:
    force_verdict = "FAIL"

if mae_e_raw >= 500:
    e_note = f"raw energy MAE high ({mae_e_raw:.0f}); centered = {mae_e_ctr:.1f} (likely reference offset)"
else:
    e_note = f"energy MAE in reasonable range; centered = {mae_e_ctr:.1f}"

verdict = force_verdict

print(f"=== {MODEL_LABEL} zero-shot on rMD17 aspirin ===")
print(f"N structures: {len(e_pred)}")
print(f"")
print(f"Force MAE:   {mae_f:.2f} meV/A")
print(f"Force RMSE:  {rmse_f:.2f} meV/A    (verdict: {force_verdict})")
print(f"")
print(f"Energy MAE (raw):      {mae_e_raw:.2f} meV/atom")
print(f"Energy RMSE (raw):     {rmse_e_raw:.2f} meV/atom")
print(f"Energy MAE (centered): {mae_e_ctr:.2f} meV/atom  <- offset-corrected")
print(f"Energy RMSE (centered):{rmse_e_ctr:.2f} meV/atom")
print(f"")
print(f"Note: {e_note}")
print(f"")
print(f"OVERALL VERDICT: {verdict}")

for k, v in [
    ("force_mae_meV_per_ang", mae_f),
    ("force_rmse_meV_per_ang", rmse_f),
    ("energy_mae_meV_per_atom_raw", mae_e_raw),
    ("energy_rmse_meV_per_atom_raw", rmse_e_raw),
    ("energy_mae_meV_per_atom_centered", mae_e_ctr),
    ("energy_rmse_meV_per_atom_centered", rmse_e_ctr),
    ("verdict", verdict),
    ("force_verdict", force_verdict),
    ("n_structures", len(e_pred)),
]:
    subprocess.run(["python", "time_logger.py", "set_metric", k,
                    f"{v:.4f}" if isinstance(v, float) else str(v)])

is_off = "OFF" in MODEL_LABEL.upper()
if is_off:
    ref_ctx = (
        "MACE-OFF23 is trained on organic molecules (SPICE), so rMD17 aspirin is IN-domain. "
        "Expected force RMSE ~10-30 meV/A (Kovacs et al. 2023 MACE-OFF). This is the 'comfort zone' "
        "contrast to MACE-MP-0 (inorganic Materials Project training domain)."
    )
else:
    ref_ctx = (
        "Trained-from-scratch MACE on rMD17 aspirin reports ~5 meV/atom energy, "
        "~20-50 meV/A force (Kovacs et al. JCP 2023). MACE-MP-0 medium zero-shot "
        "on organic molecules expected ~100-300 meV/A force RMSE since training "
        "domain is primarily inorganic Materials Project."
    )

out = {
    "verdict": verdict,
    "model": MODEL_LABEL,
    "primary_metric": "force_rmse_meV_per_ang",
    "n_structures": len(e_pred),
    "force": {
        "mae_meV_per_ang": round(mae_f, 4),
        "rmse_meV_per_ang": round(rmse_f, 4),
    },
    "energy_per_atom_meV": {
        "mae_raw": round(mae_e_raw, 4),
        "rmse_raw": round(rmse_e_raw, 4),
        "mae_centered": round(mae_e_ctr, 4),
        "rmse_centered": round(rmse_e_ctr, 4),
    },
    "energy_note": e_note,
    "thresholds": {
        "force_GOOD_rmse": "< 100 meV/A",
        "force_PASS_rmse": "< 300 meV/A",
    },
    "reference_context": ref_ctx,
    "interpretation": (
        f"Zero-shot inference of {MODEL_LABEL} on rMD17 aspirin. "
        "Pipeline pass means the model loads, runs batch inference via mace_eval_configs CLI, "
        "and produces predictions with reasonable physical magnitude."
    ),
}
json.dump(out, open(OUT_PATH, "w"), indent=2)
print(f"\nWrote {OUT_PATH}")
