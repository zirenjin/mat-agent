"""Evaluate a fine-tuned MACE model on the 50 rMD17 test frames + apply the 4x hard gate.
Forces are reference-free (E0-invariant) -> Force MAE is the physical hard gate.
Usage: python eval_ft.py <model.model> <ftest.extxyz> <branch_label> <out_dir>
Writes <out_dir>/<label>_verdict.json and <out_dir>/<label>_preds.npz (per-atom forces for Pearson r)."""
import sys, json, numpy as np
from ase.io import read
from mace.calculators import MACECalculator

model_path, test_file, label, out_dir = sys.argv[1:5]
GATE_MAE = 26.4          # meV/A = 4x the paper's 6.6 from-scratch Force MAE target
PAPER_TARGET = 6.6

atoms = read(test_file, ":")
calc = MACECalculator(model_paths=[model_path], device="cuda", default_dtype="float64")

E_ref, E_pred, nat, mol_idx = [], [], [], []
F_ref_list, F_pred_list = [], []
for a in atoms:
    e_ref = float(a.info["ref_energy"]); f_ref = np.asarray(a.arrays["ref_forces"], float)
    b = a.copy(); b.calc = calc
    e = float(b.get_potential_energy()); f = np.asarray(b.get_forces(), float)
    E_ref.append(e_ref); E_pred.append(e); F_ref_list.append(f_ref); F_pred_list.append(f)
    nat.append(len(a)); mol_idx.append(int(a.info.get("mol_index", -1)))

E_ref = np.array(E_ref); E_pred = np.array(E_pred); nat = np.array(nat); mol_idx = np.array(mol_idx)
F_ref = np.stack(F_ref_list); F_pred = np.stack(F_pred_list)        # (N,21,3)
fr = F_ref.reshape(-1)*1000.0; fp = F_pred.reshape(-1)*1000.0       # meV/A, component-wise

force_mae  = float(np.mean(np.abs(fp-fr)))
force_rmse = float(np.sqrt(np.mean((fp-fr)**2)))
dE = (E_pred - E_ref)*1000.0                                        # meV, total energy
e_mae_total    = float(np.mean(np.abs(dE)))
e_rmse_total   = float(np.sqrt(np.mean(dE**2)))
e_mae_centered = float(np.mean(np.abs(dE - dE.mean())))
e_mae_peratom  = float(np.mean(np.abs(dE/nat)))
force_scale_rms = float(np.sqrt(np.mean(fr**2)))

verdict = "PASS" if force_mae <= GATE_MAE else "HARD_GATE_VIOLATION"
out = {
  "branch": label, "model_path": model_path, "n_test": len(atoms), "n_atoms_per_struct": int(nat[0]),
  "hard_gate": {"metric": "force_mae_meV_per_ang", "threshold_meV_per_ang": GATE_MAE,
                "rule": "4x paper from-scratch target (6.6 meV/A)", "value": round(force_mae,4),
                "verdict": verdict, "margin_to_paper_x": round(force_mae/PAPER_TARGET,2)},
  "force": {"mae_meV_per_ang": round(force_mae,4), "rmse_meV_per_ang": round(force_rmse,4),
            "true_force_scale_rms_meV_per_ang": round(force_scale_rms,2)},
  "energy": {"mae_total_meV": round(e_mae_total,4), "rmse_total_meV": round(e_rmse_total,4),
             "mae_centered_meV": round(e_mae_centered,4), "mae_per_atom_meV": round(e_mae_peratom,4),
             "mean_signed_offset_meV": round(float(dE.mean()),4),
             "note": "E0s=average -> absolute energy on rMD17 reference; total≈centered confirms no constant offset"},
}
json.dump(out, open(f"{out_dir}/{label}_verdict.json","w"), indent=2)
np.savez(f"{out_dir}/{label}_preds.npz", F_pred=F_pred, F_ref=F_ref, E_pred=E_pred, E_ref=E_ref,
         nat=nat, mol_idx=mol_idx, numbers=np.asarray(atoms[0].numbers))
print(json.dumps(out, indent=2))
print(f"[eval] {label}: Force MAE={force_mae:.3f} meV/A  ->  {verdict}  (gate {GATE_MAE})")
