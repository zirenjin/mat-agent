"""Post-fine-tuning force-error correlation between FT-MP0 and FT-OFF23, vs the zero-shot baseline.
Question (task): are the two models' force errors still ~independent (r~0.17) or now collinear (r>0.8)?
Zero-shot r is recomputed on the SAME 50 test frames (mapped by mol_index) for an apples-to-apples delta.
Usage: python pearson_ft.py <mp0_preds.npz> <off23_preds.npz> <out_json>"""
import sys, json, numpy as np
from ase.io import read

mp_npz, off_npz, out_json = sys.argv[1:4]
ZS_MP  = f"{__import__('os').path.dirname(__file__)}/../runs/mace_rmd17/predictions.xyz"
ZS_OFF = f"{__import__('os').path.dirname(__file__)}/../runs/maceoff_rmd17/predictions.xyz"
TRUTH1K = f"{__import__('os').path.dirname(__file__)}/../data/rmd17_aspirin_1k/test.extxyz"

def pearson(a, b): return float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0,1])

mp  = np.load(mp_npz);  off = np.load(off_npz)
assert np.array_equal(mp["mol_idx"], off["mol_idx"]), "test-frame mol_idx mismatch between branches"
F_ref = mp["F_ref"]; mol50 = mp["mol_idx"]
assert np.allclose(F_ref, off["F_ref"]), "ref forces differ between branches"

# ---- post-FT errors on the 50 test frames ----
e_mp_ft  = (mp["F_pred"]  - F_ref).reshape(-1)*1000.0    # meV/A
e_off_ft = (off["F_pred"] - F_ref).reshape(-1)*1000.0
r_ft_50  = pearson(e_mp_ft, e_off_ft)

# ---- zero-shot baseline, mapped to the SAME 50 frames by mol_index ----
truth = read(TRUTH1K, ":")                      # 1000 frames, positionally aligned to zero-shot preds
mp_zs_all  = read(ZS_MP, ":"); off_zs_all = read(ZS_OFF, ":")
assert len(truth)==len(mp_zs_all)==len(off_zs_all)==1000
midx = np.array([int(a.info["mol_index"]) for a in truth])
pos = {int(m): i for i, m in enumerate(midx)}
Fmp_zs = np.stack([mp_zs_all[pos[int(m)]].arrays["MACE_forces"]  for m in mol50])
Foff_zs= np.stack([off_zs_all[pos[int(m)]].arrays["MACE_forces"] for m in mol50])
Fref_zs= np.stack([truth[pos[int(m)]].arrays["ref_forces"]       for m in mol50])
assert np.allclose(Fref_zs, F_ref, atol=1e-6), "zero-shot/FT ref-force frame mapping mismatch"
e_mp_zs  = (Fmp_zs  - Fref_zs).reshape(-1)*1000.0
e_off_zs = (Foff_zs - Fref_zs).reshape(-1)*1000.0
r_zs_50  = pearson(e_mp_zs, e_off_zs)

# ---- zero-shot r on the full 1000 (reproduce the ~0.17 reference) ----
Fmp_all = np.stack([a.arrays["MACE_forces"] for a in mp_zs_all])
Foff_all= np.stack([a.arrays["MACE_forces"] for a in off_zs_all])
Fref_all= np.stack([a.arrays["ref_forces"] for a in truth])
r_zs_1000 = pearson(Fmp_all-Fref_all, Foff_all-Fref_all)

def mae(x): return round(float(np.mean(np.abs(x))),3)
regime = ("highly COLLINEAR (errors now aligned)" if r_ft_50 > 0.8 else
          "still roughly INDEPENDENT" if r_ft_50 < 0.3 else "PARTIALLY correlated")
out = {
  "analysis": "Force-error Pearson r between FT-MP0 and FT-OFF23 (post fine-tuning) vs zero-shot",
  "n_test_frames": int(len(mol50)), "n_force_components": int(e_mp_ft.size),
  "zero_shot": {"r_force_error_1000frames": round(r_zs_1000,4),
                "r_force_error_50testframes": round(r_zs_50,4),
                "mp0_force_mae_meV_per_ang": mae(e_mp_zs), "off23_force_mae_meV_per_ang": mae(e_off_zs)},
  "post_finetune": {"r_force_error_50testframes": round(r_ft_50,4),
                    "ft_mp0_force_mae_meV_per_ang": mae(e_mp_ft), "ft_off23_force_mae_meV_per_ang": mae(e_off_ft)},
  "delta_r": round(r_ft_50 - r_zs_50, 4),
  "regime_post_ft": regime,
}
json.dump(out, open(out_json,"w"), indent=2)
print(json.dumps(out, indent=2))
print(f"\nZero-shot r (1000): {r_zs_1000:.3f} | zero-shot r (50): {r_zs_50:.3f} | POST-FT r (50): {r_ft_50:.3f}  -> {regime}")
