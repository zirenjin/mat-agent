"""
mp_vs_off_force_delta.py — direct model-vs-model force comparison on rMD17 aspirin.

delta = F_MP - F_OFF  (prediction vs prediction; NOT vs ref)
- component-wise MAE / RMSE (meV/A), matching verdict.py convention
- broken down by element (C/H/O)
- diagnostics: sanity RMSE vs ref (should reproduce 337.5 / 447.2),
  error-correlation r(e_MP, e_OFF), and the analytic bounds for RMSE(delta).
"""
import json
import numpy as np
from ase.io import read

MP = read("runs/mace_rmd17/predictions.xyz", ":")
OFF = read("runs/maceoff_rmd17/predictions.xyz", ":")
TRUTH = read("data/rmd17_aspirin_1k/test.extxyz", ":")
assert len(MP) == len(OFF) == len(TRUTH), f"count mismatch {len(MP)}/{len(OFF)}/{len(TRUTH)}"

# Verify identical structures / atom ordering across the two prediction files.
Z = np.asarray(MP[0].numbers)
for a, b in zip(MP, OFF):
    assert np.array_equal(a.numbers, b.numbers) and np.array_equal(a.numbers, Z), "atom-order mismatch"

F_mp = np.array([a.arrays["MACE_forces"] for a in MP])   # (N,21,3) eV/A
F_off = np.array([a.arrays["MACE_forces"] for a in OFF])  # (N,21,3)
F_ref = np.array([a.arrays["ref_forces"] for a in TRUTH]) # (N,21,3)
N, NA, _ = F_mp.shape

delta = F_mp - F_off          # model - model
e_mp = F_mp - F_ref           # MP error vs PBE ref
e_off = F_off - F_ref         # OFF error vs PBE ref

def stats(arr_eVA):
    c = arr_eVA.reshape(-1) * 1000.0  # meV/A, component-wise
    return {
        "mae_meV_per_ang": round(float(np.mean(np.abs(c))), 4),
        "rmse_meV_per_ang": round(float(np.sqrt(np.mean(c**2))), 4),
        "mean_signed_meV_per_ang": round(float(np.mean(c)), 4),
        "n_components": int(c.size),
    }

ELEMENTS = {1: "H", 6: "C", 8: "O"}
by_el = {}
for z, sym in ELEMENTS.items():
    mask = (Z == z)
    if not mask.any():
        continue
    d = delta[:, mask, :]
    by_el[sym] = {"n_atoms_per_struct": int(mask.sum()),
                  "n_atoms_total": int(mask.sum()) * N,
                  **stats(d)}

overall = stats(delta)
mp_vs_ref = stats(e_mp)
off_vs_ref = stats(e_off)

# Force scale for context (RMS magnitude of the true PBE forces).
force_scale_rms = round(float(np.sqrt(np.mean((F_ref.reshape(-1) * 1000.0) ** 2))), 2)

# Error correlation (component-wise) between the two models' errors vs ref.
emp_c = e_mp.reshape(-1) * 1000.0
eoff_c = e_off.reshape(-1) * 1000.0
r_err = float(np.corrcoef(emp_c, eoff_c)[0, 1])
# Correlation between the two raw force fields.
r_force = float(np.corrcoef(F_mp.reshape(-1), F_off.reshape(-1))[0, 1])

mp_r = mp_vs_ref["rmse_meV_per_ang"]
off_r = off_vs_ref["rmse_meV_per_ang"]
actual = overall["rmse_meV_per_ang"]
bound_collinear = round(abs(off_r - mp_r), 2)                 # r(e)=1  -> |off-mp|
bound_orthogonal = round(float(np.sqrt(max(off_r**2 - mp_r**2, 0.0))), 2)  # e_off=e_mp+indep
bound_independent = round(float(np.sqrt(mp_r**2 + off_r**2)), 2)            # r(e)=0

# Which regime is the actual delta closest to?
cands = {"~|off-mp| (errors collinear, r_e=1)": bound_collinear,
         "~sqrt(off^2-mp^2) (off=mp+orthogonal)": bound_orthogonal,
         "~sqrt(mp^2+off^2) (errors independent, r_e=0)": bound_independent}
closest = min(cands, key=lambda k: abs(cands[k] - actual))

answer_110 = abs(actual - bound_collinear) <= 30  # "close to 447-337≈110"?
if answer_110:
    interp = (
        f"RMSE(delta)={actual:.1f} ~ |off-mp|={bound_collinear:.1f} meV/A. The two models' "
        f"errors-vs-ref are highly COLLINEAR (r_e={r_err:.3f}): OFF behaves like MP plus an "
        f"aligned systematic offset. Consistent with a clean functional shift; OFF has no large "
        f"independent error component."
    )
else:
    interp = (
        f"RMSE(delta)={actual:.1f} meV/A is NOT close to the naive 447-337={bound_collinear:.1f}. "
        f"RMSEs don't subtract linearly: the collinear limit is {bound_collinear:.1f}, the "
        f"orthogonal-extra limit sqrt(off^2-mp^2)={bound_orthogonal:.1f}, the independent limit "
        f"sqrt(mp^2+off^2)={bound_independent:.1f}. Actual is closest to '{closest}'. "
        f"Error correlation r(e_MP,e_OFF)={r_err:.3f}, raw-force correlation r(F_MP,F_OFF)={r_force:.3f}. "
        f"The MP-OFF force difference ({actual:.0f} meV/A, ~{100*actual/force_scale_rms:.0f}% of the "
        f"{force_scale_rms:.0f} meV/A force scale) is a real, structured PBE-vs-wB97M-D3 difference, "
        f"not a small clean offset."
    )

out = {
    "analysis": "MP-0 vs OFF23 direct force comparison on rMD17 aspirin (model-vs-model, not vs ref)",
    "n_structures": N,
    "n_atoms_per_struct": NA,
    "molecule_composition": {ELEMENTS[z]: int((Z == z).sum()) for z in ELEMENTS if (Z == z).any()},
    "convention": "component-wise over all structs*atoms*xyz; meV/A; delta = F_MP - F_OFF",
    "delta_F_MP_minus_F_OFF": {"overall": overall, "by_element": by_el},
    "sanity_vs_ref": {
        "MP_force_rmse_meV_per_ang": mp_r,
        "MP_force_mae_meV_per_ang": mp_vs_ref["mae_meV_per_ang"],
        "OFF_force_rmse_meV_per_ang": off_r,
        "OFF_force_mae_meV_per_ang": off_vs_ref["mae_meV_per_ang"],
        "note": "should reproduce the verdict JSONs (MP 337.5 / OFF 447.2)",
    },
    "true_force_scale_rms_meV_per_ang": force_scale_rms,
    "hypothesis_check": {
        "question": "Is RMSE(F_MP - F_OFF) close to 447-337 ~ 110 meV/A?",
        "actual_delta_rmse_meV_per_ang": actual,
        "bound_collinear_errors_r1": bound_collinear,
        "bound_orthogonal_extra": bound_orthogonal,
        "bound_independent_errors_r0": bound_independent,
        "closest_regime": closest,
        "pearson_r_error_MP_vs_OFF": round(r_err, 4),
        "pearson_r_force_MP_vs_OFF": round(r_force, 4),
        "answer_close_to_110": bool(answer_110),
        "interpretation": interp,
    },
}
json.dump(out, open("results/mp_vs_off_force_delta.json", "w"), indent=2)

print(f"N={N}, atoms/struct={NA}, composition={out['molecule_composition']}")
print(f"delta=F_MP-F_OFF  overall: MAE={overall['mae_meV_per_ang']}  RMSE={overall['rmse_meV_per_ang']} meV/A")
for sym, s in by_el.items():
    print(f"  {sym} (x{s['n_atoms_per_struct']}/struct): MAE={s['mae_meV_per_ang']}  RMSE={s['rmse_meV_per_ang']}  mean={s['mean_signed_meV_per_ang']}")
print(f"sanity vs ref: MP RMSE={mp_r}  OFF RMSE={off_r}  (force scale RMS={force_scale_rms})")
print(f"bounds: collinear|off-mp|={bound_collinear}  orthogonal={bound_orthogonal}  independent={bound_independent}")
print(f"actual delta RMSE={actual}  -> closest: {closest}")
print(f"r(e_MP,e_OFF)={r_err:.4f}  r(F_MP,F_OFF)={r_force:.4f}  close_to_110={answer_110}")
print("Wrote results/mp_vs_off_force_delta.json")
