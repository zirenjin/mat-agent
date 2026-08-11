#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import zipfile
from pathlib import Path

import ase.io
import numpy as np
import pandas as pd
from ase import Atoms
from ase.optimize import FIRE
from tqdm import tqdm

E_FORM_DFT = "e_form_per_atom_mp2020_corrected"
EACH_TRUE = "e_above_hull_mp2020_corrected_ppd_mp"
UNIQ_PROTO = "unique_prototype"
MAT_ID = "material_id"


def frechet_cell_filter_cls():
    try:
        from ase.filters import FrechetCellFilter

        return FrechetCellFilter
    except Exception:
        from ase.constraints import FrechetCellFilter

        return FrechetCellFilter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="label=/path/model")
    p.add_argument("--wbm-summary", default="/work/2023-12-13-wbm-summary.csv.gz")
    p.add_argument("--atoms-zip", default="/work/2024-08-04-wbm-initial-atoms.extxyz.zip")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-dir", default="/work/playground/cache/matbench_discovery_mace_f1_relax")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int, default=0)
    p.add_argument("--energy-reference", default="wbm_mp2020_fit", choices=["wbm_mp2020_fit", "mace_e0"])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_models(entries):
    out = []
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"bad model spec {entry!r}")
        label, path = entry.split("=", 1)
        if not Path(path).is_file():
            raise SystemExit(f"missing model {label}: {path}")
        out.append((label, path))
    return out


def fp(path):
    st = Path(path).stat()
    raw = f"{Path(path).resolve()}:{st.st_size}:{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def read_atoms_zip(path, max_samples=0, start_index=0, end_index=0):
    out = {}
    with zipfile.ZipFile(path) as zf:
        names = [name for name in zf.namelist() if name.endswith(".extxyz")]
        if end_index:
            names = names[:end_index]
        if start_index:
            names = names[start_index:]
        if max_samples:
            names = names[:max_samples]
        for name in tqdm(names, desc="read-wbm-atoms"):
            text = zf.read(name).decode()
            atoms = ase.io.read(io.StringIO(text), format="extxyz", index=-1)
            if not isinstance(atoms, Atoms):
                atoms = atoms[-1]
            mat_id = str(atoms.info.get(MAT_ID) or atoms.info.get("mat_id") or Path(name).stem)
            atoms.info[MAT_ID] = mat_id
            out[mat_id] = atoms
    return out


def make_calc(path, device, dtype):
    from mace.calculators import MACECalculator

    return MACECalculator(model_paths=path, device=device, default_dtype=dtype)


def mace_e0_refs(calc):
    from ase.data import chemical_symbols

    model = calc.models[0]
    nums = model.atomic_numbers.detach().cpu().numpy().astype(int).tolist()
    e0 = model.atomic_energies_fn.atomic_energies.detach().cpu().numpy().astype(float).tolist()
    return {chemical_symbols[z]: energy for z, energy in zip(nums, e0)}


def formula_counts(formula):
    counts = {}
    for elem, amount in re.findall(r"([A-Z][a-z]?)([0-9.]+)?", str(formula)):
        counts[elem] = counts.get(elem, 0.0) + (float(amount) if amount else 1.0)
    if not counts:
        raise ValueError(f"failed to parse formula {formula!r}")
    return counts


def fit_wbm_mp2020_refs(wbm):
    elems = sorted({elem for formula in wbm["formula"] for elem in formula_counts(formula)})
    elem_idx = {elem: idx for idx, elem in enumerate(elems)}
    rows = []
    rhs = []
    for _, row in wbm.iterrows():
        counts = formula_counts(row["formula"])
        vec = np.zeros(len(elems))
        for elem, amount in counts.items():
            vec[elem_idx[elem]] = amount
        n_sites = float(row["n_sites"])
        corrected_total = float(row["uncorrected_energy"]) + float(row["e_correction_per_atom_mp2020"]) * n_sites
        ref_sum = corrected_total - float(row[E_FORM_DFT]) * n_sites
        rows.append(vec)
        rhs.append(ref_sum)
    refs, *_ = np.linalg.lstsq(np.vstack(rows), np.array(rhs), rcond=None)
    return dict(zip(elems, refs.tolist()))


def relax_one(atoms0, calc, fmax, max_steps):
    atoms = atoms0.copy()
    atoms.calc = calc
    filt = frechet_cell_filter_cls()(atoms)
    opt = FIRE(filt, logfile=None)
    converged = False
    err = ""
    try:
        converged = bool(opt.run(fmax=fmax, steps=max_steps))
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        max_force = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0
    except Exception as exc:
        energy = math.nan
        max_force = math.nan
        err = repr(exc)
    return atoms, energy, max_force, int(getattr(opt, "nsteps", -1)), converged, err


def predict(label, path, atoms_by_id, refs, cache_dir, device, dtype, fmax, max_steps, overwrite):
    cache = Path(cache_dir) / (
        f"{label}_{fp(path)}_{device}_{dtype}_FIRE_FCF_fmax={fmax}_steps={max_steps}.csv.gz"
    )
    calc = make_calc(path, device, dtype)
    if cache.is_file() and not overwrite:
        print(f"[mbd-f1-relax] cache hit {cache}", flush=True)
        df = pd.read_csv(cache)
    else:
        rows = []
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(cache.suffix + ".tmp")
        for idx, (mid, atoms0) in enumerate(tqdm(atoms_by_id.items(), desc=f"relax/{label}"), start=1):
            atoms, energy, max_force, n_steps, converged, err = relax_one(atoms0, calc, fmax, max_steps)
            rows.append(
                {
                    MAT_ID: mid,
                    "energy": energy,
                    "n_atoms": len(atoms),
                    "formula": atoms.get_chemical_formula(),
                    "max_force": max_force,
                    "n_steps": n_steps,
                    "converged": converged,
                    "error": err,
                }
            )
            if idx % 100 == 0:
                pd.DataFrame(rows).to_csv(tmp, index=False)
        df = pd.DataFrame(rows)
        df.to_csv(cache, index=False)
        if tmp.is_file():
            tmp.unlink()
    eform = []
    for _, row in df.iterrows():
        atoms = atoms_by_id[row[MAT_ID]]
        if not np.isfinite(row.energy):
            eform.append(math.nan)
            continue
        missing = sorted(set(atoms.symbols) - set(refs))
        if missing:
            eform.append(math.nan)
            continue
        ref = sum(refs[symbol] for symbol in atoms.symbols)
        eform.append((float(row.energy) - ref) / len(atoms))
    df["e_form_per_atom_pred"] = eform
    return df


def stable_metrics(each_true, each_pred):
    true = pd.to_numeric(pd.Series(each_true), errors="coerce")
    pred = pd.to_numeric(pd.Series(each_pred), errors="coerce")
    actual_pos = true <= 0
    actual_neg = true > 0
    model_pos = pred <= 0
    model_neg = pred > 0
    nan = pred.isna()
    model_pos[nan] = False
    model_neg[nan] = True
    tp = int((actual_pos & model_pos).sum())
    fn = int((actual_pos & model_neg).sum())
    fp = int((actual_neg & model_pos).sum())
    tn = int((actual_neg & model_neg).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    valid = ~(true.isna() | pred.isna())
    err = (pred[valid] - true[valid]).to_numpy()
    return dict(
        F1=f1,
        Precision=precision,
        Recall=recall,
        Accuracy=(tp + tn) / (tp + tn + fp + fn),
        TP=tp,
        FP=fp,
        TN=tn,
        FN=fn,
        MAE=float(np.abs(err).mean()),
        RMSE=float(np.sqrt((err * err).mean())),
        missing_preds=int(pred.isna().sum()),
    )


def metrics_for(wbm, pred):
    df = wbm.copy()
    df["e_form_per_atom_pred"] = pred
    df["each_pred"] = df[EACH_TRUE] + df["e_form_per_atom_pred"] - df[E_FORM_DFT]
    subsets = {
        "full_test_set": df,
        "unique_prototypes": df[df[UNIQ_PROTO].astype(bool)],
        "most_stable_10k": df.nsmallest(min(10000, len(df)), "each_pred"),
    }
    out = {}
    for name, sub in subsets.items():
        metrics = stable_metrics(sub[EACH_TRUE], sub["each_pred"])
        metrics["n"] = int(len(sub))
        out[name] = metrics
    return out


def main():
    args = parse_args()
    models = parse_models(args.models)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atoms_all = read_atoms_zip(args.atoms_zip, args.max_samples, args.start_index, args.end_index)
    wbm_all = pd.read_csv(args.wbm_summary).set_index(MAT_ID)
    ids = [mid for mid in atoms_all if mid in wbm_all.index]
    atoms = {mid: atoms_all[mid] for mid in ids}
    wbm = wbm_all.loc[ids]
    wbm_refs = fit_wbm_mp2020_refs(wbm_all.reset_index()) if args.energy_reference == "wbm_mp2020_fit" else None
    summary = {
        "n_structures": len(wbm),
        "n_atoms_without_summary": len(atoms_all) - len(atoms),
        "wbm_summary": args.wbm_summary,
        "atoms_zip": args.atoms_zip,
        "energy_reference": args.energy_reference,
        "relaxation": {
            "test_task": "IS2RE-SR",
            "ase_optimizer": "FIRE",
            "cell_filter": "FrechetCellFilter",
            "fmax": args.fmax,
            "max_steps": args.max_steps,
        },
        "models": {},
    }
    for label, path in models:
        if args.energy_reference == "mace_e0":
            calc = make_calc(path, args.device, args.dtype)
            refs_for_model = mace_e0_refs(calc)
            del calc
        else:
            refs_for_model = wbm_refs
        dfp = predict(
            label,
            path,
            atoms,
            refs_for_model,
            args.cache_dir,
            args.device,
            args.dtype,
            args.fmax,
            args.max_steps,
            args.overwrite,
        )
        pred = dfp.set_index(MAT_ID).reindex(wbm.index)["e_form_per_atom_pred"]
        pred_file = out_dir / f"{label}-wbm-IS2RE-FIRE.csv.gz"
        pred.reset_index().to_csv(pred_file, index=False)
        metrics = metrics_for(wbm, pred)
        summary["models"][label] = {"path": path, "pred_file": str(pred_file), "metrics": metrics}
        print(
            f"[mbd-f1-relax] {label} F1 full={metrics['full_test_set']['F1']:.4f} "
            f"unique={metrics['unique_prototypes']['F1']:.4f} "
            f"top10k={metrics['most_stable_10k']['F1']:.4f}",
            flush=True,
        )
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[mbd-f1-relax] wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
