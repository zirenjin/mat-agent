#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, io, json, math, zipfile
from pathlib import Path
from typing import Any
import ase.io
import numpy as np
import pandas as pd
from ase import Atoms
from tqdm import tqdm

E_FORM_DFT = "e_form_per_atom_mp2020_corrected"
EACH_TRUE = "e_above_hull_mp2020_corrected_ppd_mp"
UNIQ_PROTO = "unique_prototype"
MAT_ID = "material_id"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="label=/path/model")
    p.add_argument("--wbm-summary", default="/work/2023-12-13-wbm-summary.csv.gz")
    p.add_argument("--atoms-zip", default="/work/2024-08-04-wbm-initial-atoms.extxyz.zip")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-dir", default="/work/playground/cache/matbench_discovery_mace_f1")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def parse_models(entries):
    out=[]
    for e in entries:
        if "=" not in e: raise SystemExit(f"bad model spec {e!r}")
        label,path=e.split("=",1)
        if not Path(path).is_file(): raise SystemExit(f"missing model {label}: {path}")
        out.append((label,path))
    return out

def fp(path):
    st=Path(path).stat(); raw=f"{Path(path).resolve()}:{st.st_size}:{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]

def read_atoms_zip(path, max_samples=0):
    out={}
    with zipfile.ZipFile(path) as zf:
        names=[n for n in zf.namelist() if n.endswith(".extxyz")]
        if max_samples: names=names[:max_samples]
        for name in tqdm(names, desc="read-wbm-atoms"):
            text=zf.read(name).decode()
            atoms=ase.io.read(io.StringIO(text), format="extxyz", index=-1)
            if not isinstance(atoms, Atoms): atoms=atoms[-1]
            mat_id=str(atoms.info.get(MAT_ID) or atoms.info.get("mat_id") or Path(name).stem)
            atoms.info[MAT_ID]=mat_id
            out[mat_id]=atoms
    return out

def make_calc(path, device, dtype):
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=path, device=device, default_dtype=dtype)

def mace_e0_refs(calc):
    from ase.data import chemical_symbols
    model=calc.models[0]
    nums=model.atomic_numbers.detach().cpu().numpy().astype(int).tolist()
    e0=model.atomic_energies_fn.atomic_energies.detach().cpu().numpy().astype(float).tolist()
    return {chemical_symbols[z]: e for z,e in zip(nums,e0)}

def predict(label, path, atoms_by_id, cache_dir, device, dtype, overwrite):
    cache=Path(cache_dir)/f"{label}_{fp(path)}_{device}_{dtype}_energies.csv.gz"
    calc=make_calc(path, device, dtype)
    refs=mace_e0_refs(calc)
    if cache.is_file() and not overwrite:
        print(f"[mbd-f1] cache hit {cache}", flush=True)
        df=pd.read_csv(cache)
    else:
        rows=[]
        for mid, atoms0 in tqdm(atoms_by_id.items(), desc=f"mace/{label}"):
            atoms=atoms0.copy(); atoms.calc=calc
            try:
                energy=float(atoms.get_potential_energy())
                err=""
            except Exception as exc:
                energy=math.nan; err=repr(exc)
            rows.append({MAT_ID:mid,"energy":energy,"n_atoms":len(atoms),"formula":atoms.get_chemical_formula(),"error":err})
        df=pd.DataFrame(rows); cache.parent.mkdir(parents=True, exist_ok=True); df.to_csv(cache,index=False)
    eform=[]
    for _, row in df.iterrows():
        atoms=atoms_by_id[row[MAT_ID]]
        if not np.isfinite(row.energy):
            eform.append(math.nan); continue
        missing=sorted(set(atoms.symbols)-set(refs))
        if missing:
            eform.append(math.nan)
            continue
        ref=sum(refs[s] for s in atoms.symbols)
        eform.append((float(row.energy)-ref)/len(atoms))
    df["e_form_per_atom_pred"]=eform
    return df

def stable_metrics(each_true, each_pred):
    true=pd.to_numeric(pd.Series(each_true), errors="coerce")
    pred=pd.to_numeric(pd.Series(each_pred), errors="coerce")
    actual_pos=true <= 0; actual_neg=true > 0
    model_pos=pred <= 0; model_neg=pred > 0
    nan=pred.isna(); model_pos[nan]=False; model_neg[nan]=True
    tp=int((actual_pos & model_pos).sum()); fn=int((actual_pos & model_neg).sum())
    fp=int((actual_neg & model_pos).sum()); tn=int((actual_neg & model_neg).sum())
    precision=tp/(tp+fp) if tp+fp else float("nan")
    recall=tp/(tp+fn) if tp+fn else float("nan")
    f1=2*precision*recall/(precision+recall) if precision+recall else float("nan")
    valid=~(true.isna()|pred.isna())
    err=(pred[valid]-true[valid]).to_numpy()
    return dict(F1=f1, Precision=precision, Recall=recall, Accuracy=(tp+tn)/(tp+tn+fp+fn), TP=tp, FP=fp, TN=tn, FN=fn, MAE=float(np.abs(err).mean()), RMSE=float(np.sqrt((err*err).mean())), missing_preds=int(pred.isna().sum()))

def metrics_for(wbm, pred):
    df=wbm.copy(); df["e_form_per_atom_pred"]=pred
    df["each_pred"]=df[EACH_TRUE] + df["e_form_per_atom_pred"] - df[E_FORM_DFT]
    subsets={"full_test_set":df, "unique_prototypes":df[df[UNIQ_PROTO].astype(bool)], "most_stable_10k":df.nsmallest(min(10000,len(df)), "each_pred")}
    out={}
    for name, sub in subsets.items():
        m=stable_metrics(sub[EACH_TRUE], sub["each_pred"]); m["n"]=int(len(sub)); out[name]=m
    return out

def main():
    args=parse_args(); models=parse_models(args.models)
    out_dir=Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    atoms_all=read_atoms_zip(args.atoms_zip, args.max_samples)
    wbm_all=pd.read_csv(args.wbm_summary).set_index(MAT_ID)
    ids=[mid for mid in atoms_all if mid in wbm_all.index]
    atoms={mid: atoms_all[mid] for mid in ids}
    wbm=wbm_all.loc[ids]
    summary={"n_structures":len(wbm), "n_atoms_without_summary":len(atoms_all)-len(atoms), "wbm_summary":args.wbm_summary, "atoms_zip":args.atoms_zip, "energy_reference":"per-model MACE atomic_energies_fn E0", "models":{}}
    for label,path in models:
        dfp=predict(label,path,atoms,args.cache_dir,args.device,args.dtype,args.overwrite)
        pred=dfp.set_index(MAT_ID).reindex(wbm.index)["e_form_per_atom_pred"]
        pred_file=out_dir/f"{label}-wbm-initial-IS2E.csv.gz"
        pred.reset_index().to_csv(pred_file,index=False)
        mets=metrics_for(wbm,pred)
        summary["models"][label]={"path":path,"pred_file":str(pred_file),"metrics":mets}
        full_f1 = mets["full_test_set"]["F1"]
        uniq_f1 = mets["unique_prototypes"]["F1"]
        top_f1 = mets["most_stable_10k"]["F1"]
        print(f"[mbd-f1] {label} F1 full={full_f1:.4f} unique={uniq_f1:.4f} top10k={top_f1:.4f}", flush=True)
        (out_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True))
    print(f"[mbd-f1] wrote {out_dir / 'summary.json'}")
if __name__ == "__main__": main()
