#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from tqdm import tqdm

MAT_ID = "material_id"
E_FORM_DFT = "e_form_per_atom_mp2020_corrected"
EACH_TRUE = "e_above_hull_mp2020_corrected_ppd_mp"
UNIQ_PROTO = "unique_prototype"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Join official-compatible MACE relax JSON and evaluate MBD F1.")
    p.add_argument("--model-name", required=True)
    p.add_argument("--relax-glob", required=True)
    p.add_argument("--wbm-summary", default="/work/third_party/matbench-discovery/data/wbm/2023-12-13-wbm-summary.csv.gz")
    p.add_argument("--wbm-cse", default="/work/third_party/matbench-discovery/data/wbm/2022-10-19-wbm-computed-structure-entries.jsonl.gz")
    p.add_argument("--mp-elemental-refs", default="/work/third_party/matbench-discovery/data/mp/2023-02-07-mp-elemental-reference-entries.json.gz")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-error-threshold", type=float, default=5.0)
    return p.parse_args()


def calc_energy_from_e_refs(formula: str, ref_energies: dict[str, float], total_energy: float) -> float:
    from pymatgen.core import Composition

    comp = Composition(formula)
    ref = sum(ref_energies[el.symbol] * amt for el, amt in comp.items())
    return (total_energy - ref) / comp.num_atoms


def stable_metrics(each_true: pd.Series, each_pred: pd.Series) -> dict[str, float]:
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
    return {
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err * err).mean())),
        "missing_preds": int(pred.isna().sum()),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path().glob(args.relax_glob) if not args.relax_glob.startswith("/") else Path("/").glob(args.relax_glob[1:]))
    if not files:
        raise SystemExit(f"No files matching {args.relax_glob!r}")
    print(f"Found {len(files):,} relax files", flush=True)

    df_relax = pd.concat(pd.read_json(path, lines=True) for path in tqdm(files, desc="read relax json"))
    df_relax = df_relax.drop_duplicates(MAT_ID, keep="last").set_index(MAT_ID)
    print(f"Relax rows: {len(df_relax):,}", flush=True)

    df_wbm = pd.read_csv(args.wbm_summary).set_index(MAT_ID)
    df_cse = pd.read_json(args.wbm_cse, lines=True).set_index(MAT_ID)
    df_refs = pd.read_json(args.mp_elemental_refs, typ="series")
    ref_entries = [ComputedEntry.from_dict(dct) for dct in df_refs]
    ref_energies = {entry.composition.elements[0].symbol: entry.energy_per_atom for entry in ref_entries}

    cse_by_id = {
        mat_id: ComputedStructureEntry.from_dict(dct)
        for mat_id, dct in tqdm(df_cse["computed_structure_entry"].items(), total=len(df_cse), desc="hydrate WBM CSE")
    }

    entries: list[ComputedStructureEntry] = []
    ids: list[str] = []
    for mat_id, row in tqdm(df_relax.iterrows(), total=len(df_relax), desc="ML energies to CSEs"):
        if row["mace_structure"] is None or not np.isfinite(row["mace_energy"]):
            continue
        cse = cse_by_id[mat_id]
        cse._energy = float(row["mace_energy"])  # noqa: SLF001
        cse._structure = Structure.from_dict(row["mace_structure"])  # noqa: SLF001
        entries.append(cse)
        ids.append(mat_id)

    processed = MaterialsProject2020Compatibility().process_entries(entries, verbose=True, clean=True)
    if len(processed) != len(entries):
        raise SystemExit(f"not all entries processed: {len(processed)=} {len(entries)=}")

    pred = pd.Series(index=df_wbm.index, dtype="float64", name="e_form_per_atom_mace")
    for mat_id, entry in tqdm(zip(ids, processed, strict=True), total=len(ids), desc="formation energies"):
        pred.loc[mat_id] = calc_energy_from_e_refs(df_wbm.loc[mat_id, "formula"], ref_energies, entry.energy)

    if args.max_error_threshold is not None:
        bad = (pred - df_wbm[E_FORM_DFT]).abs() > args.max_error_threshold
        pred.loc[bad] = pd.NA
        print(f"{int(bad.sum()):,} unrealistic preds filtered at {args.max_error_threshold} eV/atom", flush=True)

    df_eval = df_wbm.copy()
    df_eval["e_form_per_atom_mace"] = pred
    df_eval["each_pred"] = df_eval[EACH_TRUE] + df_eval["e_form_per_atom_mace"] - df_eval[E_FORM_DFT]

    uniq = df_eval[df_eval[UNIQ_PROTO].astype(bool)]
    top10k = uniq.nsmallest(min(10_000, len(uniq)), "each_pred")
    metrics = {
        "full_test_set": stable_metrics(df_eval[EACH_TRUE], df_eval["each_pred"]) | {"n": int(len(df_eval))},
        "unique_prototypes": stable_metrics(uniq[EACH_TRUE], uniq["each_pred"]) | {"n": int(len(uniq))},
        "most_stable_10k": stable_metrics(top10k[EACH_TRUE], top10k["each_pred"]) | {"n": int(len(top10k))},
    }

    pred_file = out_dir / f"{args.model_name}-wbm-IS2RE-FIRE-official.csv.gz"
    summary_file = out_dir / f"{args.model_name}-summary.json"
    pred.reset_index().to_csv(pred_file, index=False)
    summary = {
        "model_name": args.model_name,
        "relax_glob": args.relax_glob,
        "n_relax_files": len(files),
        "n_relax_rows": int(len(df_relax)),
        "n_predictions": int(pred.notna().sum()),
        "max_error_threshold": args.max_error_threshold,
        "pred_file": str(pred_file),
        "metrics": metrics,
    }
    summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
