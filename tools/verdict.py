"""
verdict.py — Compute force/energy RMSE and emit a unified verdict JSON.

All evaluate.sh wrappers call this script. It is model-agnostic: it reads
pred_energy / pred_forces from predictions and ref_energy / ref_forces from
ground truth, using the helpers in ase_conventions.py.

Thresholds (default):
    force_GOOD_rmse  < 100 meV/A
    force_PASS_rmse  < 300 meV/A

Override with --threshold-good / --threshold-pass. For OC20, use 60 / 120.

Usage:
    python tools/verdict.py \
        --predictions /path/to/preds.xyz \
        --truth /path/to/truth.extxyz \
        --output /path/to/verdict.json \
        --model-label "MACE-MP-0 medium" \
        [--threshold-good 100] \
        [--threshold-pass 300] \
        [--dataset-label "rMD17 aspirin"] \
        [--reference-context "..." ]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Import the shared field helpers from the tools directory
# (When called from evaluate.sh, PYTHONPATH includes repo root)
try:
    from ase_conventions import (
        get_pred_energy,
        get_pred_forces,
        get_ref_energy,
        get_ref_forces,
    )
except ImportError:
    # Fallback for when tools/ is not on path — inline the logic
    import warnings
    warnings.warn("ase_conventions not importable; using inline fallbacks")
    def _get_info_key(atoms, *keys):
        for k in keys:
            if k in atoms.info:
                return float(atoms.info[k])
        return None
    def _get_array_key(atoms, *keys):
        for k in keys:
            if k in atoms.arrays:
                return np.asarray(atoms.arrays[k], dtype=np.float64)
        return None
    get_pred_energy = lambda a: _get_info_key(a, "pred_energy", "MACE_energy", "mace_energy", "dpmd_energy", "energy")
    get_pred_forces = lambda a: _get_array_key(a, "pred_forces", "MACE_forces", "mace_forces", "dpmd_forces", "forces")
    get_ref_energy  = lambda a: _get_info_key(a, "ref_energy", "energy")
    get_ref_forces  = lambda a: _get_array_key(a, "ref_forces", "forces")


def parse_args():
    p = argparse.ArgumentParser(description="Compute force/energy metrics and emit verdict JSON")
    p.add_argument("--predictions", required=True, help="extxyz from inference (pred_energy/pred_forces)")
    p.add_argument("--truth", required=True, help="extxyz with ref_energy/ref_forces")
    p.add_argument("--output", required=True, help="path for verdict JSON")
    p.add_argument("--model-label", required=True, help="human-readable model name")
    p.add_argument("--dataset-label", default="", help="dataset description")
    p.add_argument("--threshold-good", type=float, default=100.0, help="force RMSE upper bound for GOOD (meV/A)")
    p.add_argument("--threshold-pass", type=float, default=300.0, help="force RMSE upper bound for PASS (meV/A)")
    p.add_argument("--reference-context", default="", help="paragraph about model training domain")
    p.add_argument("--interpretation", default="", help="paragraph about what this result means")
    return p.parse_args()


def main():
    args = parse_args()

    from ase.io import read

    preds = read(args.predictions, ":")
    truths = read(args.truth, ":")
    print(f"Loaded {len(preds)} predicted + {len(truths)} truth structures")

    if len(preds) != len(truths):
        print(f"WARN: count mismatch — preds={len(preds)} truth={len(truths)}; "
              f"truncating to min({len(preds)},{len(truths)})")
        n = min(len(preds), len(truths))
        preds = preds[:n]
        truths = truths[:n]

    # Collect energy + force pairs
    e_pred, e_true, f_errs = [], [], []
    n_skipped = 0

    for p, t in zip(preds, truths):
        ep = get_pred_energy(p)
        et = get_ref_energy(t)
        fp = get_pred_forces(p)
        ft = get_ref_forces(t)

        if ep is None or et is None or fp is None or ft is None:
            n_skipped += 1
            continue

        e_pred.append(ep / len(p))
        e_true.append(et / len(t))
        f_errs.append((fp - ft).flatten())

    if n_skipped > 0:
        print(f"WARN: skipped {n_skipped} structures (missing fields)")

    if not e_pred:
        print("FATAL: no valid structure pairs found", file=sys.stderr)
        sys.exit(3)

    e_pred = np.array(e_pred)
    e_true = np.array(e_true)
    de = e_pred - e_true

    # Energy metrics
    mae_e_raw = float(np.mean(np.abs(de))) * 1000
    rmse_e_raw = float(np.sqrt(np.mean(de ** 2))) * 1000
    mae_e_ctr = float(np.mean(np.abs(de - de.mean()))) * 1000
    rmse_e_ctr = float(np.sqrt(np.mean((de - de.mean()) ** 2))) * 1000

    # Force metrics
    f_all = np.concatenate(f_errs)
    mae_f = float(np.mean(np.abs(f_all))) * 1000
    rmse_f = float(np.sqrt(np.mean(f_all ** 2))) * 1000

    # OC20-style per-tag metrics (if tags present in ground truth)
    force_by_tag = None
    force_free_atoms = None
    force_all_atoms = None
    fixed_atom_fraction_pct = None

    if "tags" in truths[0].arrays:
        tag_errors: dict[int, list] = {}
        for p, t in zip(preds, truths):
            fp = get_pred_forces(p)
            ft = get_ref_forces(t)
            tags = t.arrays.get("tags")
            if fp is None or ft is None or tags is None:
                continue
            for tag in np.unique(tags):
                mask = tags == tag
                tag_errors.setdefault(int(tag), []).append((fp[mask] - ft[mask]).flatten())

        force_by_tag = {}
        for tag, errs in sorted(tag_errors.items()):
            flat = np.concatenate(errs)
            force_by_tag[f"tag_{tag}"] = {
                "tag": tag,
                "n_atoms": len(flat) // 3,
                "mae_meV_per_ang": round(float(np.mean(np.abs(flat))) * 1000, 4),
                "rmse_meV_per_ang": round(float(np.sqrt(np.mean(flat ** 2))) * 1000, 4),
            }

        # Free atoms (tags 1+2 in OC20) vs all
        free_tags = {1, 2}
        free_errs = []
        all_errs = []
        n_free = n_all = 0
        for err_list, tag_val in zip(tag_errors.values(), tag_errors.keys()):
            for err in err_list[tag_val] if isinstance(err_list, dict) else []:
                pass  # logic above already split
        # Re-compute from tag_errors
        for tag, errs_list in tag_errors.items():
            flat = np.concatenate(errs_list)
            n_all += len(flat) // 3
            if int(tag) in free_tags:
                free_errs.append(flat)
                n_free += len(flat) // 3
            else:
                all_errs.append(flat)

        if free_errs:
            flat_free = np.concatenate(free_errs)
            force_free_atoms = {
                "mae_meV_per_ang": round(float(np.mean(np.abs(flat_free))) * 1000, 4),
                "rmse_meV_per_ang": round(float(np.sqrt(np.mean(flat_free ** 2))) * 1000, 4),
                "n_atoms": n_free,
                "note": "PRIMARY judge — surface+adsorbate (tags 1,2)",
            }
        if all_errs or free_errs:
            flat_all = np.concatenate(f_errs)  # already all atoms
            force_all_atoms = {
                "mae_meV_per_ang": round(mae_f, 4),
                "rmse_meV_per_ang": round(rmse_f, 4),
                "n_atoms": len(flat_all) // 3,
                "note": ("includes fixed subsurface atoms whose DFT reference forces are zeroed; "
                         "inflated and NOT the standard OC20 force metric"),
            }
            total_atoms = sum(len(t) for t in truths)
            n_fixed = total_atoms - n_free
            fixed_atom_fraction_pct = round(n_fixed / total_atoms * 100, 1) if total_atoms > 0 else 0.0

    # Verdict
    if rmse_f < args.threshold_good:
        force_verdict = "GOOD"
    elif rmse_f < args.threshold_pass:
        force_verdict = "PASS"
    else:
        force_verdict = "FAIL"

    # Energy note
    if mae_e_raw >= 500:
        e_note = (f"raw energy MAE high ({mae_e_raw:.0f} meV/atom); "
                  f"centered = {mae_e_ctr:.1f} (likely reference offset)")
    else:
        e_note = f"energy MAE in reasonable range; centered = {mae_e_ctr:.1f} meV/atom"

    ref_ctx = args.reference_context or (
        f"Zero-shot inference of {args.model_label} on {args.dataset_label or 'unknown dataset'}."
    )
    interp = args.interpretation or (
        f"Force RMSE = {rmse_f:.1f} meV/A — "
        f"{'within expected range for zero-shot transfer' if force_verdict != 'FAIL' else 'above expected range'}."
    )

    # --- Console report ---
    dataset_label = args.dataset_label or "unknown"
    print("\n=== {} on {} ===".format(args.model_label, dataset_label))
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
    print(f"OVERALL VERDICT: {force_verdict}")

    # --- Build output dict ---
    out: dict = {
        "verdict": force_verdict,
        "model": args.model_label,
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
            "PASS": args.threshold_pass,
            "GOOD": args.threshold_good,
            "force_GOOD_rmse": f"< {args.threshold_good:.0f} meV/A",
            "force_PASS_rmse": f"< {args.threshold_pass:.0f} meV/A",
        },
        "reference_context": ref_ctx,
        "interpretation": interp,
    }

    if dataset_label := args.dataset_label:
        out["dataset"] = dataset_label

    # OC20 extensions
    if force_free_atoms is not None:
        out["force_free_atoms"] = force_free_atoms
        out["force_all_atoms"] = force_all_atoms
        out["force_by_tag"] = force_by_tag
        out["fixed_atom_fraction_pct"] = fixed_atom_fraction_pct

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
