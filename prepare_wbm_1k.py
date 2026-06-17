"""
prepare_wbm_1k.py — Download and slice the WBM dataset down to 1000 structures.

What you get
------------
data/wbm_1k/
├── test.extxyz       # 1000 WBM initial structures with formation_energy + e_above_hull
├── dp_test/          # same 1000 in DeePMD format
├── wbm_summary.csv   # the slice of summary metadata (for F1 / stability calc later)
└── manifest.json

Why initial atoms (not relaxed)?
--------------------------------
The matbench-discovery task is IS2RE-SR: "from Initial Structure to Relaxed Energy,
with Structure Relaxation". So we feed models the *initial* (perturbed) structure and
they should predict the *relaxed* energy. The relaxed energy lives in wbm_summary.

Why no forces in the truth?
---------------------------
WBM only ships initial + final structures + final energies. No forces.
This means for *evaluation*: we compare predicted energy vs DFT formation energy. ✓
For *training/fine-tuning*: WBM is NOT suitable for force training. See note below.

Forces for training
-------------------
If your training SOP needs forces (MACE/DPMD both default to force loss):
  Option A — Energy-only training: set force_weight=0 in MACE config, or
             energy_only loss in DPMD. Works for pipeline test, won't give
             a good model.
  Option B — Use a small MPtrj slice for training (has forces but is in
             foundation models' training set; only OK for pipeline test).
             See `prepare_mptrj_train.py` for that path.

Usage
-----
    pip install matbench-discovery
    python prepare_wbm_1k.py --n 1000 --out data/wbm_1k --seed 42

First run downloads ~50 MB to ~/.cache/matbench-discovery/, subsequent runs are fast.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=1000, help="Number of structures to slice")
    p.add_argument("--out", type=str, default="data/wbm_1k", help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-dp", action="store_true", help="Skip DeePMD conversion")
    p.add_argument("--source", choices=["initial", "relaxed"], default="initial",
                   help="Use initial (perturbed) or relaxed atoms. For IS2RE-SR benchmark use 'initial'.")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ----- 1. Trigger downloads via matbench_discovery -----
    # The package auto-downloads files to ~/.cache/matbench-discovery/ on first .path access.
    print("[wbm_prep] importing matbench_discovery (will download ~50 MB on first run)...")
    try:
        from matbench_discovery.data import DataFiles, df_wbm
    except Exception as e:
        print(f"ERROR: matbench-discovery failed to import / download: {e}", file=sys.stderr)
        print("  - Check network access to figshare.com", file=sys.stderr)
        print("  - Behind GFW? Try export HTTP_PROXY=... or use the manual figshare URLs in this script", file=sys.stderr)
        sys.exit(1)

    print(f"[wbm_prep] df_wbm has {len(df_wbm)} rows")
    print(f"[wbm_prep] columns: {list(df_wbm.columns)}")

    # ----- 2. Sample N rows -----
    rng = np.random.default_rng(args.seed)
    n = min(args.n, len(df_wbm))
    idx = rng.choice(len(df_wbm), size=n, replace=False)
    idx.sort()
    df_sub = df_wbm.iloc[idx].copy()
    print(f"[wbm_prep] sampled {len(df_sub)} rows (seed={args.seed})")

    # Save the sliced summary — needed later for F1 / stability classification
    summary_path = out / "wbm_summary.csv"
    df_sub.to_csv(summary_path)
    print(f"[wbm_prep] wrote {summary_path}")

    # ----- 3. Load structures (initial or relaxed) -----
    from ase.io import iread, write
    import zipfile
    import tempfile

    atoms_zip_file = (
        DataFiles.wbm_initial_atoms.path if args.source == "initial"
        else DataFiles.wbm_relaxed_atoms.path
    )
    print(f"[wbm_prep] reading structures from {atoms_zip_file}")

    # The file is a .zip containing one extxyz inside. Unzip on the fly.
    target_ids = set(df_sub["material_id"].astype(str)) if "material_id" in df_sub.columns \
                 else set(df_sub.index.astype(str))
    print(f"[wbm_prep] looking for {len(target_ids)} material IDs")

    # Detect zip layout: one big extxyz vs per-structure files ({material_id}.extxyz)
    with zipfile.ZipFile(atoms_zip_file) as zf:
        all_names = zf.namelist()
        inner_files = [n for n in all_names if n.endswith((".extxyz", ".xyz"))]

    kept = []
    seen_count = 0

    if len(inner_files) == 1:
        # Legacy layout: single extxyz, iterate frames and match by material_id
        with zipfile.ZipFile(atoms_zip_file) as zf:
            inner = inner_files[0]
            print(f"[wbm_prep] single-file layout: reading {inner}")
            with tempfile.NamedTemporaryFile(suffix=".extxyz", delete=False) as tmp:
                tmp.write(zf.read(inner))
                tmp_path = tmp.name
        for atoms in iread(tmp_path, format="extxyz"):
            seen_count += 1
            mid = str(atoms.info.get("material_id") or atoms.info.get("mat_id") or "")
            if mid in target_ids:
                atoms.info["material_id"] = mid
                kept.append(atoms)
                if len(kept) >= len(target_ids):
                    break
            if seen_count % 50000 == 0:
                print(f"  scanned {seen_count} / found {len(kept)}")
    else:
        # New layout: one file per structure named {material_id}.extxyz
        print(f"[wbm_prep] per-structure layout: {len(inner_files)} files in zip")
        name_map = {n.replace(".extxyz", "").replace(".xyz", ""): n for n in inner_files}
        with zipfile.ZipFile(atoms_zip_file) as zf:
            for mid in sorted(target_ids):
                fname = name_map.get(mid)
                if fname is None:
                    print(f"  warn: {mid} not found in zip", file=sys.stderr)
                    continue
                seen_count += 1
                with tempfile.NamedTemporaryFile(suffix=".extxyz", delete=False) as tmp:
                    tmp.write(zf.read(fname))
                    tmp_path = tmp.name
                try:
                    atoms_in = list(iread(tmp_path, format="extxyz"))
                    atoms = atoms_in[0] if atoms_in else None
                finally:
                    import os as _os; _os.unlink(tmp_path)
                if atoms is None:
                    continue
                atoms.info["material_id"] = mid
                # Attach formation energy + e_above_hull from summary
                try:
                    row = df_sub.loc[mid] if mid in df_sub.index else \
                          df_sub.loc[df_sub["material_id"] == mid].iloc[0]
                    for col in ("e_form_per_atom_mp2020_corrected", "uncorrected_energy_per_atom"):
                        if col in row.index:
                            atoms.info["e_form_per_atom"] = float(row[col])
                            atoms.info["energy"] = float(row[col]) * len(atoms)
                            break
                    for col in ("e_above_hull_mp2020_corrected_ppd_mp", "e_above_hull_wbm"):
                        if col in row.index:
                            atoms.info["e_above_hull"] = float(row[col])
                            break
                except Exception as e:
                    print(f"  warn: metadata for {mid}: {e}", file=sys.stderr)
                kept.append(atoms)
                if len(kept) % 100 == 0:
                    print(f"  loaded {len(kept)}/{len(target_ids)}")

    print(f"[wbm_prep] matched {len(kept)} structures (scanned {seen_count} total)")
    if len(kept) < len(target_ids) * 0.9:
        print(f"WARNING: only matched {len(kept)} of {len(target_ids)} expected. "
              f"material_id column name may have changed.", file=sys.stderr)

    # ----- 4. Write extxyz -----
    extxyz_path = out / "test.extxyz"
    write(str(extxyz_path), kept, format="extxyz")
    print(f"[wbm_prep] wrote {extxyz_path} ({len(kept)} structures)")

    # ----- 5. (optional) DeePMD format conversion -----
    dp_dir = None
    if not args.no_dp:
        try:
            import dpdata
            dp_dir = out / "dp_test"
            ms = dpdata.MultiSystems.from_file(str(extxyz_path), fmt="quip/gap/xyz")
            ms.to_deepmd_npy(str(dp_dir))
            n_frames = sum(s.get_nframes() for s in ms)
            print(f"[wbm_prep] wrote DeePMD format: {dp_dir} ({n_frames} frames)")
        except ImportError:
            print("WARN: dpdata not installed; skipping DPMD conversion", file=sys.stderr)
        except Exception as e:
            print(f"WARN: DPMD conversion failed: {e}", file=sys.stderr)

    # ----- 6. Manifest -----
    manifest = {
        "source": "matbench-discovery WBM",
        "atoms_source": args.source,
        "n_total": len(kept),
        "seed": args.seed,
        "has_forces": False,
        "notes": (
            "WBM ships only initial/relaxed structures + final energies. No forces. "
            "Suitable for inference + evaluation. For training that needs forces, "
            "use MPtrj (with leakage caveat) or set force_weight=0."
        ),
        "files": {
            "test_extxyz": str(extxyz_path),
            "wbm_summary_csv": str(summary_path),
        },
    }
    if dp_dir is not None and dp_dir.exists():
        manifest["files"]["dp_test_dir"] = str(dp_dir)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[wbm_prep] manifest → {out/'manifest.json'}")

    print("\nDone. Next steps:")
    print(f"  - For inference SOPs: use {extxyz_path} as test set")
    print(f"  - For evaluation: read e_form_per_atom and e_above_hull from {summary_path}")
    print( "  - For training: see prepare_mptrj_train.py (or use --energy-only training mode)")


if __name__ == "__main__":
    main()
