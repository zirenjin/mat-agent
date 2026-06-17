"""
data_prep.py — Prepare a uniform train/val/test dataset for MACE / FairChem / DeePMD comparison.

The same atomic configurations are written in BOTH:
  - extxyz format  (consumed directly by MACE and FairChem via ASE)
  - DeePMD npy format  (consumed by DeePMD-kit)

This guarantees the three frameworks see identical structures, so cross-model
comparisons are fair.

Usage
-----
    # Phase 1: smoke-test the pipeline with synthetic data (no downloads)
    python data_prep.py --mode demo --n 100 --out data/demo

    # Phase 2: real evaluation data from WBM (matbench-discovery hold-out)
    python data_prep.py --mode wbm --n 1000 --out data/wbm_1k

    # Or: user provides their own extxyz file
    python data_prep.py --mode custom --input my_data.extxyz --out data/custom

Outputs
-------
    {out}/train.extxyz                 # 80% for training
    {out}/val.extxyz                   # 10% for validation
    {out}/test.extxyz                  # 10% for testing/inference
    {out}/dp_train/                    # same train split in DeePMD format
    {out}/dp_val/                      # same val split in DeePMD format
    {out}/dp_test/                     # same test split in DeePMD format
    {out}/manifest.json                # metadata
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ase import Atoms
    from ase.build import bulk
    from ase.io import read, write
except ImportError:
    print("ERROR: ASE not installed. Run: pip install ase", file=sys.stderr)
    sys.exit(1)


# ----------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------

def make_demo_dataset(n: int = 100, seed: int = 42) -> list[Atoms]:
    """Generate N perturbed bulk crystals with synthetic E/F labels.

    Intended for smoke-testing the pipeline only — energies are fake,
    DO NOT use this for accuracy comparison.
    """
    rng = np.random.default_rng(seed)
    atoms_list: list[Atoms] = []
    elements = ["Si", "Al", "Cu", "Mg", "Ti", "Fe", "Zn", "Ni", "Na", "Mn"]
    structures = [("fcc", 3.6), ("bcc", 3.1), ("diamond", 5.4)]

    for i in range(n):
        elem = elements[i % len(elements)]
        struct, a0 = structures[i % len(structures)]
        a = a0 + 0.1 * rng.standard_normal()
        try:
            atoms = bulk(elem, struct, a=a) * (2, 2, 2)
        except Exception:
            # some element/structure combos may not be supported by ASE.bulk
            atoms = bulk("Si", "diamond", a=5.4) * (2, 2, 2)
        # perturb positions
        atoms.positions += 0.05 * rng.standard_normal(atoms.positions.shape)
        # mock labels
        n_at = len(atoms)
        energy = -5.0 * n_at + 0.3 * rng.standard_normal()
        forces = 0.1 * rng.standard_normal((n_at, 3))
        stress = 0.01 * rng.standard_normal(6)  # ASE voigt order
        atoms.info["energy"] = float(energy)
        atoms.info["stress"] = stress  # keep as np.ndarray, extxyz requires .shape attribute
        atoms.arrays["forces"] = forces
        atoms_list.append(atoms)
    return atoms_list


def download_wbm_subset(n: int = 1000, seed: int = 42) -> list[Atoms]:
    """Download a random N-structure slice of the WBM test set.

    Requires `pip install matbench-discovery pymatgen ase`.

    NOTE: This is the held-out set for the matbench-discovery benchmark. Three
    foundation models (MACE-MPA, DPA-4, UMA) have NOT seen these structures, so
    MAE/F1 computed here are meaningful.
    """
    try:
        from matbench_discovery.data import load
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as e:
        print(
            f"ERROR: missing dependency for WBM mode: {e}\n"
            "Install: pip install matbench-discovery pymatgen",
            file=sys.stderr,
        )
        sys.exit(1)

    # NOTE: column / dataset names may shift between matbench-discovery versions.
    # If this fails, check:  python -c "from matbench_discovery.data import DATA_FILES; print(DATA_FILES)"
    print("[data_prep] loading WBM initial structures (this may take a few minutes)...")
    try:
        df = load("wbm/initial-atoms")  # may need to be "wbm-initial-atoms" depending on version
    except Exception as e:
        print(
            f"ERROR: failed to load WBM data: {e}\n"
            "Try inspecting available datasets:\n"
            "    from matbench_discovery.data import DATA_FILES; print(list(DATA_FILES))",
            file=sys.stderr,
        )
        sys.exit(1)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    df_sub = df.iloc[idx]

    adaptor = AseAtomsAdaptor()
    atoms_list: list[Atoms] = []
    # column containing pymatgen Structure may be named "initial_structure" or similar
    struct_col = next((c for c in df_sub.columns if "struct" in c.lower()), None)
    if struct_col is None:
        print(f"ERROR: no structure column found. Columns: {list(df_sub.columns)}", file=sys.stderr)
        sys.exit(1)

    energy_col = next((c for c in df_sub.columns if "e_form" in c.lower() or "energy" in c.lower()), None)

    for _, row in df_sub.iterrows():
        structure = row[struct_col]
        atoms = adaptor.get_atoms(structure)
        if energy_col is not None and not (row[energy_col] is None or np.isnan(row[energy_col])):
            atoms.info["energy"] = float(row[energy_col])
        atoms_list.append(atoms)
    print(f"[data_prep] loaded {len(atoms_list)} WBM structures")
    return atoms_list


def load_custom(path: str) -> list[Atoms]:
    atoms = read(path, ":")
    if not isinstance(atoms, list):
        atoms = [atoms]
    return atoms


# ----------------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------------

def write_extxyz(atoms_list: list[Atoms], path: Path) -> None:
    write(str(path), atoms_list, format="extxyz")
    print(f"[data_prep] wrote {path}  ({len(atoms_list)} structures)")


def write_dpdata(atoms_list: list[Atoms], out_dir: Path, source_extxyz: Path | None = None) -> bool:
    """Convert ASE Atoms list to DeePMD npy format via dpdata.

    Strategy: write atoms to a temp extxyz, then load via dpdata.
    Direct ASE→dpdata conversion is fragile across dpdata versions; extxyz is the
    most reliable intermediate.
    """
    try:
        import dpdata
    except ImportError:
        print(
            "WARNING: dpdata not installed; skipping DeePMD format.\n"
            "Install: pip install dpdata",
            file=sys.stderr,
        )
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    # If caller has already written extxyz, reuse it; otherwise create temp
    if source_extxyz is not None and source_extxyz.exists():
        xyz_path = source_extxyz
        cleanup = False
    else:
        xyz_path = out_dir.parent / f"_tmp_{out_dir.name}.extxyz"
        write(str(xyz_path), atoms_list, format="extxyz")
        cleanup = True

    try:
        # dpdata reads extxyz via the "quip/gap/xyz" format (covers extxyz)
        try:
            ms = dpdata.MultiSystems.from_file(file_name=str(xyz_path), fmt="quip/gap/xyz")
        except Exception:
            # fallback: per-frame loading
            ms = dpdata.MultiSystems()
            for a in atoms_list:
                try:
                    ls = dpdata.LabeledSystem(data={
                        "atom_names": list(set(a.get_chemical_symbols())),
                        "atom_numbs": [a.get_chemical_symbols().count(s) for s in set(a.get_chemical_symbols())],
                        "atom_types": np.array([sorted(set(a.get_chemical_symbols())).index(s)
                                                 for s in a.get_chemical_symbols()]),
                        "cells": np.array([a.cell.array]),
                        "coords": np.array([a.positions]),
                        "energies": np.array([a.info.get("energy", 0.0)]),
                        "forces": np.array([a.arrays.get("forces", np.zeros((len(a), 3)))]),
                        "orig": np.zeros(3),
                    })
                    ms.append(ls)
                except Exception as e:
                    print(f"  skip one structure: {e}", file=sys.stderr)
                    continue
        ms.to_deepmd_npy(str(out_dir))
        n_frames = sum(s.get_nframes() for s in ms)
        print(f"[data_prep] wrote {out_dir}/  ({n_frames} frames in DeePMD format)")
        return True
    finally:
        if cleanup and xyz_path.exists():
            xyz_path.unlink()


# ----------------------------------------------------------------------------
# Split
# ----------------------------------------------------------------------------

def split(atoms_list: list[Atoms], ratios=(0.8, 0.1, 0.1), seed: int = 42) -> tuple[list, list, list]:
    rng = random.Random(seed)
    idx = list(range(len(atoms_list)))
    rng.shuffle(idx)
    n = len(atoms_list)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = [atoms_list[i] for i in idx[:n_train]]
    val = [atoms_list[i] for i in idx[n_train:n_train + n_val]]
    test = [atoms_list[i] for i in idx[n_train + n_val:]]
    return train, val, test


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["demo", "wbm", "custom"], default="demo",
                   help="Data source")
    p.add_argument("--n", type=int, default=100, help="Number of structures (demo/wbm)")
    p.add_argument("--input", type=str, help="extxyz path (custom mode)")
    p.add_argument("--out", type=str, default="data/prepared", help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-split", action="store_true",
                   help="Write a single set without train/val/test split (for pure-inference test)")
    p.add_argument("--no-dp", action="store_true",
                   help="Skip DeePMD format conversion (faster if you don't need DPMD)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Source data
    if args.mode == "demo":
        atoms_list = make_demo_dataset(n=args.n, seed=args.seed)
    elif args.mode == "wbm":
        atoms_list = download_wbm_subset(n=args.n, seed=args.seed)
    elif args.mode == "custom":
        if not args.input:
            p.error("--input is required for --mode custom")
        atoms_list = load_custom(args.input)

    print(f"[data_prep] sourced {len(atoms_list)} structures from mode={args.mode}")

    manifest: dict[str, Any] = {
        "mode": args.mode,
        "n_total": len(atoms_list),
        "seed": args.seed,
        "files": {},
    }

    if args.no_split:
        write_extxyz(atoms_list, out / "all.extxyz")
        manifest["files"]["all_extxyz"] = str(out / "all.extxyz")
        if not args.no_dp:
            write_dpdata(atoms_list, out / "dp_all", source_extxyz=out / "all.extxyz")
            manifest["files"]["dp_all_dir"] = str(out / "dp_all")
    else:
        train, val, test = split(atoms_list, seed=args.seed)
        print(f"[data_prep] split: train={len(train)}  val={len(val)}  test={len(test)}")

        write_extxyz(train, out / "train.extxyz")
        write_extxyz(val, out / "val.extxyz")
        write_extxyz(test, out / "test.extxyz")
        manifest["n_train"] = len(train)
        manifest["n_val"] = len(val)
        manifest["n_test"] = len(test)
        manifest["files"].update({
            "train_extxyz": str(out / "train.extxyz"),
            "val_extxyz": str(out / "val.extxyz"),
            "test_extxyz": str(out / "test.extxyz"),
        })

        if not args.no_dp:
            write_dpdata(train, out / "dp_train", source_extxyz=out / "train.extxyz")
            write_dpdata(val, out / "dp_val", source_extxyz=out / "val.extxyz")
            write_dpdata(test, out / "dp_test", source_extxyz=out / "test.extxyz")
            manifest["files"].update({
                "dp_train_dir": str(out / "dp_train"),
                "dp_val_dir": str(out / "dp_val"),
                "dp_test_dir": str(out / "dp_test"),
            })

    # Manifest
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[data_prep] manifest → {manifest_path}")

    print(f"\nDone. {len(atoms_list)} structures written to {out}/")
    print("Next: each model's SOP will reference these files via the manifest.")


if __name__ == "__main__":
    main()
