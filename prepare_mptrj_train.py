"""
prepare_mptrj_train.py — Pull a small MPtrj slice with forces for training tests.

Why this exists
---------------
WBM has no forces. If your training SOP needs force labels (MACE & DPMD defaults),
you need a separate small chunk from MPtrj.

LEAKAGE WARNING
---------------
MPtrj is THE TRAINING SET of MACE-MPA-0, DPA-3.1, UMA, MatterSim, and CHGNet.
Whatever you "train" on here, the foundation model has already seen.

⚠️  This is FINE for "does the pipeline run" testing.
⚠️  This is USELESS for "does fine-tuning improve accuracy" — you'd see ~0 change.

For a real accuracy test you would need genuinely held-out trajectories
(e.g. recent VASP results not yet in MPtrj). Not in scope for the smoke test.

Output
------
data/mptrj_1k/
├── train.extxyz   (800 structures)
├── val.extxyz     (100 structures)
├── dp_train/      (DeePMD format, optional)
├── dp_val/        (DeePMD format, optional)
└── manifest.json

Usage
-----
    python prepare_mptrj_train.py --n 1000 --out data/mptrj_1k
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
import tempfile
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--out", type=str, default="data/mptrj_1k")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-dp", action="store_true")
    p.add_argument("--val-frac", type=float, default=0.1)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[mptrj_prep] downloading MPtrj extxyz (~700 MB on first run)...")
    try:
        from matbench_discovery.data import DataFiles
        from ase.io import iread, write
    except Exception as e:
        print(f"ERROR: missing dependency: {e}", file=sys.stderr)
        sys.exit(1)

    # MPtrj is shipped as a zipped extxyz
    mptrj_zip = DataFiles.mp_trj_extxyz.path
    print(f"[mptrj_prep] reading from {mptrj_zip}")

    with zipfile.ZipFile(mptrj_zip) as zf:
        inner = [n for n in zf.namelist() if n.endswith((".extxyz", ".xyz"))][0]
        with tempfile.NamedTemporaryFile(suffix=".extxyz", delete=False) as tmp:
            tmp.write(zf.read(inner))
            tmp_path = tmp.name
    print(f"[mptrj_prep] inner file extracted: {tmp_path}")

    # MPtrj is ~1.6M frames — we don't want to load all into memory. Reservoir sampling.
    rng = random.Random(args.seed)
    reservoir: list = []
    n_total = 0
    print(f"[mptrj_prep] reservoir sampling {args.n} structures from stream...")
    for atoms in iread(tmp_path, format="extxyz"):
        # Only keep structures that have BOTH energy and forces
        has_E = "energy" in atoms.info or any(k.endswith("energy") for k in atoms.info)
        has_F = "forces" in atoms.arrays
        if not (has_E and has_F):
            continue
        n_total += 1
        if len(reservoir) < args.n:
            reservoir.append(atoms)
        else:
            j = rng.randint(0, n_total - 1)
            if j < args.n:
                reservoir[j] = atoms
        if n_total % 100000 == 0:
            print(f"  scanned {n_total} / kept {len(reservoir)}")

    print(f"[mptrj_prep] scanned {n_total} frames, kept {len(reservoir)}")

    # Split
    rng.shuffle(reservoir)
    n_val = int(len(reservoir) * args.val_frac)
    val_atoms = reservoir[:n_val]
    train_atoms = reservoir[n_val:]
    print(f"[mptrj_prep] split: train={len(train_atoms)} val={len(val_atoms)}")

    train_path = out / "train.extxyz"
    val_path = out / "val.extxyz"
    write(str(train_path), train_atoms, format="extxyz")
    write(str(val_path), val_atoms, format="extxyz")
    print(f"[mptrj_prep] wrote {train_path}, {val_path}")

    # DPMD format
    files = {"train_extxyz": str(train_path), "val_extxyz": str(val_path)}
    if not args.no_dp:
        try:
            import dpdata
            for split, atoms_list, xyz in [("train", train_atoms, train_path),
                                            ("val", val_atoms, val_path)]:
                dp_path = out / f"dp_{split}"
                ms = dpdata.MultiSystems.from_file(str(xyz), fmt="quip/gap/xyz")
                ms.to_deepmd_npy(str(dp_path))
                files[f"dp_{split}_dir"] = str(dp_path)
                print(f"[mptrj_prep] wrote {dp_path}")
        except Exception as e:
            print(f"WARN: DPMD conversion failed: {e}", file=sys.stderr)

    manifest = {
        "source": "MPtrj (via matbench-discovery)",
        "n_train": len(train_atoms),
        "n_val": len(val_atoms),
        "seed": args.seed,
        "has_forces": True,
        "leakage_warning": (
            "MPtrj is the training set of MACE-MPA, DPA-3, UMA, MatterSim, CHGNet. "
            "Fine-tuning on this slice will show near-zero improvement. "
            "Use for pipeline test only, NOT accuracy comparison."
        ),
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[mptrj_prep] manifest → {out/'manifest.json'}")
    print("\nDone. Use this for the 'training' SOP step. Pair with wbm_1k for inference/eval.")


if __name__ == "__main__":
    main()
