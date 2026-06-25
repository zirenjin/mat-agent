#!/usr/bin/env python3
"""Convert scout-model predictions into worker-model training datasets.

The bridge maps canonical pred_* pseudo-labels into ref_* training labels and
serializes the result for DeePMD, MACE, or FairChem. It is designed for
cross-architecture active-learning loops where one model scouts OOD structures
and another model consumes the pseudo-labeled configurations.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.io import read, write

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
from ase_conventions import (  # noqa: E402
    PRED_ENERGY_KEY,
    PRED_FORCES_KEY,
    REF_ENERGY_KEY,
    REF_FORCES_KEY,
    get_pred_energy,
    get_pred_forces,
    set_ref,
)

SUPPORTED_TARGETS = {"deepmd", "mace", "fairchem"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-xyz", required=True, help="Predicted extxyz with pred_energy/pred_forces")
    p.add_argument("--target-format", required=True, choices=sorted(SUPPORTED_TARGETS))
    p.add_argument("--output-dir", required=True, help="Training-ready dataset directory")
    p.add_argument("--output-name", default="train.extxyz", help="Output extxyz name for mace/fairchem")
    p.add_argument("--type-map", nargs="*", default=None, help="Optional global element ordering for DeepMD type_map.raw")
    p.add_argument("--append", action="store_true", help="Append extxyz frames for mace/fairchem when output exists")
    p.add_argument("--strict", action="store_true", default=True, help="Fail on missing pred labels")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src_xyz)
    out = Path(args.output_dir)
    if not src.is_file():
        raise SystemExit(f"source extxyz not found: {src}")
    frames = _read_frames(src)
    converted = [_pseudo_to_ref(atoms, index=i, strict=args.strict) for i, atoms in enumerate(frames)]
    out.mkdir(parents=True, exist_ok=True)

    if args.target_format == "deepmd":
        summary = _write_deepmd(converted, out, args.type_map)
    else:
        summary = _write_extxyz(converted, out, args.output_name, append=args.append, target=args.target_format)

    manifest = {
        "source": str(src),
        "target_format": args.target_format,
        "n_frames": len(converted),
        "energy_key": REF_ENERGY_KEY,
        "forces_key": REF_FORCES_KEY,
        "summary": summary,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def _read_frames(path: Path) -> list[Atoms]:
    frames = read(str(path), ":")
    if isinstance(frames, Atoms):
        return [frames]
    return list(frames)


def _pseudo_to_ref(atoms: Atoms, index: int, strict: bool) -> Atoms:
    atoms = atoms.copy()
    energy = get_pred_energy(atoms)
    forces = get_pred_forces(atoms)
    if energy is None or forces is None:
        msg = f"frame {index} missing {PRED_ENERGY_KEY}/{PRED_FORCES_KEY} pseudo-labels"
        if strict:
            raise ValueError(msg)
        energy = 0.0 if energy is None else energy
        forces = np.zeros((len(atoms), 3), dtype=np.float64) if forces is None else forces
    forces = np.asarray(forces, dtype=np.float64)
    if forces.shape != (len(atoms), 3):
        raise ValueError(f"frame {index} has invalid forces shape {forces.shape}; expected {(len(atoms), 3)}")
    set_ref(atoms, float(energy), forces)
    atoms.info.pop(PRED_ENERGY_KEY, None)
    if PRED_FORCES_KEY in atoms.arrays:
        del atoms.arrays[PRED_FORCES_KEY]
    atoms.calc = None
    return _strip_training_noise(atoms)


def _strip_training_noise(atoms: Atoms) -> Atoms:
    keep_info = {REF_ENERGY_KEY, "config_type", "head", "source_id", "sample_weight"}
    keep_arrays = {REF_FORCES_KEY, "numbers", "positions"}
    for key in list(atoms.info.keys()):
        if key not in keep_info and not key.startswith("ref_"):
            atoms.info.pop(key, None)
    for key in list(atoms.arrays.keys()):
        if key not in keep_arrays and not key.startswith("ref_"):
            del atoms.arrays[key]
    return atoms


def _group_key(atoms: Atoms) -> tuple[tuple[str, ...], int]:
    return tuple(atoms.get_chemical_symbols()), len(atoms)


def _write_deepmd(frames: list[Atoms], out: Path, type_map: list[str] | None) -> dict[str, object]:
    groups: dict[tuple[tuple[str, ...], int], list[Atoms]] = defaultdict(list)
    for atoms in frames:
        groups[_group_key(atoms)].append(atoms)

    global_type_map = list(type_map or sorted({sym for atoms in frames for sym in atoms.get_chemical_symbols()}))
    if not global_type_map:
        raise ValueError("empty type map")
    type_index = {sym: i for i, sym in enumerate(global_type_map)}

    systems = []
    for sys_idx, (_, atoms_list) in enumerate(groups.items()):
        sys_dir = out / f"sys_{sys_idx:04d}"
        set_dir = sys_dir / "set.000"
        set_dir.mkdir(parents=True, exist_ok=True)
        first = atoms_list[0]
        symbols = first.get_chemical_symbols()
        type_raw = np.asarray([type_index[sym] for sym in symbols], dtype=np.int32)
        coords = []
        boxes = []
        energies = []
        forces = []
        for atoms in atoms_list:
            if atoms.get_chemical_symbols() != symbols:
                raise ValueError("internal grouping error: symbol order mismatch")
            coords.append(np.asarray(atoms.get_positions(), dtype=np.float64).reshape(-1))
            cell = np.asarray(atoms.get_cell().array, dtype=np.float64)
            if atoms.cell.rank != 3 or abs(np.linalg.det(cell)) < 1e-12:
                cell = np.zeros((3, 3), dtype=np.float64)
            boxes.append(cell.reshape(-1))
            energies.append(float(atoms.info[REF_ENERGY_KEY]))
            forces.append(np.asarray(atoms.arrays[REF_FORCES_KEY], dtype=np.float64).reshape(-1))
        np.save(set_dir / "coord.npy", np.asarray(coords, dtype=np.float64))
        np.save(set_dir / "box.npy", np.asarray(boxes, dtype=np.float64))
        np.save(set_dir / "energy.npy", np.asarray(energies, dtype=np.float64))
        np.save(set_dir / "force.npy", np.asarray(forces, dtype=np.float64))
        (sys_dir / "type.raw").write_text("\n".join(str(int(x)) for x in type_raw) + "\n")
        (sys_dir / "type_map.raw").write_text("\n".join(global_type_map) + "\n")
        systems.append({
            "path": str(sys_dir),
            "n_frames": len(atoms_list),
            "n_atoms": len(first),
            "symbols": symbols,
        })
    return {"type_map": global_type_map, "systems": systems}


def _write_extxyz(frames: list[Atoms], out: Path, output_name: str, append: bool, target: str) -> dict[str, object]:
    path = out / output_name
    mode = "a" if append and path.exists() else "w"
    write(str(path), frames, format="extxyz", append=(mode == "a"))
    return {"path": str(path), "n_frames": len(frames), "target": target}


if __name__ == "__main__":
    main()
