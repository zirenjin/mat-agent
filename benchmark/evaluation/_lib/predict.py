"""Candidate inference -> standardized prediction records.

The candidate only ever receives geometry (species / positions / cell / pbc).
Reference labels stay in the parent evaluator. Predictions come back as plain
JSONL so the parent can build standardized scorer inputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .sandbox import run_in_sandbox


def infer_frames_target(payload: dict) -> dict:
    """Runs inside the sandbox child."""
    import numpy as np
    from ase.io import read

    from benchmark.evaluation._lib.calculator import build_calculator

    calc = build_calculator(payload["descriptor"])
    frames = read(payload["geometry_path"], index=":")
    if not isinstance(frames, list):
        frames = [frames]
    out = Path(payload["out_path"])
    with out.open("w") as fh:
        for i, atoms in enumerate(frames):
            atoms.calc = calc
            rec: dict[str, Any] = {"index": i, "natoms": len(atoms)}
            rec["energy"] = float(atoms.get_potential_energy())
            rec["forces"] = np.asarray(atoms.get_forces(), dtype=float).tolist()
            try:
                s = np.asarray(atoms.get_stress(voigt=True), dtype=float)
                rec["stress"] = s.tolist()
            except Exception:  # noqa: BLE001 -- stress is optional
                rec["stress"] = None
            fh.write(json.dumps(rec) + "\n")
    return {"n": len(frames), "out_path": str(out)}


def _write_geometry_only(atoms_list: list, path: Path) -> None:
    from ase import Atoms
    from ase.io import write

    clean = [Atoms(numbers=a.numbers, positions=a.positions, cell=a.cell, pbc=a.pbc) for a in atoms_list]
    write(path, clean, format="extxyz")


def predict_frames(descriptor: dict, atoms_list: list, out_dir: Path, *, sandbox: bool = True) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    geom = out_dir / "geometry_only.xyz"
    preds = out_dir / "predictions.jsonl"
    _write_geometry_only(atoms_list, geom)
    payload = {"descriptor": descriptor, "geometry_path": str(geom), "out_path": str(preds)}
    if sandbox:
        run_in_sandbox(infer_frames_target, payload, write_allow=[str(out_dir)])
    else:
        infer_frames_target(payload)
    return [json.loads(line) for line in preds.read_text().splitlines() if line.strip()]
