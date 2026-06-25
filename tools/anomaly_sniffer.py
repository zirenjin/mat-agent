#!/usr/bin/env python3
"""Physics-informed failure diagnostics for MLIP runs.

The sniffer turns raw training/runtime logs plus the latest trajectory cache into
structured JSON that an upstream agent can use for retry, data cleaning, or
active-learning decisions.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read

LOG_PATTERNS: list[tuple[str, str, str]] = [
    ("OOM", r"cuda[^\n]*(out of memory|oom)|outofmemory|cublas.*alloc|hip[^\n]*out of memory", "REDUCE_BATCH_SIZE"),
    ("NUMERICAL_NAN", r"\bnan\b|\binf\b|non[- ]?finite|floating point exception", "LOWER_LR_AND_CLIP_GRADIENTS"),
    ("GRADIENT_EXPLOSION", r"grad(ient)?[^\n]*(explod|overflow)|loss[^\n]*(explod|diverg)|diverged|too large", "LOWER_LR_AND_CLIP_GRADIENTS"),
]

VOLUME_KEYS = (
    "reference_volume_A3",
    "ref_volume_A3",
    "bulk_volume_A3",
    "equilibrium_volume_A3",
    "standard_bulk_volume_A3",
)
VOLUME_PER_ATOM_KEYS = (
    "reference_volume_per_atom_A3",
    "ref_volume_per_atom_A3",
    "bulk_volume_per_atom_A3",
    "equilibrium_volume_per_atom_A3",
    "standard_bulk_volume_per_atom_A3",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log_file_path", nargs="?", help="Log file to inspect")
    p.add_argument("latest_trajectory_path", nargs="?", help="Latest .extxyz/.xyz trajectory cache")
    p.add_argument("--log-file", dest="log_file_flag", help="Log file to inspect")
    p.add_argument("--trajectory", dest="trajectory_flag", help="Latest .extxyz/.xyz trajectory cache")
    p.add_argument("--last-n", type=int, default=5, help="Number of trailing frames to inspect")
    p.add_argument("--overlap-cutoff", type=float, default=0.8, help="Atomic overlap cutoff in Angstrom")
    p.add_argument("--reference-volume", type=float, default=None, help="Reference bulk volume in A^3")
    p.add_argument("--reference-volume-per-atom", type=float, default=None, help="Reference bulk volume per atom in A^3")
    p.add_argument("--json-indent", type=int, default=2)
    args = p.parse_args()
    args.log_file_path = args.log_file_flag or args.log_file_path
    args.latest_trajectory_path = args.trajectory_flag or args.latest_trajectory_path
    if not args.log_file_path:
        p.error("log_file_path or --log-file is required")
    return args


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_file_path)
    log_text = _read_text(log_path)
    signals = _parse_log_signals(log_text)

    frames: list[Atoms] = []
    traj_error = None
    if args.latest_trajectory_path:
        try:
            frames = _read_last_frames(Path(args.latest_trajectory_path), args.last_n)
        except Exception as exc:  # Keep diagnostics usable even if trajectory is corrupt.
            traj_error = f"{type(exc).__name__}: {exc}"

    physical = _physical_diagnostics(
        frames,
        overlap_cutoff=args.overlap_cutoff,
        reference_volume=args.reference_volume,
        reference_volume_per_atom=args.reference_volume_per_atom,
    )
    result = _classify(signals, physical, traj_error)
    print(json.dumps(result, indent=args.json_indent, sort_keys=False))


def _read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except FileNotFoundError:
        return ""


def _parse_log_signals(text: str) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for error_class, pattern, hint in LOG_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            line = _line_around(text, match.start())
            hits.append({"error_class": error_class, "remediation_hint": hint, "evidence": line[:500]})
    return hits


def _line_around(text: str, index: int) -> str:
    start = text.rfind("\n", 0, index) + 1
    end = text.find("\n", index)
    if end < 0:
        end = len(text)
    return text[start:end].strip()


def _read_last_frames(path: Path, n: int) -> list[Atoms]:
    if not path.exists():
        return []
    frames = read(str(path), f"-{max(n, 1)}:")
    if isinstance(frames, Atoms):
        return [frames]
    return list(frames)


def _physical_diagnostics(
    frames: list[Atoms],
    overlap_cutoff: float,
    reference_volume: float | None,
    reference_volume_per_atom: float | None,
) -> dict[str, Any]:
    if not frames:
        return {
            "frames_inspected": 0,
            "min_atomic_distance_A": None,
            "atomic_overlap": False,
            "volume_distortion_ratio": None,
            "volume_reference_source": None,
            "failure_mode": None,
        }

    min_dist = math.inf
    ratios: list[float] = []
    ref_source = None
    for idx, atoms in enumerate(frames):
        frame_min = _min_pair_distance(atoms)
        if frame_min is not None:
            min_dist = min(min_dist, frame_min)
        ratio, source = _volume_ratio(atoms, frames, idx, reference_volume, reference_volume_per_atom)
        if ratio is not None and np.isfinite(ratio):
            ratios.append(float(ratio))
            ref_source = ref_source or source

    min_value = None if min_dist is math.inf else float(min_dist)
    atomic_overlap = min_value is not None and min_value < overlap_cutoff
    volume_ratio = min(ratios) if ratios else None
    compressed = volume_ratio is not None and volume_ratio < 0.80
    expanded = volume_ratio is not None and volume_ratio > 1.35

    if atomic_overlap:
        failure_mode = "ATOMIC_COLLISION_IN_PHASE_SPACE"
    elif compressed:
        failure_mode = "HIGH_PRESSURE_VOLUME_COLLAPSE"
    elif expanded:
        failure_mode = "LATTICE_EXPANSION_NON_EQUILIBRIUM"
    else:
        failure_mode = None

    return {
        "frames_inspected": len(frames),
        "min_atomic_distance_A": min_value,
        "atomic_overlap": atomic_overlap,
        "volume_distortion_ratio": volume_ratio,
        "volume_reference_source": ref_source,
        "failure_mode": failure_mode,
    }


def _min_pair_distance(atoms: Atoms) -> float | None:
    n = len(atoms)
    if n < 2:
        return None
    try:
        distances = atoms.get_all_distances(mic=bool(np.any(atoms.pbc)))
    except Exception:
        distances = atoms.get_all_distances(mic=False)
    distances = np.asarray(distances, dtype=float)
    distances[distances <= 0.0] = np.inf
    value = float(np.min(distances))
    return value if np.isfinite(value) else None


def _volume_ratio(
    atoms: Atoms,
    frames: list[Atoms],
    frame_index: int,
    reference_volume: float | None,
    reference_volume_per_atom: float | None,
) -> tuple[float | None, str | None]:
    current = float(abs(atoms.get_volume())) if atoms.cell.rank == 3 else 0.0
    if current <= 0.0:
        return None, None

    ref = reference_volume
    source = "cli_reference_volume" if ref else None
    if ref is None and reference_volume_per_atom is not None:
        ref = reference_volume_per_atom * max(len(atoms), 1)
        source = "cli_reference_volume_per_atom"
    if ref is None:
        for key in VOLUME_KEYS:
            if key in atoms.info:
                ref = float(atoms.info[key])
                source = f"atoms.info.{key}"
                break
    if ref is None:
        for key in VOLUME_PER_ATOM_KEYS:
            if key in atoms.info:
                ref = float(atoms.info[key]) * max(len(atoms), 1)
                source = f"atoms.info.{key}"
                break
    if ref is None and frames:
        first = frames[0]
        first_vol = float(abs(first.get_volume())) if first.cell.rank == 3 else 0.0
        if first_vol > 0.0:
            ref = first_vol * max(len(atoms), 1) / max(len(first), 1)
            source = "first_frame_volume_per_atom"
    if ref is None or ref <= 0.0:
        return None, None
    return current / ref, source


def _classify(signals: list[dict[str, str]], physical: dict[str, Any], traj_error: str | None) -> dict[str, Any]:
    failure_mode = physical.get("failure_mode")
    has_log_failure = bool(signals)
    has_physical_failure = bool(failure_mode)

    if has_physical_failure:
        error_class = "PHYSICAL_INVARIANT_VIOLATION"
        hint = "GEOMETRY_RELAXATION_REQUIRED" if physical.get("atomic_overlap") else "CHECK_DISTORTED_CELL_OR_DATA_FILTER"
        status = "DIVERGED"
    elif any(s["error_class"] == "OOM" for s in signals):
        error_class = "OOM"
        hint = "REDUCE_BATCH_SIZE"
        status = "DIVERGED"
    elif has_log_failure:
        error_class = signals[0]["error_class"]
        hint = signals[0]["remediation_hint"]
        status = "DIVERGED"
    else:
        error_class = None
        hint = "NO_ACTION_REQUIRED"
        status = "OK"

    diagnostics = dict(physical)
    diagnostics["log_signals"] = signals
    if traj_error:
        diagnostics["trajectory_error"] = traj_error

    return {
        "status": status,
        "error_class": error_class,
        "diagnostics": diagnostics,
        "remediation_hint": hint,
    }


if __name__ == "__main__":
    main()
