"""International space-group symbol for an ASE Atoms via spglib.

spglib ships with phonopy, so it is present in the evaluation image. If it is
somehow missing the caller marks the space_group endpoint unavailable rather
than guessing.
"""
from __future__ import annotations

from typing import Any


class SymmetryUnavailable(RuntimeError):
    pass


def spacegroup_symbol(atoms: Any, symprec: float = 1e-3) -> str:
    try:
        import spglib
    except ImportError as exc:  # pragma: no cover - image always has spglib
        raise SymmetryUnavailable("spglib not importable") from exc
    cell = (atoms.get_cell()[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    sym = spglib.get_spacegroup(cell, symprec=symprec)
    if sym is None:
        raise SymmetryUnavailable("spglib could not resolve a space group")
    # e.g. "Fm-3m (225)" -> "Fm-3m"
    return sym.split(" (")[0].replace(" ", "")
