"""Evaluator-side adapters: ddmms MACE-MOF0 release -> standardized inputs.

This is the ONLY place project-specific release layout (zip internals, file
names, pymatgen JSON, DOS ``.dat`` format) is parsed. The generic scorers in
``scoring/`` never see any of it -- they receive plain CSV / vectors produced
here.

Endpoint availability is fixed by what the release actually contains:

| endpoint                          | MOF-5 | UiO-66 | MOF-74 | MIL-53 | source |
|-----------------------------------|-------|--------|--------|--------|--------|
| phonon_dos_mae (vs DFT)           |  yes  |  yes   |  yes   |  yes   | DFT_DOS.zip |
| phonon_frequency_rmse (vs DFT)    |  no   |  no    |  no    |  no    | no DFT band structure released |
| imaginary_mode_count              |  yes  |  yes   |  yes   |  yes   | candidate phonopy run |
| unit_cell_length_percent_error    |  yes  |  yes   |  yes   |  yes   | optimized structures.zip/DFT |
| space_group_match                 |  yes  |  yes   |  yes   |  yes   | optimized structures.zip/DFT |
| bulk_modulus_relative_error       |  yes  |  yes   |  yes   |  yes   | BulkModulus_data.zip |
| thermal_expansion_relative_error  |  yes  |  yes   |  no    |  no    | NegativeThermalExpansion.csv |
"""
from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "data/mace_mof_0/raw"

MOFS = ("MOF-5", "UiO-66", "MOF-74", "MIL-53")

AVAILABILITY = {
    m: {
        "phonon_dos_mae": True,
        "phonon_frequency_rmse": False,   # no DFT phonon band structure in the release
        "imaginary_mode_count": True,
        "unit_cell_length_percent_error": True,
        "space_group_match": True,
        "bulk_modulus_relative_error": True,
        "thermal_expansion_relative_error": m in ("MOF-5", "UiO-66"),
    }
    for m in MOFS
}

_DOS_NAME = {"MOF-5": "MOF-5", "UiO-66": "UiO-66", "MOF-74": "MOF-74", "MIL-53": "MIL-53"}


class ReleaseRefs:
    def __init__(self) -> None:
        self._dos_zip = RAW / "phonons/DFT_DOS.zip"
        self._opt_zip = RAW / "optimized structures.zip"
        self._bulk_zip = RAW / "BulkModulus_data.zip"
        self._nte_csv = RAW / "NegativeThermalExpansion.csv"

    # --- initial structure -------------------------------------------------
    def initial_structure(self, mof: str):
        from ase.io import read

        z = RAW / f"phonons/{mof}.zip"
        with zipfile.ZipFile(z) as zf:
            name = next(n for n in zf.namelist() if n.endswith(f"{mof}/POSCAR"))
            text = zf.read(name).decode()
        return read(io.StringIO(text), format="vasp")

    # --- DFT relaxed structure ------------------------------------------
    def dft_relaxed(self, mof: str):
        import json as _json

        from ase import Atoms

        with zipfile.ZipFile(self._opt_zip) as zf:
            name = next(n for n in zf.namelist()
                        if n.endswith(f"DFT/{mof}_geom-opt.json") and "__MACOSX" not in n)
            doc = _json.loads(zf.read(name).decode())
        struct = (doc[-1] if isinstance(doc, list) else doc)
        struct = struct.get("structure", struct)
        lat = struct["lattice"]["matrix"]
        species = [s["species"][0]["element"] if isinstance(s["species"], list) else s["label"]
                   for s in struct["sites"]]
        frac = [s["abc"] for s in struct["sites"]]
        atoms = Atoms(symbols=species, scaled_positions=frac, cell=lat, pbc=True)
        return atoms

    def dft_cell_lengths(self, mof: str) -> tuple[float, float, float]:
        atoms = self.dft_relaxed(mof)
        a, b, c = atoms.cell.lengths()
        return float(a), float(b), float(c)

    def dft_space_group(self, mof: str, symprec: float = 1e-3) -> str:
        from .symmetry import spacegroup_symbol

        return spacegroup_symbol(self.dft_relaxed(mof), symprec=symprec)

    # --- DFT phonon DOS --------------------------------------------------
    def dft_dos(self, mof: str) -> tuple[list[float], list[float]]:
        with zipfile.ZipFile(self._dos_zip) as zf:
            name = next(n for n in zf.namelist()
                        if n.endswith(f"DFT/{_DOS_NAME[mof]}-total_dos.dat") and "__MACOSX" not in n)
            lines = zf.read(name).decode().splitlines()
        freq, dos = [], []
        for ln in lines:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            freq.append(float(parts[0]))
            dos.append(float(parts[1]))
        return freq, dos

    # --- DFT bulk modulus (GPa) ---------------------------------------
    def dft_bulk_modulus(self, mof: str) -> float | None:
        want = {"MOF-5": "MOF-5", "UiO-66": "UiO-66", "MOF-74": "MOF-74", "MIL-53": "MIL-53"}[mof]
        with zipfile.ZipFile(self._bulk_zip) as zf:
            for fname in ("BulkModulus_data/DFT.csv", "BulkModulus_data/Exp_and_DFT.csv"):
                try:
                    rows = list(csv.DictReader(io.StringIO(zf.read(fname).decode())))
                except KeyError:
                    continue
                for row in rows:
                    if (row.get("mof") or "").strip() == want:
                        for key in ("dft (GPa)", "DFT (GPa)", "dft", "DFT"):
                            if row.get(key):
                                return float(row[key])
        return None

    # --- DFT thermal expansion (1e-6 / K) ---------------------------
    def dft_cte(self, mof: str) -> tuple[list[float], list[float]] | None:
        if mof not in ("MOF-5", "UiO-66"):
            return None
        rows = list(csv.reader(self._nte_csv.read_text().splitlines()))
        header = rows[0]
        col = next((i for i, h in enumerate(header) if h.startswith(mof)), None)
        if col is None:
            return None
        temps, ctes = [], []
        for row in rows[1:]:
            if len(row) <= col or not row[0].strip():
                continue
            temps.append(float(row[0]))
            ctes.append(float(row[col]))
        return temps, ctes
