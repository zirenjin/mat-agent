"""Stage B -- MOF lattice dynamics.

Per MOF:  candidate calculator
            -> tight cell relaxation                (sandboxed)
            -> phonopy finite-displacement phonons  (sandboxed)
            -> EOS energy-volume scan               (sandboxed)
          -> evaluator builds standardized inputs vs the ddmms DFT references
          -> frozen scorers in scoring/physics/mace_mof_0/ produce every number.

Frozen protocol constants live in ``PROTOCOL`` (imported from evaluate_mof0).
Endpoints with no fair DFT reference in the release are returned as ``None`` with
a ``reason`` -- never a proxy.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .metrics import run_scorer
from .release_adapters import AVAILABILITY, ReleaseRefs
from .sandbox import run_in_sandbox

EV_A3_TO_GPA = 160.21766208


# --------------------------------------------------------------------------
# sandbox-child target: everything that needs the candidate model
# --------------------------------------------------------------------------
def relax_and_phonons_target(payload: dict) -> dict:
    import numpy as np
    from ase.io import read

    from benchmark.evaluation._lib.calculator import build_calculator

    P = payload["protocol"]
    calc = build_calculator(payload["descriptor"])
    atoms = read(payload["structure_path"], format="extxyz")
    atoms.calc = calc

    # --- tight cell relaxation ---
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    relax_log = payload["work"] + "/relax.log"
    flt = FrechetCellFilter(atoms)
    opt = FIRE(flt, logfile=relax_log)
    opt.run(fmax=P["relax_fmax_eV_per_A"], steps=P["relax_max_steps"])
    fmax_final = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    relaxed = {
        "cell": atoms.get_cell()[:].tolist(),
        "scaled_positions": atoms.get_scaled_positions().tolist(),
        "numbers": atoms.get_atomic_numbers().tolist(),
        "cell_lengths": [float(x) for x in atoms.cell.lengths()],
        "volume": float(atoms.get_volume()),
        "converged": bool(fmax_final <= P["relax_fmax_eV_per_A"] * 5),
        "fmax_final_eV_per_A": fmax_final,
        "n_steps": int(opt.get_number_of_steps()),
    }

    # --- finite-displacement phonons ---
    from phonopy import Phonopy

    import tools.mace_phonopy_workflow as mpw

    sc = payload["supercell"]
    ph = Phonopy(mpw.ase_atoms_to_phonopy(atoms), np.diag(sc))
    ph.generate_displacements(distance=P["phonon_displacement_A"])
    force_sets = []
    for scell in ph.supercells_with_displacements:
        sat = mpw.phonopy_atoms_to_ase(scell)
        sat.calc = calc
        force_sets.append(sat.get_forces())
    ph.forces = force_sets
    ph.produce_force_constants()
    ph.run_mesh(P["phonon_mesh"], with_eigenvectors=False, is_mesh_symmetry=False)
    freqs = np.asarray(ph.get_mesh_dict()["frequencies"], dtype=float).reshape(-1)
    fmin, fmax = float(freqs.min()), float(freqs.max())
    ph.run_total_dos(freq_min=fmin, freq_max=fmax,
                     freq_pitch=(fmax - fmin) / P["dos_points"], sigma=P["dos_sigma_THz"])
    tdos = ph.get_total_dos_dict()

    # --- EOS energy-volume scan (isotropic strain, ions relaxed at fixed cell) ---
    eos = []
    base_cell = atoms.get_cell()[:]
    base_spos = atoms.get_scaled_positions()
    base_num = atoms.get_atomic_numbers()
    for strain in P["eos_strain_grid"]:
        from ase import Atoms

        s = (1.0 + strain) ** (1.0 / 3.0)
        cell_s = base_cell * s
        a2 = Atoms(numbers=base_num, scaled_positions=base_spos, cell=cell_s, pbc=True)
        a2.calc = calc
        FIRE(a2, logfile=payload["work"] + "/eos.log").run(
            fmax=P["eos_fmax_eV_per_A"], steps=P["eos_max_steps"])
        eos.append([float(a2.get_volume()), float(a2.get_potential_energy())])

    return {
        "relaxed": relaxed,
        "phonon": {
            "supercell": list(sc),
            "mesh": list(P["phonon_mesh"]),
            "frequencies_THz": freqs.tolist(),
            "min_frequency_THz": fmin,
            "dos_frequency_THz": list(map(float, tdos["frequency_points"])),
            "dos": list(map(float, tdos["total_dos"])),
        },
        "eos": eos,
    }


# --------------------------------------------------------------------------
# parent-side scoring
# --------------------------------------------------------------------------
def _supercell_for(atoms, min_len: float, cap: int) -> list[int]:
    return [min(cap, max(1, math.ceil(min_len / L))) for L in atoms.cell.lengths()]


def _norm_dos(freq, dos, lo, hi):
    import numpy as np

    f = np.asarray(freq, dtype=float)
    d = np.clip(np.asarray(dos, dtype=float), 0.0, None)
    m = (f >= lo) & (f <= hi)
    f, d = f[m], d[m]
    area = np.trapz(d, f)
    if area > 0:
        d = d / area
    return f, d


def _score_dos(cand: dict, mof: str, refs: ReleaseRefs, work: Path) -> float:
    import numpy as np

    df, dd = refs.dft_dos(mof)
    lo = max(min(df), min(cand["dos_frequency_THz"]), 0.0)
    hi = min(max(df), max(cand["dos_frequency_THz"]))
    gf, gd = _norm_dos(df, dd, lo, hi)
    cf, cd = _norm_dos(cand["dos_frequency_THz"], cand["dos"], lo, hi)
    cd_on_grid = np.interp(gf, cf, cd)
    ref_csv = work / f"{mof}_dos_ref.csv"
    pred_csv = work / f"{mof}_dos_pred.csv"
    ref_csv.write_text("frequency,dos\n" + "\n".join(f"{x},{y}" for x, y in zip(gf, gd, strict=True)) + "\n")
    pred_csv.write_text("frequency,dos\n" + "\n".join(f"{x},{y}" for x, y in zip(gf, cd_on_grid, strict=True)) + "\n")
    return run_scorer("physics", "phonon_dos_mae",
                      ["--ref", str(ref_csv), "--pred", str(pred_csv),
                       "--x-column", "frequency", "--column", "dos"])["value"]


def _score_cell(cand_lengths, mof: str, refs: ReleaseRefs) -> dict:
    a, b, c = refs.dft_cell_lengths(mof)
    per_axis = []
    for pred, ref in zip(cand_lengths, (a, b, c), strict=True):
        per_axis.append(abs(run_scorer("physics", "unit_cell_length_percent_error",
                                       ["--pred", repr(pred), "--ref", repr(ref)])["value"]))
    return {"per_axis_abs_pct": per_axis, "mean_abs_pct": sum(per_axis) / 3.0}


def _score_spacegroup(relaxed: dict, mof: str, refs: ReleaseRefs) -> dict:
    from ase import Atoms

    from .symmetry import SymmetryUnavailable, spacegroup_symbol

    try:
        cand = spacegroup_symbol(Atoms(numbers=relaxed["numbers"],
                                       scaled_positions=relaxed["scaled_positions"],
                                       cell=relaxed["cell"], pbc=True))
        ref = refs.dft_space_group(mof)
    except SymmetryUnavailable as exc:
        return {"value": None, "reason": str(exc)}
    res = run_scorer("physics", "space_group_match", ["--pred", cand, "--ref", ref])
    return {"value": bool(res["value"]), "pred": cand, "ref": ref}


def _score_bulk_modulus(eos: list, mof: str, refs: ReleaseRefs) -> dict:
    ref = refs.dft_bulk_modulus(mof)
    if ref is None:
        return {"value": None, "reason": "no DFT bulk modulus in release"}
    from ase.eos import EquationOfState

    vols = [v for v, _ in eos]
    ens = [e for _, e in eos]
    try:
        _, _, b_ev_a3 = EquationOfState(vols, ens, eos="birchmurnaghan").fit()
    except Exception as exc:  # noqa: BLE001
        return {"value": None, "reason": f"EOS fit failed: {exc}"}
    b_gpa = b_ev_a3 * EV_A3_TO_GPA
    rel = run_scorer("physics", "bulk_modulus_relative_error",
                     ["--pred", repr(b_gpa), "--ref", repr(ref)])["value"]
    return {"value": rel, "candidate_GPa": b_gpa, "dft_GPa": ref}


def run_stage_b(descriptor: dict, out_dir: Path, *, mofs=None, sandbox: bool = True,
                protocol: dict, seed: int = 2024, with_qha: bool = False) -> dict:
    refs = ReleaseRefs()
    mofs = mofs or list(AVAILABILITY)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {}

    for mof in mofs:
        work = out_dir / mof
        work.mkdir(parents=True, exist_ok=True)
        init = refs.initial_structure(mof)
        sc = _supercell_for(init, protocol["supercell_min_length_A"], protocol["supercell_cap"])
        struct_path = work / "initial.xyz"
        from ase.io import write

        write(struct_path, init, format="extxyz")

        payload = {
            "descriptor": descriptor, "structure_path": str(struct_path),
            "protocol": protocol, "supercell": sc, "work": str(work), "seed": seed,
        }
        if sandbox:
            cand = run_in_sandbox(relax_and_phonons_target, payload, write_allow=[str(work)],
                                  timeout=protocol["stage_b_timeout_s"])
        else:
            cand = relax_and_phonons_target(payload)
        (work / "candidate_raw.json").write_text(json.dumps(cand, indent=2))

        av = AVAILABILITY[mof]
        endpoints: dict = {}
        endpoints["phonon_dos_mae"] = _score_dos(cand["phonon"], mof, refs, work) if av["phonon_dos_mae"] else None
        endpoints["phonon_frequency_rmse"] = None  # no DFT band structure released
        freqs_file = work / "freqs.txt"
        freqs_file.write_text("\n".join(repr(x) for x in cand["phonon"]["frequencies_THz"]) + "\n")
        endpoints["imaginary_mode_count"] = run_scorer(
            "physics", "imaginary_mode_count",
            ["--input", str(freqs_file), "--threshold", repr(protocol["imaginary_threshold_THz"])])["value"]
        cell = _score_cell(cand["relaxed"]["cell_lengths"], mof, refs)
        endpoints["unit_cell_length_percent_error"] = cell["mean_abs_pct"]
        sg = _score_spacegroup(cand["relaxed"], mof, refs)
        endpoints["space_group_match"] = sg["value"]
        bm = _score_bulk_modulus(cand["eos"], mof, refs)
        endpoints["bulk_modulus_relative_error"] = bm["value"]
        if with_qha and av["thermal_expansion_relative_error"]:
            endpoints["thermal_expansion_relative_error"] = _score_thermal_expansion(
                descriptor, mof, refs, work, protocol, sandbox)
        else:
            endpoints["thermal_expansion_relative_error"] = None

        result[mof] = {
            "endpoints": endpoints,
            "availability": av,
            "details": {"cell": cell, "space_group": sg, "bulk_modulus": bm,
                        "min_frequency_THz": cand["phonon"]["min_frequency_THz"],
                        "supercell": sc, "relax": cand["relaxed"]},
        }
    return result


def _score_thermal_expansion(descriptor, mof, refs, work, protocol, sandbox) -> float | None:
    """QHA CTE vs DFT NegativeThermalExpansion.csv at the protocol reference T.

    Off by default (--with-qha): needs phonons at several volumes and roughly
    triples Stage-B cost. Returns the relative error of alpha_V at T_ref.
    """
    dft = refs.dft_cte(mof)
    if dft is None:
        return None
    # Implemented as a follow-up: run phonons at protocol["qha_strains"], feed
    # phonopy-qha, read alpha_V(T_ref). Until wired, be explicit:
    raise NotImplementedError("QHA path is gated; run with --with-qha only after it is validated")
