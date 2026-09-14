"""Stage A -- ML generalization on the sequestered framework holdout.

Reads the evaluator-only sequestered file, runs sandboxed candidate inference on
geometry only, then scores energy / force / stress against DFT labels using the
frozen scorers in ``scoring/machine_learning/``. No metric is computed here.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from .metrics import run_scorer
from .predict import predict_frames

REPO = Path(__file__).resolve().parents[3]
SEQUESTERED = REPO / "data/mace_mof_0/discovery/_sequestered/sequestered_ml.xyz"
METALS = {"Zn", "Zr", "Al", "Mg", "Ti", "Ce", "Hf", "Cu", "Cd", "In", "Ga", "Sn", "Sr"}


def _fw_id(system_name: str) -> str:
    return "fw_" + hashlib.sha256(system_name.encode()).hexdigest()[:6]


def _voigt_from_matrix(m: list[list[float]]) -> list[float]:
    return [m[0][0], m[1][1], m[2][2], m[1][2], m[0][2], m[0][1]]


def _write_col(path: Path, values: list[float]) -> None:
    path.write_text("v\n" + "\n".join(repr(float(x)) for x in values) + "\n")


def _score_pair(tag: str, y_true: list[float], y_pred: list[float], work: Path) -> dict[str, float]:
    tf, pf = work / f"{tag}_true.csv", work / f"{tag}_pred.csv"
    _write_col(tf, y_true)
    _write_col(pf, y_pred)
    args = ["--true-file", str(tf), "--pred-file", str(pf), "--column", "v"]
    rmse = run_scorer("machine_learning", "rmse", args)["value"]
    mae = run_scorer("machine_learning", "mae", args)["value"]
    return {"rmse": rmse, "mae": mae, "n": len(y_true)}


def _load_labeled(path: Path):
    from ase.io import read

    frames = read(str(path), index=":")
    return frames if isinstance(frames, list) else [frames]


def run_stage_a(descriptor: dict, out_dir: Path, *, sandbox: bool = True,
                sequestered_path: Path | None = None) -> dict:
    src = sequestered_path or SEQUESTERED
    if not src.is_file():
        raise FileNotFoundError(f"sequestered-ML file missing: {src} (run build_discovery_splits.py)")
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = _load_labeled(src)
    preds = predict_frames(descriptor, frames, out_dir / "infer", sandbox=sandbox)
    if len(preds) != len(frames):
        raise RuntimeError(f"prediction count {len(preds)} != frame count {len(frames)}")

    e_t, e_p = [], []
    f_t, f_p = [], []
    s_t, s_p = [], []
    per_fw: dict[str, dict] = defaultdict(lambda: {"e_t": [], "e_p": [], "f_t": [], "f_p": [], "metal": "?", "natoms": 0})

    for atoms, pr in zip(frames, preds, strict=True):
        nat = len(atoms)
        info = atoms.info
        name = str(info.get("system_name", "unknown"))
        fw = _fw_id(name)
        rec = per_fw[fw]
        rec["metal"] = next((s for s in atoms.get_chemical_symbols() if s in METALS), "?")
        rec["natoms"] = nat

        e_ref = float(info["dft_energy"]) / nat * 1000.0
        e_prd = float(pr["energy"]) / nat * 1000.0
        e_t.append(e_ref)
        e_p.append(e_prd)
        rec["e_t"].append(e_ref)
        rec["e_p"].append(e_prd)

        fr = atoms.arrays["dft_forces"].reshape(-1) * 1000.0
        fp = [c * 1000.0 for row in pr["forces"] for c in row]
        f_t.extend(fr.tolist())
        f_p.extend(fp)
        rec["f_t"].extend(fr.tolist())
        rec["f_p"].extend(fp)

        if pr["stress"] is not None and "dft_stress" in info:
            dft_s = info["dft_stress"]
            if hasattr(dft_s, "tolist"):
                dft_s = dft_s.tolist()
            ref_v = _voigt_from_matrix(dft_s) if (isinstance(dft_s, list) and dft_s and isinstance(dft_s[0], list)) else list(dft_s)
            s_t.extend([x * 1000.0 for x in ref_v])
            s_p.extend([x * 1000.0 for x in pr["stress"]])

    work = out_dir / "scorer_inputs"
    work.mkdir(exist_ok=True)
    overall = {
        "energy_meV_per_atom": _score_pair("energy", e_t, e_p, work),
        "force_meV_per_A": _score_pair("force", f_t, f_p, work),
    }
    if s_t:
        overall["stress_meV_per_A3"] = _score_pair("stress", s_t, s_p, work)

    by_framework = {}
    for fw, rec in sorted(per_fw.items()):
        by_framework[fw] = {
            "metal": rec["metal"],
            "natoms": rec["natoms"],
            "n_frames": len(rec["e_t"]),
            "energy_rmse_meV_per_atom": _score_pair(f"{fw}_e", rec["e_t"], rec["e_p"], work)["rmse"],
            "force_rmse_meV_per_A": _score_pair(f"{fw}_f", rec["f_t"], rec["f_p"], work)["rmse"],
        }

    flat = {
        "energy_rmse_meV_per_atom": overall["energy_meV_per_atom"]["rmse"],
        "energy_mae_meV_per_atom": overall["energy_meV_per_atom"]["mae"],
        "force_rmse_meV_per_A": overall["force_meV_per_A"]["rmse"],
        "force_mae_meV_per_A": overall["force_meV_per_A"]["mae"],
    }
    if "stress_meV_per_A3" in overall:
        flat["stress_rmse_meV_per_A3"] = overall["stress_meV_per_A3"]["rmse"]
        flat["stress_mae_meV_per_A3"] = overall["stress_meV_per_A3"]["mae"]

    return {"overall": flat, "by_framework": by_framework,
            "n_frames": len(frames), "n_systems": len(per_fw)}
