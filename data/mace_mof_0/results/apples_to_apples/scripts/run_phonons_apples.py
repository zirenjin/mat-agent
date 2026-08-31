#!/usr/bin/env python3
"""Run paper-aligned Janus/Phonopy phonons with an explicit 11x11x11 mesh."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from janus_core.calculations.phonons import Phonons


def load_protocol(path: Path) -> dict:
    with path.open("r", encoding="utf8") as handle:
        return yaml.safe_load(handle)


def run_one(protocol: dict, material: str, output_root: Path) -> None:
    phonon_cfg = protocol["phonons"]
    mat_cfg = protocol["materials"][material]
    out_dir = output_root / material
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = out_dir / material
    calc_kwargs = {"model": protocol["model"]["path"]}
    if protocol["model"].get("head"):
        calc_kwargs["head"] = protocol["model"]["head"]
    calc = Phonons(
        struct_path=mat_cfg["structure"],
        arch="mace_mp",
        device="cuda",
        calc_kwargs=calc_kwargs,
        calcs=[],
        supercell=mat_cfg["supercell"],
        displacement=phonon_cfg["finite_difference_displacement_angstrom"],
        minimize=True,
        minimize_kwargs={"fmax": phonon_cfg["relaxation"]["fmax_ev_per_angstrom"]},
        temp_min=phonon_cfg["temperature_grid_K"]["min"],
        temp_max=phonon_cfg["temperature_grid_K"]["max"],
        temp_step=phonon_cfg["temperature_grid_K"]["step"],
        write_full=phonon_cfg["write_full"],
        file_prefix=prefix,
    )

    calc.calc_force_constants()
    calc.write_force_constants()
    calc.calc_bands()
    calc.calc_dos(mesh=phonon_cfg["mesh"])

    # Janus 0.6.4 CLI does not expose mesh for thermal properties; set it explicitly.
    calc.results["phonon"].run_mesh(phonon_cfg["mesh"])
    calc.results["phonon"].run_thermal_properties(
        t_min=phonon_cfg["temperature_grid_K"]["min"],
        t_max=phonon_cfg["temperature_grid_K"]["max"],
        t_step=phonon_cfg["temperature_grid_K"]["step"],
    )
    calc.results["thermal_properties"] = calc.results["phonon"].get_thermal_properties_dict()
    calc.write_thermal_props()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="data/mace_mof_0/results/apples_to_apples/configs/phonon_protocol.yml")
    parser.add_argument("--output-root", default="data/mace_mof_0/results/apples_to_apples/outputs/phonons")
    parser.add_argument("materials", nargs="*")
    args = parser.parse_args()

    protocol = load_protocol(Path(args.protocol))
    materials = args.materials or list(protocol["materials"])
    for name in materials:
        run_one(protocol, name, Path(args.output_root))
