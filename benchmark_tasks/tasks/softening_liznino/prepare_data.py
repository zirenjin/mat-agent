from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from ase.io import write


def atoms_from_record(frame_id: str, material_id: str, record: dict) -> Atoms:
    structure = record["structure"]
    lattice = structure["lattice"]["matrix"]
    sites = structure["sites"]
    symbols = [site["species"][0]["element"] for site in sites]
    positions = [site["xyz"] for site in sites]
    atoms = Atoms(symbols=symbols, positions=positions, cell=lattice, pbc=True)
    atoms.info["frame_id"] = frame_id
    atoms.info["material_id"] = material_id
    atoms.info["ref_energy"] = float(record["vasp_e"])
    atoms.arrays["ref_forces"] = np.asarray(record["vasp_f"], dtype=float)
    if "vasp_s" in record:
        atoms.info["ref_stress"] = np.asarray(record["vasp_s"], dtype=float).reshape(-1).tolist()
    return atoms


def deterministic_records(data: dict, max_frames: int) -> list[Atoms]:
    atoms_list: list[Atoms] = []
    for material_id in sorted(data):
        for frame_id in sorted(data[material_id]):
            atoms_list.append(atoms_from_record(frame_id, material_id, data[material_id][frame_id]))
            if len(atoms_list) >= max_frames:
                return atoms_list
    return atoms_list


def choose_splits(atoms_list: list[Atoms], oracle_n: int, validation_n: int, hidden_n: int) -> tuple[list[Atoms], list[Atoms], list[Atoms]]:
    all_elements = {symbol for atoms in atoms_list for symbol in atoms.get_chemical_symbols()}
    validation: list[Atoms] = []
    validation_ids: set[str] = set()
    covered: set[str] = set()

    for atoms in atoms_list:
        elements = set(atoms.get_chemical_symbols())
        if not elements <= covered:
            validation.append(atoms)
            validation_ids.add(str(atoms.info["frame_id"]))
            covered |= elements
        if covered >= all_elements:
            break

    for atoms in atoms_list:
        if len(validation) >= validation_n:
            break
        frame_id = str(atoms.info["frame_id"])
        if frame_id not in validation_ids:
            validation.append(atoms)
            validation_ids.add(frame_id)

    visible_elements = {symbol for atoms in validation for symbol in atoms.get_chemical_symbols()}
    hidden: list[Atoms] = []
    hidden_ids: set[str] = set()
    for atoms in atoms_list:
        frame_id = str(atoms.info["frame_id"])
        if frame_id in validation_ids:
            continue
        if set(atoms.get_chemical_symbols()) <= visible_elements:
            hidden.append(atoms)
            hidden_ids.add(frame_id)
        if len(hidden) >= hidden_n:
            break

    oracle: list[Atoms] = []
    for atoms in atoms_list:
        frame_id = str(atoms.info["frame_id"])
        if frame_id not in validation_ids and frame_id not in hidden_ids:
            oracle.append(atoms)
        if len(oracle) >= oracle_n:
            break

    if len(oracle) < oracle_n or len(validation) < validation_n or len(hidden) < hidden_n:
        raise ValueError("Could not construct requested element-covered reduced split")
    return oracle, validation, hidden


def write_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reduced WBM softening data for the first benchmark experiment.")
    parser.add_argument("--raw-json", default="data/raw/benchmark/softening_liznino/WBM_high_energy_states.json")
    parser.add_argument("--output", default="data/prepared/benchmark/softening_liznino/wbm_softening_reduced.extxyz")
    parser.add_argument("--split-manifest", default="benchmark/envs/softening_liznino/split_manifest.yaml")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--oracle-n", type=int, default=60)
    parser.add_argument("--validation-n", type=int, default=30)
    parser.add_argument("--hidden-n", type=int, default=30)
    args = parser.parse_args()

    raw_path = Path(args.raw_json)
    data = json.loads(raw_path.read_text())
    atoms_list = deterministic_records(data, args.max_frames)
    if len(atoms_list) < args.oracle_n + args.validation_n + args.hidden_n:
        raise ValueError("Reduced subset is smaller than requested splits")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, atoms_list, format="extxyz")

    oracle, validation, hidden = choose_splits(atoms_list, args.oracle_n, args.validation_n, args.hidden_n)

    manifest_path = Path(args.split_manifest)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["strategy"] = "deterministic_reduced_wbm_with_validation_element_coverage"
    manifest["oracle_pool"]["n"] = len(oracle)
    manifest["validation"]["n"] = len(validation)
    manifest["hidden_test"]["n"] = len(hidden)
    manifest["status"] = "prepared_reduced_wbm"

    write_ids(Path(manifest["oracle_pool"]["ids_file"]), [a.info["frame_id"] for a in oracle])
    write_ids(Path(manifest["validation"]["ids_file"]), [a.info["frame_id"] for a in validation])
    write_ids(Path(manifest["hidden_test"]["ids_file"]), [a.info["frame_id"] for a in hidden])
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))

    summary = {
        "raw_json": str(raw_path),
        "output": str(output),
        "n_total_raw_frames": sum(len(v) for v in data.values()),
        "n_total_raw_materials": len(data),
        "n_reduced_frames": len(atoms_list),
        "oracle_n": len(oracle),
        "validation_n": len(validation),
        "hidden_n": len(hidden),
        "elements": sorted({symbol for atoms in atoms_list for symbol in atoms.get_chemical_symbols()}),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
