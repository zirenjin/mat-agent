# MACE-MOF-0 Physics Benchmark

Paper: Towards universal machine learning force fields for metal-organic frameworks
DOI: `10.1038/s41524-025-01611-8`
Public assets: https://github.com/ddmms/data/tree/main/mace-mof-0

This folder groups the physics tests aligned with the MACE-MOF-0 public model/data
release. The public repository primarily provides model and data artifacts, so the
scripts here are thin Bash-first metric entrypoints that operate on those artifacts
or on derived scalar/vector outputs from a model run.

Downloaded assets are stored outside git under `data/mace_mof_0/`; the tracked
provenance file is `data/mace_mof_0/manifest.json`.

## CLI Commands

| Script | Property Tested | CLI Example |
| --- | --- | --- |
| `unit_cell_length_percent_error.py` | Percent deviation of equilibrium unit-cell length from the DFT/reference value. | `python scoring/physics/mace_mof_0/unit_cell_length_percent_error.py --pred 10.1 --ref 10.0` |
| `space_group_match.py` | Whether predicted and reference space-group labels are preserved after optimization. | `python scoring/physics/mace_mof_0/space_group_match.py --pred Fm-3m --ref Fm-3m` |
| `imaginary_mode_count.py` | Number of phonon frequencies below an imaginary-mode threshold. | `python scoring/physics/mace_mof_0/imaginary_mode_count.py --frequencies=-0.01,-0.00001,1.0 --threshold -0.0001` |
| `phonon_frequency_rmse.py` | RMSE between predicted and reference phonon frequencies on matched modes. | `python scoring/physics/mace_mof_0/phonon_frequency_rmse.py --y-true 1,2,3 --y-pred 1,2,4` |
| `phonon_dos_mae.py` | MAE between predicted and reference phonon DOS samples on the same grid. | `python scoring/physics/mace_mof_0/phonon_dos_mae.py --y-true 1,2,3 --y-pred 1,2,4` |
| `bulk_modulus_relative_error.py` | Relative error in bulk modulus. | `python scoring/physics/mace_mof_0/bulk_modulus_relative_error.py --pred 18.0 --ref 20.0` |
| `thermal_expansion_relative_error.py` | Relative error in thermal expansion coefficient. | `python scoring/physics/mace_mof_0/thermal_expansion_relative_error.py --pred -3.1 --ref -3.0` |

## Asset Layout

```text
data/mace_mof_0/models/mofs_v1.model
data/mace_mof_0/models/mofs_v2.model
data/mace_mof_0/raw/data_training.tar.xz
data/mace_mof_0/raw/127 Curated Dataset.zip
data/mace_mof_0/raw/curated_test.zip
data/mace_mof_0/raw/optimized structures.zip
data/mace_mof_0/raw/BulkModulus_data.zip
data/mace_mof_0/raw/NegativeThermalExpansion.csv
data/mace_mof_0/raw/phonons/*.zip
```
