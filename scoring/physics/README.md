# Physics Metrics

Physics metrics are organized around the MOF phonon benchmark quantities used in
Elena et al., npj Computational Materials 2025: equilibrium cell geometry,
space-group preservation, phonon DOS / frequency errors, imaginary-mode removal,
bulk modulus, and thermal expansion.

Each metric is a standalone Bash-callable Python file:

```bash
python scoring/physics/unit_cell_length_percent_error.py --pred 10.1 --ref 10.0
python scoring/physics/space_group_match.py --pred Fm-3m --ref Fm-3m
python scoring/physics/imaginary_mode_count.py --frequencies -0.01,0.0,1.2 --threshold -0.0001
python scoring/physics/phonon_frequency_rmse.py --y-true 1,2,3 --y-pred 1,2,4
python scoring/physics/bulk_modulus_relative_error.py --pred 18.0 --ref 20.0
```
