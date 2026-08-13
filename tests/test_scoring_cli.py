from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ScoringCliTest(unittest.TestCase):
    def run_json(self, *args: str) -> dict[str, object]:
        proc = subprocess.run([sys.executable, *args], cwd=ROOT, check=True, text=True, capture_output=True)
        return json.loads(proc.stdout)

    def test_machine_learning_mae_cli(self) -> None:
        out = self.run_json('scoring/machine_learning/mae.py', '--y-true', '1,2,3', '--y-pred', '1,2,5')
        self.assertEqual(out['metric'], 'mae')
        self.assertEqual(out['value'], 2 / 3)

    def test_machine_learning_rmse_cli(self) -> None:
        out = self.run_json('scoring/machine_learning/rmse.py', '--y-true', '1,2,3', '--y-pred', '1,2,5')
        self.assertEqual(out['metric'], 'rmse')
        self.assertEqual(round(float(out['value']), 6), round((4 / 3) ** 0.5, 6))

    def test_machine_learning_r2_cli(self) -> None:
        out = self.run_json('scoring/machine_learning/r2.py', '--y-true', '1,2,3', '--y-pred', '1,2,3')
        self.assertEqual(out['metric'], 'r2')
        self.assertEqual(out['value'], 1.0)

    def test_physics_unit_cell_percent_error_cli(self) -> None:
        out = self.run_json('scoring/physics/mace_mof_0/unit_cell_length_percent_error.py', '--pred', '10.5', '--ref', '10.0')
        self.assertEqual(out['metric'], 'unit_cell_length_percent_error')
        self.assertEqual(out['value'], 5.0)

    def test_physics_imaginary_mode_count_cli(self) -> None:
        out = self.run_json('scoring/physics/mace_mof_0/imaginary_mode_count.py', '--frequencies=-0.01,-0.00001,1.0', '--threshold', '-0.0001')
        self.assertEqual(out['metric'], 'imaginary_mode_count')
        self.assertEqual(out['value'], 1.0)


if __name__ == '__main__':
    unittest.main()
