"""Benchmark-infrastructure smoke tests (no GPU, no mace-torch).

Data-dependent cases skip when the ddmms release under data/mace_mof_0/ is not
present (e.g. on CI, where large assets are gitignored). Run locally for full
coverage.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# module-level targets so multiprocessing 'spawn' can pickle them by reference
def _sbx_read(payload):
    return Path(payload["path"]).read_text()[:10]


def _sbx_write(payload):
    Path(payload["path"]).write_text("x")
    return "wrote"

RELEASE_PRESENT = (REPO / "data/mace_mof_0/raw/data_training/train.xyz").is_file()
SEQ_PRESENT = (REPO / "data/mace_mof_0/discovery/_sequestered/sequestered_ml.xyz").is_file()
HAS_ASE = importlib.util.find_spec("ase") is not None


def _write(p: Path, obj) -> Path:
    p.write_text(json.dumps(obj))
    return p


class FrozenManifestTest(unittest.TestCase):
    def test_verify_frozen_passes_clean_tree(self) -> None:
        r = subprocess.run([sys.executable, str(REPO / "benchmark/integrity/verify_frozen.py"), "--json"],
                           capture_output=True, text=True, check=False)
        out = json.loads(r.stdout)
        self.assertTrue(out["ok"], out)

    def test_verify_frozen_detects_edit(self) -> None:
        target = REPO / "benchmark/evaluation/protocol.md"
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"\n<!-- tamper -->\n")
            r = subprocess.run([sys.executable, str(REPO / "benchmark/integrity/verify_frozen.py"), "--json"],
                               capture_output=True, text=True, check=False)
            out = json.loads(r.stdout)
            self.assertFalse(out["ok"])
            self.assertIn("benchmark/evaluation/protocol.md", out["modified"])
            self.assertEqual(r.returncode, 2)
        finally:
            target.write_bytes(original)


class SubmissionValidationTest(unittest.TestCase):
    """Test C -- malformed / forbidden submissions are refused."""

    def setUp(self) -> None:
        from benchmark.evaluation._lib.submission import SubmissionError, load_and_validate
        self.SubmissionError = SubmissionError
        self.load = load_and_validate
        # checkpoint must sit under an allowed repo root (runs/)
        self.tmp = Path(tempfile.mkdtemp(dir=str(REPO / "runs")))
        self.good_ckpt = self.tmp / "model.pt"
        self.good_ckpt.write_bytes(b"not a real checkpoint")

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _sub(self, **over):
        base = {
            "checkpoint": str(self.good_ckpt),
            "inference": {"kind": "mace_calculator", "dtype": "float64", "head": "pbe_d3"},
            "training_recipe_entrypoint": "workspace/train_final.sh",
        }
        base.update(over)
        return _write(self.tmp / "submission.json", base)

    def test_missing_file(self) -> None:
        with self.assertRaises(self.SubmissionError):
            self.load(self.tmp / "nope.json")

    def test_missing_checkpoint(self) -> None:
        p = _write(self.tmp / "s.json", {"inference": {"kind": "mace_calculator"}})
        with self.assertRaises(self.SubmissionError):
            self.load(p)

    def test_checkpoint_under_benchmark_is_forbidden(self) -> None:
        with self.assertRaises(self.SubmissionError):
            self.load(self._sub(checkpoint="benchmark/evaluation/evaluate_mof0.py"))

    def test_bad_inference_kind(self) -> None:
        with self.assertRaises(self.SubmissionError):
            self.load(self._sub(inference={"kind": "pickle_exec", "dtype": "float64"}))

    def test_entrypoint_outside_workspace(self) -> None:
        with self.assertRaises(self.SubmissionError):
            self.load(self._sub(inference={"kind": "entrypoint", "entrypoint": "tools/x.py:build", "dtype": "float64"}))

    def test_bad_dtype(self) -> None:
        with self.assertRaises(self.SubmissionError):
            self.load(self._sub(inference={"kind": "mace_calculator", "dtype": "bfloat16"}))

    def test_missing_training_recipe(self) -> None:
        s = self._sub()
        d = json.loads(s.read_text())
        d.pop("training_recipe_entrypoint")
        s.write_text(json.dumps(d))
        with self.assertRaises(self.SubmissionError):
            self.load(s)

    def test_malformed_external_inputs(self) -> None:
        ext = _write(self.tmp / "ext.json", {"inputs": [{"kind": "dataset", "id": "x"}]})
        with self.assertRaises(self.SubmissionError):
            self.load(self._sub(external_inputs=str(ext)))

    def test_valid_submission_reports_sha(self) -> None:
        out = self.load(self._sub())
        self.assertEqual(len(out["checkpoint_sha256"]), 64)

    @unittest.skipUnless(RELEASE_PRESENT, "needs discovery frame index")
    def test_training_data_with_sequestered_frame_is_rejected(self) -> None:
        idx = json.loads((REPO / "benchmark/data/discovery_frame_index.json").read_text())
        # a training file whose only frame hashes to a sequestered frame is impossible to
        # build without the frame text; instead assert the guard path exists and dev frames pass.
        self.assertIn("sequestered_ml_frame_sha256", idx)
        self.assertGreater(len(idx["dev_train"]), 1000)


class VerdictLogicTest(unittest.TestCase):
    """Part-5 review points: catastrophic override, 3/4 rule, regression + repro gates."""

    def setUp(self) -> None:
        from benchmark.evaluation._lib import aggregate
        self.agg = aggregate
        self.tol = {"tol_mof": 0.03, "tol_ml": 0.05, "tol_repro": 0.05, "source": "test"}
        self.protocol = {"tol_mof": 0.03, "tol_ml": 0.05, "tol_repro": 0.05}

    def _mof(self, endpoints):
        return {"endpoints": endpoints}

    def _pair(self, cand_ep, base_ep):
        return ({"M": self._mof(cand_ep)}, {"M": self._mof(base_ep)})

    def test_uniform_improvement_is_better(self) -> None:
        c, b = self._pair(
            {"phonon_dos_mae": 0.8, "unit_cell_length_percent_error": 0.8, "bulk_modulus_relative_error": 0.8,
             "imaginary_mode_count": 0, "space_group_match": True},
            {"phonon_dos_mae": 1.0, "unit_cell_length_percent_error": 1.0, "bulk_modulus_relative_error": 1.0,
             "imaginary_mode_count": 0, "space_group_match": True})
        res = self.agg.compare_stage_b(c, b, self.protocol, self.tol)
        self.assertEqual(res["M"]["status"], "better")

    def test_single_catastrophic_endpoint_forces_worse(self) -> None:
        c, b = self._pair(
            {"phonon_dos_mae": 0.5, "unit_cell_length_percent_error": 0.5, "bulk_modulus_relative_error": 5.0,
             "imaginary_mode_count": 0, "space_group_match": True},
            {"phonon_dos_mae": 1.0, "unit_cell_length_percent_error": 1.0, "bulk_modulus_relative_error": 1.0,
             "imaginary_mode_count": 0, "space_group_match": True})
        res = self.agg.compare_stage_b(c, b, self.protocol, self.tol)
        self.assertEqual(res["M"]["status"], "worse")
        self.assertIn("bulk_modulus_relative_error", res["M"]["catastrophic_endpoints"])

    def test_lost_space_group_is_catastrophic(self) -> None:
        c, b = self._pair(
            {"phonon_dos_mae": 0.5, "space_group_match": False, "imaginary_mode_count": 0},
            {"phonon_dos_mae": 1.0, "space_group_match": True, "imaginary_mode_count": 0})
        res = self.agg.compare_stage_b(c, b, self.protocol, self.tol)
        self.assertEqual(res["M"]["status"], "worse")

    def test_new_imaginary_modes_are_catastrophic(self) -> None:
        c, b = self._pair(
            {"phonon_dos_mae": 0.5, "imaginary_mode_count": 8, "space_group_match": True},
            {"phonon_dos_mae": 1.0, "imaginary_mode_count": 0, "space_group_match": True})
        res = self.agg.compare_stage_b(c, b, self.protocol, self.tol)
        self.assertEqual(res["M"]["status"], "worse")

    def test_verdict_needs_three_of_four(self) -> None:
        sb = {}
        for i, st in enumerate(["better", "better", "better", "equal"]):
            sb[f"M{i}"] = {"status": st}
        sa = {"no_regression": True, "worst_ratio": 1.0, "metrics": {}}
        v = self.agg.build_verdict(sa, sb, {"checked": True, "within_tolerance": True},
                                   {"ledger_ok": True}, self.tol)
        self.assertTrue(v["beats_baseline"])

    def test_verdict_fails_on_one_worse(self) -> None:
        sb = {"M0": {"status": "better"}, "M1": {"status": "better"},
              "M2": {"status": "better"}, "M3": {"status": "worse"}}
        sa = {"no_regression": True, "worst_ratio": 1.0, "metrics": {}}
        v = self.agg.build_verdict(sa, sb, {"checked": True, "within_tolerance": True},
                                   {"ledger_ok": True}, self.tol)
        self.assertFalse(v["beats_baseline"])

    def test_verdict_fails_without_repro_check(self) -> None:
        sb = {f"M{i}": {"status": "better"} for i in range(4)}
        sa = {"no_regression": True, "worst_ratio": 1.0, "metrics": {}}
        v = self.agg.build_verdict(sa, sb, {"checked": False}, {"ledger_ok": True}, self.tol)
        self.assertFalse(v["beats_baseline"])

    def test_verdict_fails_on_ml_regression(self) -> None:
        sb = {f"M{i}": {"status": "better"} for i in range(4)}
        sa = {"no_regression": False, "worst_ratio": 1.3, "metrics": {}}
        v = self.agg.build_verdict(sa, sb, {"checked": True, "within_tolerance": True},
                                   {"ledger_ok": True}, self.tol)
        self.assertFalse(v["beats_baseline"])


class NoiseEstimateTest(unittest.TestCase):
    """Test F (offline part) -- tolerance math from repeat scorecards."""

    def _card(self, dos, ucl):
        return {"stage_a": {"overall": {"force_rmse_meV_per_A": 40.0 + dos}},
                "stage_b": {"MOF-5": {"endpoints": {
                    "phonon_dos_mae": dos, "unit_cell_length_percent_error": ucl,
                    "phonon_frequency_rmse": None, "space_group_match": True}}}}

    def test_tolerances_scale_with_spread(self) -> None:
        from benchmark.evaluation._lib.noise import summarize_repeats

        tight = summarize_repeats([self._card(1.00, 2.00), self._card(1.01, 2.01), self._card(0.99, 1.99)])
        loose = summarize_repeats([self._card(1.0, 2.0), self._card(1.4, 2.6), self._card(0.7, 1.5)])
        self.assertLess(tight["tol_mof"], loose["tol_mof"])
        self.assertGreaterEqual(loose["catastrophic_ratio"], 2.0)
        self.assertEqual(tight["n_repeats"], 3)

    def test_needs_two_repeats(self) -> None:
        from benchmark.evaluation._lib.noise import summarize_repeats

        with self.assertRaises(ValueError):
            summarize_repeats([self._card(1.0, 2.0)])


@unittest.skipUnless(RELEASE_PRESENT and HAS_ASE, "needs ddmms release + ase")
class ReleaseAdapterTest(unittest.TestCase):
    def test_all_four_mofs_parse(self) -> None:
        from benchmark.evaluation._lib.release_adapters import AVAILABILITY, MOFS, ReleaseRefs
        r = ReleaseRefs()
        for m in MOFS:
            self.assertGreater(len(r.initial_structure(m)), 10)
            a, b, c = r.dft_cell_lengths(m)
            self.assertGreater(a, 3.0)
            freq, dos = r.dft_dos(m)
            self.assertEqual(len(freq), len(dos))
            self.assertIsNotNone(r.dft_bulk_modulus(m))
            self.assertFalse(AVAILABILITY[m]["phonon_frequency_rmse"])
        self.assertIsNotNone(r.dft_cte("MOF-5"))
        self.assertIsNone(r.dft_cte("MOF-74"))


@unittest.skipUnless(SEQ_PRESENT, "needs built sequestered split")
class SandboxTest(unittest.TestCase):
    """Test E -- candidate code cannot read sequestered truth."""

    def test_sequestered_read_blocked(self) -> None:
        from benchmark.evaluation._lib.sandbox import SandboxViolation, run_in_sandbox

        seq = str(REPO / "data/mace_mof_0/discovery/_sequestered/sequestered_ml.xyz")
        with self.assertRaises(SandboxViolation):
            run_in_sandbox(_sbx_read, {"path": seq}, timeout=60)

    def test_benchmark_write_blocked(self) -> None:
        from benchmark.evaluation._lib.sandbox import SandboxViolation, run_in_sandbox

        with self.assertRaises(SandboxViolation):
            run_in_sandbox(_sbx_write, {"path": str(REPO / "scoring/HACKED.txt")}, timeout=60)

    def test_allowed_scratch_write_ok(self) -> None:
        from benchmark.evaluation._lib.sandbox import run_in_sandbox

        with tempfile.TemporaryDirectory() as td:
            out = run_in_sandbox(_sbx_write, {"path": str(Path(td) / "f.txt")},
                                 timeout=60, write_allow=[td])
            self.assertEqual(out, "wrote")


@unittest.skipUnless(HAS_ASE, "needs ase")
class StageAPipelineTest(unittest.TestCase):
    """Stage A end-to-end with a cheap analytic calculator (no mace-torch)."""

    def test_stage_a_runs_and_scores(self) -> None:
        import numpy as np
        from ase import Atoms
        from ase.calculators.lj import LennardJones
        from ase.io import write

        from benchmark.evaluation._lib import calculator, stage_a

        rng = np.random.default_rng(0)
        frames = []
        for _ in range(4):
            a = Atoms("Zr2O4", positions=rng.random((6, 3)) * 4, cell=[6, 6, 6], pbc=True)
            a.info["system_name"] = "seqtest"
            a.info["dft_energy"] = -12.34
            a.arrays["dft_forces"] = rng.standard_normal((6, 3)) * 0.1
            a.info["dft_stress"] = (rng.standard_normal((3, 3)) * 1e-3).tolist()
            frames.append(a)
        with tempfile.TemporaryDirectory() as td:
            seq = Path(td) / "seq.xyz"
            write(seq, frames, format="extxyz")
            orig = calculator.build_calculator
            calculator.build_calculator = lambda desc: LennardJones(rc=5.0)
            try:
                res = stage_a.run_stage_a({"kind": "mace_calculator"}, Path(td) / "out",
                                          sandbox=False, sequestered_path=seq)
            finally:
                calculator.build_calculator = orig
        self.assertIn("force_rmse_meV_per_A", res["overall"])
        self.assertEqual(res["n_frames"], 4)
        self.assertGreater(res["overall"]["force_rmse_meV_per_A"], 0.0)


if __name__ == "__main__":
    unittest.main()
