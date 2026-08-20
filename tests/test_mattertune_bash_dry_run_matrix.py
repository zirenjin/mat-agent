from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


class MatterTuneBashDryRunMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.tmp_path = Path(cls.tmp.name)
        cls.fake_root = cls.tmp_path / "fakepkg"
        cls.fake_mattertune = cls.fake_root / "mattertune"
        cls.work = cls.tmp_path / "work"
        cls.work.mkdir()
        cls.ckpt = cls.work / "model.pt"
        cls.ckpt.write_text("checkpoint", encoding="utf-8")
        cls.train = cls.work / "train.extxyz"
        cls.train.write_text("1\n", encoding="utf-8")
        cls.val = cls.work / "val.extxyz"
        cls.val.write_text("1\n", encoding="utf-8")
        cls.replay = cls.work / "replay.extxyz"
        cls.replay.write_text("1\n", encoding="utf-8")
        cls.fisher = cls.work / "fisher.pt"
        cls.fisher.write_text("fisher", encoding="utf-8")
        cls.reference = cls.work / "reference.pt"
        cls.reference.write_text("reference", encoding="utf-8")
        cls._write_fake_mattertune()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    @classmethod
    def _write_fake_mattertune(cls) -> None:
        for package in [
            "",
            "finetune",
            "data",
            "backbones",
            "backbones/deepmd",
            "backbones/sevennet",
            "backbones/chgnet",
            "backbones/mace_foundation",
            "recipes",
        ]:
            write(cls.fake_mattertune / package / "__init__.py")

        write(
            cls.fake_mattertune / "fake_config.py",
            """
            class ConfigObject:
                name = None

                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
                    class_name = self.__class__.__dict__.get("name")
                    if class_name is not None:
                        self.name = class_name

                def model_dump(self, **_kwargs):
                    def dump(value):
                        if hasattr(value, "model_dump"):
                            return value.model_dump()
                        if isinstance(value, list):
                            return [dump(item) for item in value]
                        if isinstance(value, tuple):
                            return [dump(item) for item in value]
                        if isinstance(value, dict):
                            return {key: dump(item) for key, item in value.items()}
                        return value
                    payload = dict(self.__dict__)
                    if self.name is not None:
                        payload["name"] = self.name
                    return {key: dump(value) for key, value in payload.items() if value is not None}
            """,
        )
        write(
            cls.fake_mattertune / "finetune/loss.py",
            "from mattertune.fake_config import ConfigObject\nclass MAELossConfig(ConfigObject):\n    name = 'mae'\n",
        )
        write(
            cls.fake_mattertune / "finetune/optimizer.py",
            "from mattertune.fake_config import ConfigObject\nclass AdamWConfig(ConfigObject):\n    name = 'AdamW'\n",
        )
        write(
            cls.fake_mattertune / "finetune/properties.py",
            """
            from mattertune.fake_config import ConfigObject

            class EnergyPropertyConfig(ConfigObject):
                def __init__(self, **kwargs):
                    super().__init__(type="energy", name="energy", **kwargs)

            class ForcesPropertyConfig(ConfigObject):
                def __init__(self, **kwargs):
                    super().__init__(type="forces", name="forces", **kwargs)

            class StressesPropertyConfig(ConfigObject):
                def __init__(self, **kwargs):
                    super().__init__(type="stresses", name="stresses", **kwargs)

            class MagneticMomentsPropertyConfig(ConfigObject):
                def __init__(self, **kwargs):
                    super().__init__(type="magnetic_moments", name="magnetic_moments", **kwargs)
            """,
        )
        write(
            cls.fake_mattertune / "data/xyz.py",
            "from mattertune.fake_config import ConfigObject\nclass XYZDatasetConfig(ConfigObject):\n    type = 'xyz'\n",
        )
        write(
            cls.fake_mattertune / "data/datamodule.py",
            "from mattertune.fake_config import ConfigObject\nclass ManualSplitDataModuleConfig(ConfigObject):\n    pass\n",
        )
        write(
            cls.fake_mattertune / "main.py",
            """
            from types import SimpleNamespace
            from mattertune.fake_config import ConfigObject

            class TrainerConfig(ConfigObject):
                pass

            class ModelCheckpointConfig(ConfigObject):
                name = 'model_checkpoint'

            class MatterTunerConfig(ConfigObject):
                pass

            class _FakeModel:
                def execution_metadata(self, trainer):
                    return {"fake_tune_called": True, "trainer": repr(trainer)}

            class MatterTuner:
                def __init__(self, config):
                    self.config = config

                def tune(self):
                    return SimpleNamespace(model=_FakeModel(), trainer=SimpleNamespace(global_step=1))
            """,
        )
        write(
            cls.fake_mattertune / "backbones/deepmd/model.py",
            "from mattertune.fake_config import ConfigObject\nclass DeepMDBackboneConfig(ConfigObject):\n    name = 'deepmd'\n    def capabilities(self):\n        return {'backbone': 'deepmd', 'full_fine_tuning': True}\n    def parity_status(self):\n        return {'backbone': 'deepmd', 'P0': 'fake'}\n",
        )
        write(
            cls.fake_mattertune / "backbones/deepmd/multihead.py",
            """
            from mattertune.fake_config import ConfigObject
            class DeepMDHeadConfig(ConfigObject):
                pass
            def multihead_capabilities():
                return {'backbone': 'deepmd', 'multi_head': True, 'multi_domain_training': True, 'head_only_fine_tuning': True}
            def multihead_parity_status():
                return {'backbone': 'deepmd', 'P0': 'fake multihead'}
            """,
        )
        write(
            cls.fake_mattertune / "backbones/deepmd/multihead_module.py",
            """
            from mattertune.fake_config import ConfigObject
            class DeepMDDomainConfig(ConfigObject):
                pass
            class DeepMDMultiHeadBackboneConfig(ConfigObject):
                name = 'deepmd_multihead'
                def capabilities(self):
                    capabilities = {}
                    capabilities['multi_domain_training'] = True
                    return capabilities
                def parity_status(self):
                    return {'backbone': 'deepmd_multihead'}
            """,
        )
        write(
            cls.fake_mattertune / "backbones/sevennet/model.py",
            "from mattertune.fake_config import ConfigObject\nclass SevenNetBackboneConfig(ConfigObject):\n    name = 'sevennet'\n    def capabilities(self):\n        return {'backbone': 'sevennet', 'full_fine_tuning': True}\n    def parity_status(self):\n        return {'backbone': 'sevennet', 'P0': 'fake'}\n",
        )
        write(
            cls.fake_mattertune / "backbones/sevennet/continual.py",
            "def continual_capabilities():\n    return {'backbone': 'sevennet', 'replay': True, 'ewc': True, 'replay_plus_ewc': True}\ndef continual_parity_status():\n    return {'backbone': 'sevennet', 'component': 'continual'}\n",
        )
        write(
            cls.fake_mattertune / "backbones/chgnet/model.py",
            "from mattertune.fake_config import ConfigObject\nclass CHGNetBackboneConfig(ConfigObject):\n    name = 'chgnet'\n    def capabilities(self):\n        return {'backbone': 'chgnet', 'full_fine_tuning': True, 'magnetic_moments': True, 'signed_magnetic_moments': False}\n    def parity_status(self):\n        return {'backbone': 'chgnet', 'P0': 'fake'}\n",
        )
        write(
            cls.fake_mattertune / "backbones/mace_foundation/model.py",
            "from mattertune.fake_config import ConfigObject\nclass MACEBackboneConfig(ConfigObject):\n    name = 'mace'\n    def capabilities(self):\n        return {'backbone': 'mace', 'single_head': True, 'multi_head': True, 'lora': True, 'full_fine_tuning': True}\n    def parity_status(self):\n        return {'backbone': 'mace', 'native_protocol_partial': True, 'synthetic_verified': True, 'tier_c_verified': False}\n",
        )
        write(
            cls.fake_mattertune / "recipes/lora.py",
            "from mattertune.fake_config import ConfigObject\nclass LoraConfig(ConfigObject):\n    pass\nclass LoRARecipeConfig(ConfigObject):\n    pass\n",
        )
        write(
            cls.fake_mattertune / "recipes/ema.py",
            "from mattertune.fake_config import ConfigObject\nclass EMARecipeConfig(ConfigObject):\n    name = 'ema'\n",
        )

    def run_adapter(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.fake_root}:{ROOT}:{env.get('PYTHONPATH', '')}"
        env["MATTERTUNE_SOURCE_ROOT"] = str(self.fake_mattertune)
        return subprocess.run(
            [sys.executable, str(ROOT / "tools" / "mattertune_train.py"), *args],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def common(self, run_name: str, *, dry_run: bool = True, val_data: bool = False) -> list[str]:
        args = [
            "--checkpoint",
            str(self.ckpt),
            "--train-data",
            str(self.train),
            "--run-dir",
            f"playground/runs/test-matrix-{run_name}",
            "--learning-rate",
            "1e-4",
        ]
        if val_data:
            args.extend(["--val-data", str(self.val)])
        if dry_run:
            args.append("--dry-run")
        args.append("--json")
        return args

    def assert_plan(
        self, proc: subprocess.CompletedProcess[str], *, model: str, backbone: str, required: set[str]
    ) -> dict[str, object]:
        self.assertEqual(proc.returncode, 0, proc.stderr)
        plan = json.loads(proc.stdout)
        self.assertEqual(plan["model_type"], model)
        self.assertEqual(plan["mattertune_backbone_name"], backbone)
        self.assertTrue(plan["mattertune_config_valid"])
        self.assertTrue(str(plan["capability_source"]).startswith("mattertune_source:"))
        self.assertTrue(required.issubset(set(plan["capabilities_required"])))
        return plan

    def test_success_feature_matrix(self) -> None:
        cases = [
            (
                "deepmd-single",
                ["deepmd", "--mode", "single", *self.common("deepmd-single"), "--properties", "energy,forces"],
                "deepmd",
                "deepmd",
                {"full_fine_tuning"},
            ),
            (
                "deepmd-multihead",
                [
                    "deepmd",
                    "--mode",
                    "multihead",
                    *self.common("deepmd-multihead"),
                    "--properties",
                    "energy,forces,stresses",
                    "--resume-head",
                    "MP_traj",
                    "--copy-head",
                    "new_task=MP_traj",
                    "--random-head",
                    "task_x",
                    "--head-seed",
                    "task_x=42",
                    "--private-fitting-head",
                    "task_x",
                    "--domain",
                    "catalyst=MP_traj",
                    "--domain",
                    "polymer=new_task",
                    "--sampling-weight",
                    "catalyst=0.5",
                    "--loss-weight",
                    "polymer=2.0",
                    "--trainable-scope",
                    "head_only",
                ],
                "deepmd",
                "deepmd_multihead",
                {"multi_head", "multi_domain_training"},
            ),
            (
                "sevennet-none",
                [
                    "sevennet",
                    "--continual-mode",
                    "none",
                    *self.common("sevennet-none"),
                    "--properties",
                    "energy,forces,stresses",
                ],
                "sevennet",
                "sevennet",
                set(),
            ),
            (
                "sevennet-replay-ewc",
                [
                    "sevennet",
                    "--continual-mode",
                    "replay-ewc",
                    *self.common("sevennet-replay-ewc"),
                    "--properties",
                    "energy,forces,stresses",
                    "--replay-data",
                    str(self.replay),
                    "--replay-batch-size",
                    "2",
                    "--replay-ratio",
                    "0.25",
                    "--fisher",
                    str(self.fisher),
                    "--reference",
                    str(self.reference),
                    "--lambda-ewc",
                    "10.0",
                ],
                "sevennet",
                "sevennet",
                {"replay", "ewc", "replay_plus_ewc"},
            ),
            (
                "chgnet-efsm",
                [
                    "chgnet",
                    "--objective",
                    "efsm",
                    "--atomref-mode",
                    "joint",
                    "--energy-basis",
                    "per_atom",
                    *self.common("chgnet-efsm"),
                    "--properties",
                    "energy,forces,stresses,magnetic_moments",
                ],
                "chgnet",
                "chgnet",
                {"full_fine_tuning"},
            ),
            (
                "mace-multihead-lora",
                [
                    "mace",
                    "--trust-checkpoint",
                    "--head-mode",
                    "multi_head",
                    "--head",
                    "MP",
                    "--lora",
                    "--lora-rank",
                    "8",
                    "--lora-alpha",
                    "1",
                    *self.common("mace-multihead-lora"),
                    "--properties",
                    "energy,forces",
                ],
                "mace",
                "mace",
                {"multi_head", "lora"},
            ),
        ]
        for name, args, model, backbone, required in cases:
            with self.subTest(name=name):
                self.assert_plan(self.run_adapter(*args), model=model, backbone=backbone, required=required)

    def test_mace_multihead_plan_passes_selected_head_to_mattertune_config(self) -> None:
        proc = self.run_adapter(
            "mace",
            "--trust-checkpoint",
            "--head-mode",
            "multi_head",
            "--head",
            "pbe_d3",
            *self.common("mace-selected-head"),
            "--properties",
            "energy,forces",
            "--energy-key",
            "dft_energy",
            "--forces-key",
            "dft_forces",
        )
        plan = self.assert_plan(proc, model="mace", backbone="mace", required={"multi_head"})
        self.assertEqual(plan["protocol"]["head"], "pbe_d3")
        self.assertEqual(plan["config_preview"]["model"]["head"], "pbe_d3")
        self.assertEqual(plan["config_preview"]["data"]["train"]["energy_key"], "dft_energy")
        self.assertEqual(plan["config_preview"]["data"]["train"]["forces_key"], "dft_forces")

    def test_mace_tier_d_plan_exposes_loss_weights_and_trainer_controls(self) -> None:
        proc = self.run_adapter(
            "mace",
            "--trust-checkpoint",
            "--head-mode",
            "single_head",
            *self.common("mace-tier-d"),
            "--properties",
            "energy,forces,stresses",
            "--energy-weight",
            "1",
            "--forces-weight",
            "10",
            "--stress-weight",
            "100",
            "--max-epochs",
            "2500",
            "--devices",
            "0,1",
            "--strategy",
            "ddp",
            "--ema",
            "--ema-decay",
            "0.995",
        )
        plan = self.assert_plan(proc, model="mace", backbone="mace", required={"single_head"})
        self.assertEqual(plan["protocol"]["loss_weights"], {"energy": 1.0, "forces": 10.0, "stresses": 100.0})
        preview = plan["config_preview"]
        properties = {item["name"]: item for item in preview["model"]["properties"]}
        self.assertEqual(properties["energy"]["loss_coefficient"], 1.0)
        self.assertEqual(properties["forces"]["loss_coefficient"], 10.0)
        self.assertEqual(properties["stresses"]["loss_coefficient"], 100.0)
        self.assertEqual(preview["trainer"]["max_epochs"], 2500)
        self.assertEqual(preview["trainer"]["devices"], [0, 1])
        self.assertEqual(preview["trainer"]["strategy"], "ddp")
        checkpoint = preview["trainer"]["checkpoint"]
        self.assertEqual(checkpoint["dirpath"], "playground/runs/test-matrix-mace-tier-d/checkpoints")
        self.assertEqual(checkpoint["filename"], "periodic-epoch={epoch:04d}-step={step}")
        self.assertEqual(checkpoint["every_n_epochs"], 100)
        self.assertEqual(checkpoint["save_last"], True)
        self.assertEqual(checkpoint["save_top_k"], -1)
        self.assertNotIn("monitor", checkpoint)
        self.assertEqual(preview["trainer"]["checkpoints"], [])
        self.assertEqual(preview["recipes"][0]["name"], "ema")
        self.assertEqual(preview["recipes"][0]["decay"], 0.995)

    def test_validation_split_adds_best_checkpoint_callback(self) -> None:
        proc = self.run_adapter(
            "mace",
            "--trust-checkpoint",
            "--head-mode",
            "single_head",
            *self.common("mace-tier-d-val", val_data=True),
            "--properties",
            "energy,forces",
            "--checkpoint-every-n-epochs",
            "25",
        )
        plan = self.assert_plan(proc, model="mace", backbone="mace", required={"single_head"})
        trainer = plan["config_preview"]["trainer"]
        self.assertEqual(trainer["checkpoint"]["filename"], "periodic-epoch={epoch:04d}-step={step}")
        self.assertEqual(trainer["checkpoint"]["every_n_epochs"], 25)
        self.assertNotIn("monitor", trainer["checkpoint"])
        self.assertEqual(len(trainer["checkpoints"]), 1)
        best = trainer["checkpoints"][0]
        self.assertEqual(best["filename"], "best-val_loss-epoch={epoch:04d}-step={step}")
        self.assertEqual(best["monitor"], "val_loss")
        self.assertEqual(best["mode"], "min")
        self.assertEqual(best["save_top_k"], 1)
        self.assertEqual(best["every_n_epochs"], 1)

    def test_mace_single_head_non_dry_run_calls_mattertuner(self) -> None:
        manifest = self.tmp_path / "run_manifest.json"
        proc = self.run_adapter(
            "mace",
            "--trust-checkpoint",
            "--head-mode",
            "single_head",
            *self.common("mace-execute", dry_run=False),
            "--properties",
            "energy,forces",
            "--max-steps",
            "1",
            "--manifest",
            str(manifest),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("training_status=completed", proc.stdout)
        run_manifest = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(run_manifest["training_status"], "completed")
        self.assertEqual(run_manifest["execution_metadata"]["fake_tune_called"], True)

    def test_mace_multihead_non_dry_run_is_refused_until_native_parity(self) -> None:
        proc = self.run_adapter(
            "mace",
            "--trust-checkpoint",
            "--head-mode",
            "multi_head",
            "--head",
            "mof0",
            *self.common("mace-multihead-execute", dry_run=False),
            "--properties",
            "energy,forces",
            "--max-steps",
            "1",
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("native MACE multi-head replay is not Tier D verified", proc.stderr)

    def test_wrong_model_flags_are_rejected(self) -> None:
        cases = [
            [
                "mace",
                "--trust-checkpoint",
                "--head-mode",
                "single_head",
                *self.common("bad-mace-replay"),
                "--properties",
                "energy,forces",
                "--replay-data",
                str(self.replay),
            ],
            [
                "sevennet",
                "--continual-mode",
                "none",
                *self.common("bad-sevennet-lora"),
                "--properties",
                "energy,forces",
                "--lora",
            ],
            [
                "chgnet",
                "--objective",
                "ef",
                "--atomref-mode",
                "preserve",
                "--energy-basis",
                "per_atom",
                *self.common("bad-chgnet-head"),
                "--properties",
                "energy,forces",
                "--head",
                "MP",
            ],
            ["deepmd", "--mode", "single", *self.common("bad-deepmd-lora"), "--properties", "energy,forces", "--lora"],
        ]
        for args in cases:
            with self.subTest(args=args[0]):
                proc = self.run_adapter(*args)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("unrecognized arguments", proc.stderr)

    def test_protocol_refusals_are_rejected_before_plan(self) -> None:
        cases = [
            (
                [
                    "mace",
                    "--trust-checkpoint",
                    "--head-mode",
                    "multi_head",
                    *self.common("bad-mace-head"),
                    "--properties",
                    "energy,forces",
                ],
                "requires --head",
            ),
            (
                [
                    "mace",
                    "--trust-checkpoint",
                    "--head-mode",
                    "single_head",
                    "--lora",
                    "--lora-rank",
                    "8",
                    *self.common("bad-mace-lora"),
                    "--properties",
                    "energy,forces",
                ],
                "--lora requires",
            ),
            (
                [
                    "sevennet",
                    "--continual-mode",
                    "replay",
                    *self.common("bad-sevennet-replay"),
                    "--properties",
                    "energy,forces",
                ],
                "requires",
            ),
            (
                [
                    "chgnet",
                    "--objective",
                    "ef",
                    "--atomref-mode",
                    "preserve",
                    "--energy-basis",
                    "per_atom",
                    *self.common("bad-chgnet-props"),
                    "--properties",
                    "energy,forces,stresses",
                ],
                "--properties must match",
            ),
            (
                [
                    "deepmd",
                    "--mode",
                    "multihead",
                    *self.common("bad-deepmd-head"),
                    "--properties",
                    "energy,forces",
                    "--random-head",
                    "task_x",
                    "--private-fitting-head",
                    "task_x",
                    "--domain",
                    "task=task_x",
                ],
                "requires --head-seed",
            ),
        ]
        for args, expected in cases:
            with self.subTest(expected=expected):
                proc = self.run_adapter(*args)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn(expected, proc.stderr)


if __name__ == "__main__":
    unittest.main()
