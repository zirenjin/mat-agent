from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_adapter():
    spec = importlib.util.spec_from_file_location("mattertune_train", ROOT / "tools" / "mattertune_train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mattertune_train"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MatterTuneCapabilityExtractionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.adapter = load_adapter()

    def test_deepmd_multihead_capabilities_are_source_derived(self) -> None:
        caps, source, errors = self.adapter.mattertune_source_capabilities("deepmd_multihead")
        self.assertIn("backbones/deepmd/multihead.py", source)
        self.assertIn("backbones/deepmd/multihead_module.py", source)
        self.assertTrue(caps["multi_head"])
        self.assertTrue(caps["multi_domain_training"])
        self.assertEqual(errors, [])

    def test_sevennet_continual_capabilities_are_merged_from_source(self) -> None:
        caps, source, errors = self.adapter.mattertune_source_capabilities("sevennet")
        self.assertIn("backbones/sevennet/model.py", source)
        self.assertIn("backbones/sevennet/continual.py", source)
        self.assertTrue(caps["full_fine_tuning"])
        self.assertTrue(caps["replay"])
        self.assertTrue(caps["ewc"])
        self.assertTrue(caps["replay_plus_ewc"])
        self.assertTrue(any("sevenn_version_pinned" in error for error in errors))

    def test_mace_capabilities_are_source_derived_but_tier_c_stays_false(self) -> None:
        caps, source, errors = self.adapter.mattertune_source_capabilities("mace")
        self.assertIn("backbones/mace_foundation/model.py", source)
        self.assertTrue(caps["full_fine_tuning"])
        self.assertTrue(caps["single_head"])
        self.assertTrue(caps["multi_head"])
        self.assertTrue(caps["lora"])
        self.assertEqual(errors, [])

        parity, _, _ = self.adapter.mattertune_source_parity("mace")
        self.assertTrue(parity["native_protocol_partial"])
        self.assertTrue(parity["synthetic_verified"])
        self.assertFalse(parity["tier_c_verified"])


if __name__ == "__main__":
    unittest.main()
