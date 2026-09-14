"""Build the submitted model as an ASE calculator. Runs inside the sandbox child.

Two submission kinds:

* ``mace_calculator`` -- load ``checkpoint`` with ``mace.calculators.MACECalculator``
  (dtype/head from the descriptor). Covers any model trained with native MACE.
* ``entrypoint`` -- import exactly the declared ``workspace/<file>.py:function``
  and call it; it must return an object with the ASE calculator interface
  (``get_potential_energy`` / ``get_forces`` / ``get_stress``). Covers a genuinely
  new architecture the agent wrote itself.

Nothing here constrains the model. The sandbox constrains only what the code may
touch (sequestered data, benchmark files, network).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]


def _load_entrypoint(spec: str) -> Any:
    mod_rel, func_name = spec.split(":")
    mod_path = (REPO / mod_rel).resolve()
    mod_path.relative_to(REPO / "workspace")  # raises if outside workspace/
    module_spec = importlib.util.spec_from_file_location(f"_candidate_{mod_path.stem}", mod_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)  # type: ignore[union-attr]
    if not hasattr(module, func_name):
        raise AttributeError(f"{mod_rel} has no attribute {func_name}")
    return getattr(module, func_name)


def _looks_like_calculator(obj: Any) -> bool:
    return all(hasattr(obj, m) for m in ("get_potential_energy", "get_forces"))


def build_calculator(descriptor: dict) -> Any:
    """descriptor = {kind, checkpoint(abs str), dtype, head?, entrypoint?}."""
    kind = descriptor["kind"]
    if kind == "mace_calculator":
        from mace.calculators import MACECalculator

        kwargs: dict[str, Any] = {"model_paths": descriptor["checkpoint"], "device": descriptor.get("device", "cpu")}
        if descriptor.get("dtype"):
            kwargs["default_dtype"] = descriptor["dtype"]
        if descriptor.get("head"):
            kwargs["head"] = descriptor["head"]
        return MACECalculator(**kwargs)

    if kind == "entrypoint":
        factory = _load_entrypoint(descriptor["entrypoint"])
        calc = factory(
            checkpoint=descriptor["checkpoint"],
            dtype=descriptor.get("dtype", "float64"),
            head=descriptor.get("head"),
            device=descriptor.get("device", "cpu"),
        )
        if not _looks_like_calculator(calc):
            raise TypeError("entrypoint did not return an ASE-style calculator")
        return calc

    raise ValueError(f"unknown inference kind: {kind!r}")
