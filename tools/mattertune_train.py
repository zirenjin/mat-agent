#!/usr/bin/env python3
"""Bash-first dry-run adapter from mat-agent CLI flags to MatterTune configs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MATTERTUNE_SOURCE_ROOT = Path(
    os.environ.get(
        "MATTERTUNE_SOURCE_ROOT",
        Path(__file__).resolve().parents[1] / "mattertune" / "src" / "mattertune",
    )
)

CAPABILITY_SOURCES: dict[str, tuple[str, ...]] = {
    "deepmd": ("backbones/deepmd/model.py",),
    "deepmd_multihead": (
        "backbones/deepmd/multihead.py",
        "backbones/deepmd/multihead_module.py",
    ),
    "sevennet": (
        "backbones/sevennet/model.py",
        "backbones/sevennet/continual.py",
    ),
    "chgnet": ("backbones/chgnet/model.py",),
    "mace": ("backbones/mace_foundation/model.py",),
}

SUPPORTED_BASH_MODELS = {"deepmd", "sevennet", "chgnet", "mace"}

SUPPORTED_PROPERTIES = {"energy", "forces", "stress", "stresses", "magmoms", "magnetic_moments"}


def dump_config_object(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return repr(obj)


def literal_dict_from_ast(node: ast.Dict) -> tuple[dict[str, Any], list[str]]:
    values: dict[str, Any] = {}
    errors: list[str] = []
    for key_node, value_node in zip(node.keys, node.values, strict=True):
        if key_node is None:
            errors.append("dict unpacking is not extractable")
            continue
        try:
            key = ast.literal_eval(key_node)
        except (ValueError, SyntaxError):
            errors.append("non-literal capability key is not extractable")
            continue
        if not isinstance(key, str):
            errors.append(f"non-string capability key is not extractable: {key!r}")
            continue
        try:
            values[key] = ast.literal_eval(value_node)
        except (ValueError, SyntaxError):
            errors.append(f"dynamic value for {key}")
    return values, errors


def literal_dict_from_return(func: ast.FunctionDef) -> tuple[dict[str, Any], list[str]] | None:
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            return literal_dict_from_ast(node.value)
    return None


def find_function(module: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def source_literal_capabilities(source_path: Path) -> tuple[dict[str, Any], list[str]]:
    text = source_path.read_text(encoding="utf-8")
    module = ast.parse(text, filename=str(source_path))
    capabilities: dict[str, Any] = {}
    errors: list[str] = []

    for function_name in ("multihead_capabilities", "continual_capabilities"):
        func = find_function(module, function_name)
        if func is None:
            continue
        literal = literal_dict_from_return(func)
        if literal is None:
            errors.append(f"{source_path}:{function_name} has no literal return dict")
        else:
            extracted, literal_errors = literal
            capabilities.update(extracted)
            errors.extend(f"{source_path}:{function_name} {error}" for error in literal_errors)

    for class_node in (node for node in ast.walk(module) if isinstance(node, ast.ClassDef)):
        func = next(
            (item for item in class_node.body if isinstance(item, ast.FunctionDef) and item.name == "capabilities"),
            None,
        )
        if func is None:
            continue
        local_caps: dict[str, Any] = {}
        literal = literal_dict_from_return(func)
        if literal is not None:
            extracted, literal_errors = literal
            local_caps.update(extracted)
            errors.extend(f"{source_path}:{class_node.name}.capabilities {error}" for error in literal_errors)
        for stmt in func.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Subscript)
                and isinstance(stmt.targets[0].value, ast.Name)
                and stmt.targets[0].value.id == "capabilities"
                and isinstance(stmt.targets[0].slice, ast.Constant)
                and isinstance(stmt.targets[0].slice.value, str)
            ):
                try:
                    local_caps[stmt.targets[0].slice.value] = ast.literal_eval(stmt.value)
                except (ValueError, SyntaxError):
                    errors.append(
                        f"{source_path}:{class_node.name}.capabilities dynamic value for {stmt.targets[0].slice.value}"
                    )
        if local_caps:
            capabilities.update(local_caps)

    return capabilities, errors


def mattertune_source_capabilities(backbone_name: str) -> tuple[dict[str, Any], str, list[str]]:
    sources = CAPABILITY_SOURCES.get(backbone_name)
    if not sources:
        return {}, "unsupported_bash_adapter_backbone", [f"no capability source mapping for {backbone_name}"]
    capabilities: dict[str, Any] = {}
    errors: list[str] = []
    read_sources: list[str] = []
    for source in sources:
        path = MATTERTUNE_SOURCE_ROOT / source
        if not path.is_file():
            errors.append(f"missing MatterTune source file: {source}")
            continue
        extracted, source_errors = source_literal_capabilities(path)
        capabilities.update(extracted)
        errors.extend(source_errors)
        read_sources.append(source)
    if capabilities:
        return capabilities, "mattertune_source:" + ",".join(read_sources), errors
    return {}, "mattertune_missing_capabilities", errors or [f"no capabilities() metadata found for {backbone_name}"]


def source_literal_parity(source_path: Path) -> tuple[dict[str, Any], list[str]]:
    text = source_path.read_text(encoding="utf-8")
    module = ast.parse(text, filename=str(source_path))
    parity: dict[str, Any] = {}
    errors: list[str] = []

    for function_name in ("multihead_parity_status", "continual_parity_status"):
        func = find_function(module, function_name)
        if func is None:
            continue
        literal = literal_dict_from_return(func)
        if literal is None:
            errors.append(f"{source_path}:{function_name} has no literal return dict")
        else:
            extracted, literal_errors = literal
            parity.update(extracted)
            errors.extend(f"{source_path}:{function_name} {error}" for error in literal_errors)

    for class_node in (node for node in ast.walk(module) if isinstance(node, ast.ClassDef)):
        func = next(
            (item for item in class_node.body if isinstance(item, ast.FunctionDef) and item.name == "parity_status"),
            None,
        )
        if func is None:
            continue
        literal = literal_dict_from_return(func)
        if literal is None:
            errors.append(f"{source_path}:{class_node.name}.parity_status has no literal return dict")
        else:
            extracted, literal_errors = literal
            parity.update(extracted)
            errors.extend(f"{source_path}:{class_node.name}.parity_status {error}" for error in literal_errors)

    return parity, errors


def mattertune_source_parity(backbone_name: str) -> tuple[dict[str, Any], str, list[str]]:
    sources = CAPABILITY_SOURCES.get(backbone_name)
    if not sources:
        return {}, "unsupported_bash_adapter_backbone", [f"no parity source mapping for {backbone_name}"]
    parity: dict[str, Any] = {}
    errors: list[str] = []
    read_sources: list[str] = []
    for source in sources:
        path = MATTERTUNE_SOURCE_ROOT / source
        if not path.is_file():
            errors.append(f"missing MatterTune source file: {source}")
            continue
        extracted, source_errors = source_literal_parity(path)
        parity.update(extracted)
        errors.extend(source_errors)
        read_sources.append(source)
    if parity:
        return parity, "mattertune_source:" + ",".join(read_sources), errors
    return {}, "mattertune_missing_parity", errors or [f"no parity_status() metadata found for {backbone_name}"]


def preflight_capabilities(backbone_name: str, required: list[str]) -> tuple[dict[str, Any], str, list[str]]:
    capabilities, source, errors = mattertune_source_capabilities(backbone_name)
    if not capabilities:
        if required:
            errors.append(f"required capabilities were not checked because {backbone_name} has no metadata")
        return capabilities, source, errors
    missing = [name for name in required if capabilities.get(name) in (None, False)]
    if missing:
        raise SystemExit(
            f"required capabilities not supported by {backbone_name}: {missing}; capability_source={source}"
        )
    return capabilities, source, errors


@dataclass
class Plan:
    model_type: str
    checkpoint_identity: str
    requested_properties: list[str]
    adaptation_mode: str
    capabilities_required: list[str]
    trainable_scope: str
    data_paths: dict[str, str | None]
    run_dir: str
    resume_mode: str
    trusted_checkpoint_requirement: str
    capabilities: dict[str, Any]
    mattertune_backbone_name: str
    capability_source: str
    capability_errors: list[str]
    parity_status: dict[str, Any]
    parity_source: str
    parity_errors: list[str]
    mattertune_config_valid: bool
    mattertune_config_class: str
    output_format: str
    config_preview: dict[str, Any]
    protocol: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "checkpoint_identity": self.checkpoint_identity,
            "requested_properties": self.requested_properties,
            "adaptation_mode": self.adaptation_mode,
            "capabilities_required": self.capabilities_required,
            "trainable_scope": self.trainable_scope,
            "data_paths": self.data_paths,
            "run_dir": self.run_dir,
            "resume_mode": self.resume_mode,
            "trusted_checkpoint_requirement": self.trusted_checkpoint_requirement,
            "capabilities": self.capabilities,
            "mattertune_backbone_name": self.mattertune_backbone_name,
            "capability_source": self.capability_source,
            "capability_errors": self.capability_errors,
            "parity_status": self.parity_status,
            "parity_source": self.parity_source,
            "parity_errors": self.parity_errors,
            "mattertune_config_valid": self.mattertune_config_valid,
            "mattertune_config_class": self.mattertune_config_class,
            "config_preview": self.config_preview,
            "protocol": self.protocol,
        }


def int_or_auto(value: str) -> int | str:
    if value == "auto":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected integer or auto") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("num-workers must be >= 0 or auto")
    return parsed


def parse_devices(value: str) -> int | str | list[int]:
    if value in {"auto", "all"}:
        return value
    if "," in value:
        try:
            devices = [int(part.strip()) for part in value.split(",") if part.strip()]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "devices must be auto, all, an integer, or comma-separated integers"
            ) from exc
        if not devices:
            raise argparse.ArgumentTypeError("devices list must not be empty")
        return devices
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("devices must be auto, all, an integer, or comma-separated integers") from exc


def parse_csv(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    normalized = []
    for item in values:
        if item == "stress":
            item = "stresses"
        if item == "magmoms":
            item = "magnetic_moments"
        normalized.append(item)
    invalid = sorted(set(normalized) - SUPPORTED_PROPERTIES)
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported properties: {invalid}")
    return normalized


def parse_key_values(items: list[str] | None, *, name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"{name} expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise SystemExit(f"{name} expects non-empty KEY=VALUE, got {item!r}")
        result[key] = value
    return result


def parse_set(items: list[str] | None) -> set[str]:
    return {item.strip() for item in items or [] if item.strip()}


def file_identity(path_value: str, label: str) -> str:
    path = Path(path_value)
    if not path.is_file():
        raise SystemExit(f"{label} path is not a readable local file: {path_value}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def trusted_name_identity(name: str) -> str:
    return f"trusted-name:{name}"


def require_existing_file(path_value: str | None, label: str) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_file():
        raise SystemExit(f"{label} path is not a readable file: {path_value}")
    return str(path)


def require_sandbox_run_dir(run_dir_value: str) -> str:
    path = Path(run_dir_value)
    parts = path.parts
    allowed = (
        path.is_relative_to("playground/runs") if not path.is_absolute() else "playground" in parts and "runs" in parts
    )
    if not allowed:
        raise SystemExit("--run-dir must be inside playground/runs")
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def property_configs(
    properties: list[str],
    *,
    energy_basis: str = "total",
    loss_coefficients: dict[str, float] | None = None,
) -> list[Any]:
    from mattertune.finetune.loss import MAELossConfig
    from mattertune.finetune.properties import (
        EnergyPropertyConfig,
        ForcesPropertyConfig,
        MagneticMomentsPropertyConfig,
        StressesPropertyConfig,
    )

    coefficients = loss_coefficients or {}
    configs: list[Any] = []
    for prop in properties:
        coefficient = float(coefficients.get(prop, 1.0))
        if prop == "energy":
            configs.append(
                EnergyPropertyConfig(loss=MAELossConfig(), loss_basis=energy_basis, loss_coefficient=coefficient)
            )
        elif prop == "forces":
            configs.append(ForcesPropertyConfig(loss=MAELossConfig(), conservative=True, loss_coefficient=coefficient))
        elif prop == "stresses":
            configs.append(
                StressesPropertyConfig(loss=MAELossConfig(), conservative=True, loss_coefficient=coefficient)
            )
        elif prop == "magnetic_moments":
            configs.append(
                MagneticMomentsPropertyConfig(
                    loss=MAELossConfig(),
                    sign_convention="unsigned_magnitude",
                    loss_coefficient=coefficient,
                )
            )
        else:
            raise SystemExit(f"unsupported property: {prop}")
    return configs


def properties_for_chgnet_objective(objective: str, cli_properties: list[str]) -> list[str]:
    objective_props = {
        "ef": ["energy", "forces"],
        "efs": ["energy", "forces", "stresses"],
        "efm": ["energy", "forces", "magnetic_moments"],
        "efsm": ["energy", "forces", "stresses", "magnetic_moments"],
    }[objective]
    if cli_properties != objective_props:
        raise SystemExit(
            f"--properties must match CHGNet --objective {objective}: "
            f"expected {','.join(objective_props)}, got {','.join(cli_properties)}"
        )
    return objective_props


def common_config(args: argparse.Namespace, model_config: Any, recipes: list[Any] | None = None) -> Any:
    from mattertune.data.datamodule import ManualSplitDataModuleConfig
    from mattertune.data.xyz import XYZDatasetConfig
    from mattertune.main import MatterTunerConfig, ModelCheckpointConfig, TrainerConfig

    recipes = list(recipes or [])
    if getattr(args, "ema", False):
        from mattertune.recipes.ema import EMARecipeConfig

        recipes.append(EMARecipeConfig(decay=args.ema_decay))

    xyz_kwargs = {
        "energy_key": args.energy_key,
        "forces_key": args.forces_key,
        "stress_key": args.stress_key,
    }
    validation = XYZDatasetConfig(src=args.val_data, **xyz_kwargs) if args.val_data else None
    checkpoint_dir = str(Path(args.run_dir) / "checkpoints")
    periodic_checkpoint = ModelCheckpointConfig(
        dirpath=checkpoint_dir,
        filename="periodic-epoch={epoch:04d}-step={step}",
        save_last=True,
        save_top_k=-1,
        every_n_epochs=args.checkpoint_every_n_epochs,
        save_on_train_epoch_end=True,
    )
    best_checkpoints = [
        ModelCheckpointConfig(
            dirpath=checkpoint_dir,
            filename="best-val_loss-epoch={epoch:04d}-step={step}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        )
    ] if args.val_data else []
    trainer_kwargs = {"inference_mode": False}
    data = ManualSplitDataModuleConfig(
        train=XYZDatasetConfig(src=args.train_data, **xyz_kwargs),
        validation=validation,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    trainer = TrainerConfig(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        deterministic=True,
        checkpoint=periodic_checkpoint,
        checkpoints=best_checkpoints,
        additional_trainer_kwargs=trainer_kwargs,
    )
    return MatterTunerConfig(model=model_config, data=data, trainer=trainer, recipes=recipes)


def build_deepmd(args: argparse.Namespace, properties: list[str]) -> tuple[Any, str, str, list[str], dict[str, Any]]:
    from mattertune.finetune.optimizer import AdamWConfig

    if args.mode == "single":
        from mattertune.backbones.deepmd.model import DeepMDBackboneConfig

        model = DeepMDBackboneConfig(
            checkpoint_path=args.checkpoint,
            model_branch=args.branch,
            properties=property_configs(properties),
            optimizer=AdamWConfig(lr=args.learning_rate),
        )
        return model, "single_branch", "full", ["full_fine_tuning"], {"mode": "single", "branch": args.branch}

    from mattertune.backbones.deepmd.multihead import DeepMDHeadConfig
    from mattertune.backbones.deepmd.multihead_module import DeepMDDomainConfig, DeepMDMultiHeadBackboneConfig

    heads: list[Any] = []
    for name in args.resume_head or []:
        heads.append(DeepMDHeadConfig(name=name, init="resume"))
    for item in args.copy_head or []:
        name, source = item.split("=", 1) if "=" in item else ("", "")
        if not name or not source:
            raise SystemExit("--copy-head expects NEW_HEAD=SOURCE_BRANCH")
        heads.append(DeepMDHeadConfig(name=name, init="copy", source_branch=source))
    seeds = parse_key_values(args.head_seed, name="--head-seed")
    private_heads = parse_set(args.private_fitting_head)
    for name in args.random_head or []:
        if name not in seeds:
            raise SystemExit(f"--random-head {name} requires --head-seed {name}=SEED")
        if name not in private_heads:
            raise SystemExit(f"--random-head {name} requires --private-fitting-head {name}")
        heads.append(DeepMDHeadConfig(name=name, init="random", random_seed=int(seeds[name]), private_fitting_net=True))
    if not heads:
        raise SystemExit("--mode multihead requires at least one --resume-head, --copy-head, or --random-head")

    domains_raw = parse_key_values(args.domain, name="--domain")
    if not domains_raw:
        raise SystemExit("--mode multihead requires at least one --domain DOMAIN=HEAD")
    sampling = parse_key_values(args.sampling_weight, name="--sampling-weight")
    loss_weights = parse_key_values(args.loss_weight, name="--loss-weight")
    domains = [
        DeepMDDomainConfig(
            domain_id=domain,
            branch=branch,
            sampling_weight=float(sampling.get(domain, "1.0")),
            loss_weight=float(loss_weights.get(domain, "1.0")),
        )
        for domain, branch in domains_raw.items()
    ]
    model = DeepMDMultiHeadBackboneConfig(
        checkpoint_path=args.checkpoint,
        heads=heads,
        domains=domains,
        trainable_scope=args.trainable_scope,
        properties=property_configs(properties),
        optimizer=AdamWConfig(lr=args.learning_rate),
    )
    protocol = {
        "mode": "multihead",
        "heads": [dump_config_object(head) for head in heads],
        "domains": domains_raw,
        "sampling_weight": sampling,
        "loss_weight": loss_weights,
    }
    return model, "multihead_multi_domain", args.trainable_scope, ["multi_head", "multi_domain_training"], protocol


def build_sevennet(args: argparse.Namespace, properties: list[str]) -> tuple[Any, str, str, list[str], dict[str, Any]]:
    from mattertune.backbones.sevennet.model import SevenNetBackboneConfig
    from mattertune.finetune.optimizer import AdamWConfig

    mode = args.continual_mode
    required: list[str] = []
    if mode in {"replay", "replay-ewc"}:
        required.append("replay")
        missing = [name for name in ("replay_data", "replay_batch_size", "replay_ratio") if getattr(args, name) is None]
        if missing:
            raise SystemExit(
                f"--continual-mode {mode} requires: {', '.join('--' + m.replace('_', '-') for m in missing)}"
            )
    if mode in {"ewc", "replay-ewc"}:
        required.append("ewc")
        if mode == "replay-ewc":
            required.append("replay_plus_ewc")
        missing = [name for name in ("fisher", "reference", "lambda_ewc") if getattr(args, name) is None]
        if missing:
            raise SystemExit(
                f"--continual-mode {mode} requires: {', '.join('--' + m.replace('_', '-') for m in missing)}"
            )
    if mode == "none":
        forbidden = ["replay_data", "replay_batch_size", "replay_ratio", "fisher", "reference", "lambda_ewc"]
        used = [name for name in forbidden if getattr(args, name) is not None]
        if used:
            raise SystemExit("--continual-mode none does not accept continual-learning artifact flags")
    model = SevenNetBackboneConfig(
        checkpoint_path=args.checkpoint,
        properties=property_configs(properties),
        optimizer=AdamWConfig(lr=args.learning_rate),
    )
    protocol = {
        "continual_mode": mode,
        "replay_data": args.replay_data,
        "replay_batch_size": args.replay_batch_size,
        "replay_ratio": args.replay_ratio,
        "fisher": args.fisher,
        "reference": args.reference,
        "lambda_ewc": args.lambda_ewc,
        "replay_update_semantics": "separate_optimizer_step",
    }
    return model, mode, "full", required, protocol


def build_chgnet(args: argparse.Namespace, properties: list[str]) -> tuple[Any, str, str, list[str], dict[str, Any]]:
    from mattertune.backbones.chgnet.model import CHGNetBackboneConfig
    from mattertune.finetune.optimizer import AdamWConfig

    objective_props = properties_for_chgnet_objective(args.objective, properties)
    model = CHGNetBackboneConfig(
        checkpoint_path=args.checkpoint,
        atomref_mode=args.atomref_mode,
        properties=property_configs(objective_props, energy_basis=args.energy_basis),
        optimizer=AdamWConfig(lr=args.learning_rate),
    )
    protocol = {
        "objective": args.objective,
        "atomref_mode": args.atomref_mode,
        "energy_basis": args.energy_basis,
        "magmom_convention": "unsigned_magnitude" if "magnetic_moments" in objective_props else None,
    }
    return model, f"{args.objective}:{args.atomref_mode}", args.atomref_mode, ["full_fine_tuning"], protocol


def build_mace(
    args: argparse.Namespace, properties: list[str]
) -> tuple[Any, str, str, list[str], list[Any], dict[str, Any]]:
    from mattertune.backbones.mace_foundation.model import MACEBackboneConfig
    from mattertune.finetune.optimizer import AdamWConfig
    from mattertune.recipes.lora import LoraConfig, LoRARecipeConfig

    if not args.trust_checkpoint:
        raise SystemExit("MACE requires --trust-checkpoint for local torch/pickle checkpoint loading")
    if args.head_mode == "multi_head" and not args.head:
        raise SystemExit("MACE --head-mode multi_head requires --head")
    loss_coefficients = {
        "energy": args.energy_weight,
        "forces": args.forces_weight,
        "stresses": args.stress_weight,
    }
    model = MACEBackboneConfig(
        pretrained_model=args.checkpoint,
        head=args.head,
        properties=property_configs(properties, loss_coefficients=loss_coefficients),
        optimizer=AdamWConfig(lr=args.learning_rate),
    )
    recipes: list[Any] = []
    required = [args.head_mode]
    mode = args.head_mode
    if args.head:
        mode += f":{args.head}"
    if args.lora:
        missing = [name for name in ("lora_rank", "lora_alpha") if getattr(args, name) is None]
        if missing:
            raise SystemExit("--lora requires: " + ", ".join("--" + name.replace("_", "-") for name in missing))
        required.append("lora")
        mode += "+lora"
        recipes.append(
            LoRARecipeConfig(lora=LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules="all-linear"))
        )
    protocol = {
        "head_mode": args.head_mode,
        "head": args.head,
        "trust_checkpoint": True,
        "checkpoint_loading": "trusted_local_torch_load",
        "lora": args.lora,
        "lora_rank": args.lora_rank if args.lora else None,
        "lora_alpha": args.lora_alpha if args.lora else None,
        "loss_weights": {key: value for key, value in loss_coefficients.items() if key in properties},
        "native_mace_multihead_replay": False,
        "tier_d_execution": args.head_mode == "single_head",
    }
    return model, mode, "full", required, recipes, protocol


def config_to_preview(config: Any) -> dict[str, Any]:
    if hasattr(config, "model_dump"):
        return config.model_dump(mode="json", exclude_none=True)
    return {"repr": repr(config)}


def emit(plan: Plan, args: argparse.Namespace) -> None:
    plan_dict = plan.as_dict()
    manifest = Path(args.manifest) if args.manifest else Path(plan.run_dir) / "execution_plan.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(plan_dict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(plan_dict, indent=2, sort_keys=True))
        return
    flat = {
        "model_type": plan.model_type,
        "checkpoint_identity": plan.checkpoint_identity,
        "requested_properties": ",".join(plan.requested_properties),
        "adaptation_mode": plan.adaptation_mode,
        "capabilities_required": ",".join(plan.capabilities_required) if plan.capabilities_required else "none",
        "trainable_scope": plan.trainable_scope,
        "train_data": plan.data_paths.get("train") or "none",
        "val_data": plan.data_paths.get("val") or "none",
        "run_dir": plan.run_dir,
        "resume_mode": plan.resume_mode,
        "trusted_checkpoint_requirement": plan.trusted_checkpoint_requirement,
        "mattertune_backbone_name": plan.mattertune_backbone_name,
        "capability_source": plan.capability_source,
        "capability_errors": ";".join(plan.capability_errors) if plan.capability_errors else "none",
        "parity_source": plan.parity_source,
        "parity_errors": ";".join(plan.parity_errors) if plan.parity_errors else "none",
        "mattertune_config_valid": str(plan.mattertune_config_valid).lower(),
        "mattertune_config_class": plan.mattertune_config_class,
        "manifest": str(manifest),
    }
    for key, value in flat.items():
        print(f"{key}={value}")
    for key in sorted(plan.capabilities):
        value = plan.capabilities[key]
        if isinstance(value, list):
            value = ",".join(str(v) for v in value)
        print(f"capability.{key}={str(value).lower()}")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--properties", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int_or_auto, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", type=parse_devices, default="auto")
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--checkpoint-every-n-epochs", type=int, default=100)
    parser.add_argument("--energy-key")
    parser.add_argument("--forces-key")
    parser.add_argument("--stress-key")
    parser.add_argument("--resume-from")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--manifest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MatterTune Bash-first training adapter")
    sub = parser.add_subparsers(dest="model_type", required=True)

    deepmd = sub.add_parser("deepmd", help="DeepMD/DPA single-branch and multi-head protocols")
    add_common(deepmd)
    deepmd.add_argument("--mode", choices=["single", "multihead"], required=True)
    deepmd.add_argument("--branch")
    deepmd.add_argument("--resume-head", action="append", metavar="HEAD")
    deepmd.add_argument("--copy-head", action="append", metavar="NEW_HEAD=SOURCE_BRANCH")
    deepmd.add_argument("--random-head", action="append", metavar="HEAD")
    deepmd.add_argument("--head-seed", action="append", metavar="HEAD=SEED")
    deepmd.add_argument("--private-fitting-head", action="append", metavar="HEAD")
    deepmd.add_argument("--domain", action="append", metavar="DOMAIN=HEAD")
    deepmd.add_argument("--sampling-weight", action="append", metavar="DOMAIN=WEIGHT")
    deepmd.add_argument("--loss-weight", action="append", metavar="DOMAIN=WEIGHT")
    deepmd.add_argument("--trainable-scope", choices=["full", "head_only"], default="full")

    sevennet = sub.add_parser("sevennet", help="SevenNet native and continual protocols")
    add_common(sevennet)
    sevennet.add_argument("--continual-mode", choices=["none", "replay", "ewc", "replay-ewc"], required=True)
    sevennet.add_argument("--replay-data")
    sevennet.add_argument("--replay-batch-size", type=int)
    sevennet.add_argument("--replay-ratio", type=float)
    sevennet.add_argument("--fisher")
    sevennet.add_argument("--reference")
    sevennet.add_argument("--lambda-ewc", type=float)

    chgnet = sub.add_parser("chgnet", help="CHGNet native objective protocols")
    add_common(chgnet)
    chgnet.add_argument("--objective", choices=["ef", "efs", "efm", "efsm"], required=True)
    chgnet.add_argument("--atomref-mode", choices=["preserve", "network_only", "atomref_only", "joint"], required=True)
    chgnet.add_argument("--energy-basis", choices=["per_atom", "total"], required=True)

    mace = sub.add_parser("mace", help="MACE trusted local checkpoint protocols")
    add_common(mace)
    mace.add_argument("--trust-checkpoint", action="store_true", required=True)
    mace.add_argument("--head-mode", choices=["single_head", "multi_head"], required=True)
    mace.add_argument("--head")
    mace.add_argument("--lora", action="store_true")
    mace.add_argument("--lora-rank", type=int)
    mace.add_argument("--lora-alpha", type=int)
    mace.add_argument("--energy-weight", type=float, default=1.0)
    mace.add_argument("--forces-weight", type=float, default=1.0)
    mace.add_argument("--stress-weight", type=float, default=1.0)
    mace.add_argument("--ema", action="store_true")
    mace.add_argument("--ema-decay", type=float, default=0.995)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run and args.model_type != "mace":
        raise SystemExit("training execution is currently Tier D only for MACE; use --dry-run for other models")
    if not args.dry_run and args.head_mode != "single_head":
        raise SystemExit(
            "MACE non-dry-run execution currently supports MatterTune single-head/full fine-tuning only; "
            "native MACE multi-head replay is not Tier D verified"
        )
    if not args.dry_run and args.max_steps == -1 and args.max_epochs is None:
        raise SystemExit("non-dry-run training requires --max-steps or --max-epochs to make the budget explicit")
    if args.learning_rate is None:
        raise SystemExit("--learning-rate is required until model-native optimizer defaults are encoded per protocol")
    if args.checkpoint_every_n_epochs < 1:
        raise SystemExit("--checkpoint-every-n-epochs must be >= 1")

    properties = parse_csv(args.properties)
    train_data = require_existing_file(args.train_data, "--train-data")
    val_data = require_existing_file(args.val_data, "--val-data")
    run_dir = require_sandbox_run_dir(args.run_dir)
    resume_mode = (
        f"resume_from:{require_existing_file(args.resume_from, '--resume-from')}" if args.resume_from else "fresh"
    )

    if args.model_type == "mace":
        ckpt_id = file_identity(args.checkpoint, "--checkpoint")
        trust_requirement = "trusted_local_torch_load"
    elif args.model_type in {"sevennet", "chgnet"} and not Path(args.checkpoint).is_file():
        ckpt_id = trusted_name_identity(args.checkpoint)
        trust_requirement = "trusted_official_name"
    else:
        ckpt_id = file_identity(args.checkpoint, "--checkpoint")
        trust_requirement = "local_sha256_required"

    try:
        if args.model_type == "deepmd":
            model_config, adaptation_mode, trainable_scope, required, protocol = build_deepmd(args, properties)
            recipes: list[Any] = []
        elif args.model_type == "sevennet":
            model_config, adaptation_mode, trainable_scope, required, protocol = build_sevennet(args, properties)
            recipes = []
        elif args.model_type == "chgnet":
            model_config, adaptation_mode, trainable_scope, required, protocol = build_chgnet(args, properties)
            recipes = []
        elif args.model_type == "mace":
            model_config, adaptation_mode, trainable_scope, required, recipes, protocol = build_mace(args, properties)
        else:
            raise AssertionError(args.model_type)
        mt_config = common_config(args, model_config, recipes)
    except Exception as exc:
        raise SystemExit(f"MatterTune config validation failed: {type(exc).__name__}: {exc}") from exc

    backbone_name = str(getattr(model_config, "name", args.model_type))
    capabilities, capability_source, capability_errors = preflight_capabilities(backbone_name, required)
    parity_status, parity_source, parity_errors = mattertune_source_parity(backbone_name)

    plan = Plan(
        model_type=args.model_type,
        checkpoint_identity=ckpt_id,
        requested_properties=properties,
        adaptation_mode=adaptation_mode,
        capabilities_required=required,
        trainable_scope=trainable_scope,
        data_paths={"train": train_data, "val": val_data},
        run_dir=run_dir,
        resume_mode=resume_mode,
        trusted_checkpoint_requirement=trust_requirement,
        capabilities=capabilities,
        mattertune_backbone_name=backbone_name,
        capability_source=capability_source,
        capability_errors=capability_errors,
        parity_status=parity_status,
        parity_source=parity_source,
        parity_errors=parity_errors,
        mattertune_config_valid=True,
        mattertune_config_class=type(mt_config).__name__,
        output_format="json" if args.json else "key=value",
        config_preview=config_to_preview(mt_config),
        protocol=protocol,
    )
    if args.dry_run:
        emit(plan, args)
        return 0

    run_manifest = Path(args.manifest) if args.manifest else Path(plan.run_dir) / "run_manifest.json"
    run_manifest.parent.mkdir(parents=True, exist_ok=True)
    run_manifest.write_text(
        json.dumps({**plan.as_dict(), "training_status": "started"}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    from mattertune.main import MatterTuner

    output = MatterTuner(mt_config).tune()
    metadata: dict[str, Any] = {}
    if hasattr(output.model, "execution_metadata"):
        try:
            execution_metadata = output.model.execution_metadata(output.trainer)
            metadata = (
                execution_metadata.model_dump(mode="json")
                if hasattr(execution_metadata, "model_dump")
                else dict(execution_metadata)
            )
        except Exception as exc:  # pragma: no cover - best-effort run artifact enrichment
            metadata = {"execution_metadata_error": f"{type(exc).__name__}: {exc}"}
    run_manifest.write_text(
        json.dumps(
            {
                **plan.as_dict(),
                "training_status": "completed",
                "execution_metadata": metadata,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("training_status=completed")
    print(f"manifest={run_manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            raise SystemExit(2) from None
        raise
