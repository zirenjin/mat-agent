#!/usr/bin/env python3
"""Bash-first dry-run adapter from mat-agent CLI flags to MatterTune configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CAPABILITY_MATRIX: dict[str, dict[str, Any]] = {
    "deepmd": {
        "single_branch": True,
        "multi_head": True,
        "multi_domain_training": True,
    },
    "sevennet": {
        "replay": True,
        "ewc": True,
        "replay_plus_ewc": True,
    },
    "chgnet": {
        "objectives": ["ef", "efs", "efm", "efsm"],
        "magnetic_moments": True,
        "signed_magnetic_moments": False,
        "atomref_modes": True,
    },
    "mace": {
        "named_head_selection": True,
        "single_head": True,
        "multi_head": True,
        "lora": True,
        "merged_deployment_export": True,
    },
}

SUPPORTED_PROPERTIES = {"energy", "forces", "stress", "stresses", "magmoms", "magnetic_moments"}


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
    allowed = path.is_relative_to("playground/runs") if not path.is_absolute() else "playground" in parts and "runs" in parts
    if not allowed:
        raise SystemExit("--run-dir must be inside playground/runs")
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def property_configs(properties: list[str], *, energy_basis: str = "total") -> list[Any]:
    from mattertune.finetune.loss import MAELossConfig
    from mattertune.finetune.properties import (
        EnergyPropertyConfig,
        ForcesPropertyConfig,
        MagneticMomentsPropertyConfig,
        StressesPropertyConfig,
    )

    configs: list[Any] = []
    for prop in properties:
        if prop == "energy":
            configs.append(EnergyPropertyConfig(loss=MAELossConfig(), loss_basis=energy_basis))
        elif prop == "forces":
            configs.append(ForcesPropertyConfig(loss=MAELossConfig(), conservative=True))
        elif prop == "stresses":
            configs.append(StressesPropertyConfig(loss=MAELossConfig(), conservative=True))
        elif prop == "magnetic_moments":
            configs.append(MagneticMomentsPropertyConfig(loss=MAELossConfig(), sign_convention="unsigned_magnitude"))
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
    from mattertune.main import MatterTunerConfig, TrainerConfig

    validation = XYZDatasetConfig(src=args.val_data) if args.val_data else None
    trainer_kwargs = {"inference_mode": False}
    data = ManualSplitDataModuleConfig(
        train=XYZDatasetConfig(src=args.train_data),
        validation=validation,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    trainer = TrainerConfig(
        max_steps=args.max_steps,
        max_epochs=None,
        deterministic=True,
        additional_trainer_kwargs=trainer_kwargs,
    )
    return MatterTunerConfig(model=model_config, data=data, trainer=trainer, recipes=recipes or [])


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
        return model, "single_branch", "full", ["single_branch"], {"mode": "single", "branch": args.branch}

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
        "heads": [getattr(head, "model_dump", lambda **_: repr(head))(mode="json") for head in heads],
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
            raise SystemExit(f"--continual-mode {mode} requires: {', '.join('--' + m.replace('_', '-') for m in missing)}")
    if mode in {"ewc", "replay-ewc"}:
        required.append("ewc")
        if mode == "replay-ewc":
            required.append("replay_plus_ewc")
        missing = [name for name in ("fisher", "reference", "lambda_ewc") if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"--continual-mode {mode} requires: {', '.join('--' + m.replace('_', '-') for m in missing)}")
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
    return model, f"{args.objective}:{args.atomref_mode}", args.atomref_mode, ["objectives", "atomref_modes"], protocol


def build_mace(args: argparse.Namespace, properties: list[str]) -> tuple[Any, str, str, list[str], list[Any], dict[str, Any]]:
    from mattertune.backbones.mace_foundation.model import MACEBackboneConfig
    from mattertune.finetune.optimizer import AdamWConfig
    from mattertune.recipes.lora import LoRARecipeConfig, LoraConfig

    if not args.trust_checkpoint:
        raise SystemExit("MACE requires --trust-checkpoint for local torch/pickle checkpoint loading")
    if args.head_mode == "multi_head" and not args.head:
        raise SystemExit("MACE --head-mode multi_head requires --head")
    model = MACEBackboneConfig(
        pretrained_model=args.checkpoint,
        properties=property_configs(properties),
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
        recipes.append(LoRARecipeConfig(lora=LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules="all-linear")))
    protocol = {
        "head_mode": args.head_mode,
        "head": args.head,
        "trust_checkpoint": True,
        "checkpoint_loading": "trusted_local_torch_load",
        "lora": args.lora,
        "lora_rank": args.lora_rank if args.lora else None,
        "lora_alpha": args.lora_alpha if args.lora else None,
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
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--resume-from")
    parser.add_argument("--dry-run", action="store_true", required=True)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run:
        raise SystemExit("only --dry-run execution planning is supported by this adapter")
    if args.learning_rate is None:
        raise SystemExit("--learning-rate is required until model-native optimizer defaults are encoded per protocol")

    properties = parse_csv(args.properties)
    train_data = require_existing_file(args.train_data, "--train-data")
    val_data = require_existing_file(args.val_data, "--val-data")
    run_dir = require_sandbox_run_dir(args.run_dir)
    resume_mode = f"resume_from:{require_existing_file(args.resume_from, '--resume-from')}" if args.resume_from else "fresh"

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

    capabilities = dict(CAPABILITY_MATRIX[args.model_type])
    missing = [name for name in required if capabilities.get(name) in (None, False)]
    if missing:
        raise SystemExit(f"required capabilities not supported by {args.model_type}: {missing}")

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
        mattertune_config_valid=True,
        mattertune_config_class=type(mt_config).__name__,
        output_format="json" if args.json else "key=value",
        config_preview=config_to_preview(mt_config),
        protocol=protocol,
    )
    emit(plan, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            raise SystemExit(2) from None
        raise
