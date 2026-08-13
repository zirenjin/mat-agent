#!/usr/bin/env python3
"""State-machine runner for resilient model training.

This controller launches a model train API, streams logs into structured
telemetry, diagnoses failures through tools/anomaly_sniffer.py, and applies a
small set of deterministic self-healing actions that an upstream agent can
inspect and control through registries.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.optimize import BFGS

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

MODEL_CHOICES = ("chgnet", "deepmd", "mace", "sevennet")
TRAIN_ENTRYPOINT = ROOT / "docker" / "entrypoints" / "train.sh"

EPOCH_RE = re.compile(r"Epoch\s+(?P<epoch>\d+).*(?:loss=|loss[: ]+)(?P<loss>[-+0-9.eE]+)")
LOSS_RE = re.compile(r"(?:loss=|loss[: ]+)(?P<loss>[-+0-9.eE]+)")
VAL_RE = re.compile(r"(?:RMSE_E_per_atom=\s*(?P<rmse_e>[-+0-9.eE]+)|RMSE_F=\s*(?P<rmse_f>[-+0-9.eE]+)|val(?:idation)?[_ -]?(?:mae|loss)[=: ]+(?P<val>[-+0-9.eE]+))")
STEP_RE = re.compile(r"(?:step|iter(?:ation)?)\s*[=: ]\s*(?P<step>\d+)", re.IGNORECASE)


@dataclass
class ControllerState:
    interrupted: bool = False
    child: subprocess.Popen[str] | None = None
    run_id: str = field(default_factory=lambda: time.strftime("%Y%m%dT%H%M%S"))


class PairRepulsionCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, cutoff: float = 0.85, k: float = 20.0):
        super().__init__()
        self.cutoff = float(cutoff)
        self.k = float(k)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(self.atoms.get_positions(), dtype=float)
        n = len(positions)
        forces = np.zeros_like(positions)
        energy = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                vec = positions[i] - positions[j]
                dist = float(np.linalg.norm(vec))
                if dist <= 1e-12 or dist >= self.cutoff:
                    continue
                unit = vec / dist
                overlap = self.cutoff - dist
                energy += 0.5 * self.k * overlap * overlap
                f = self.k * overlap * unit
                forces[i] += f
                forces[j] -= f
        self.results = {"energy": energy, "forces": forces}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--model", choices=MODEL_CHOICES, default="mace", help="MatterTune model subcommand to run")
    p.add_argument("--api-script", default=None, help="Override train.sh path")
    p.add_argument("--telemetry-file", default=None, help="JSONL telemetry path")
    p.add_argument("--controller-log", default=None, help="Raw merged subprocess log path")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--latest-trajectory", default=None, help="Trajectory cache to inspect on failure")
    p.add_argument("--sniffer", default=str(TOOLS / "anomaly_sniffer.py"))
    p.add_argument("--relax-overlap-cutoff", type=float, default=0.85)
    p.add_argument("--relax-fmax", type=float, default=0.05)
    p.add_argument("--relax-steps", type=int, default=50)
    p.add_argument("--effective-batch-size", type=int, default=None)
    p.add_argument("--", dest="dashdash", action="store_true", help=argparse.SUPPRESS)
    args, rest = p.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    return args, rest


def main() -> int:
    args, train_args = parse_args()
    api_script = Path(args.api_script) if args.api_script else TRAIN_ENTRYPOINT
    if not api_script.is_file():
        raise SystemExit(f"train API not found: {api_script}")
    telemetry = Path(args.telemetry_file or _default_output_dir(train_args) / "runner_telemetry.jsonl")
    raw_log = Path(args.controller_log or _default_output_dir(train_args) / "runner_controller.log")
    telemetry.parent.mkdir(parents=True, exist_ok=True)
    raw_log.parent.mkdir(parents=True, exist_ok=True)

    state = ControllerState()
    _install_signal_handlers(state, telemetry)

    current_args = list(train_args)
    effective_batch = args.effective_batch_size or _get_int_flag(current_args, "--batch-size", 1)
    attempt = 0
    while True:
        attempt += 1
        _emit(telemetry, {"event": "attempt_start", "attempt": attempt, "args": current_args})
        rc = _run_once(api_script, [args.model, *current_args], raw_log, telemetry, state, attempt)
        if state.interrupted:
            _degrade_state(current_args, telemetry, reason="SIGNAL_INTERRUPTED")
            return 130
        if rc == 0:
            _emit(telemetry, {"event": "attempt_success", "attempt": attempt, "returncode": rc})
            return 0
        diagnosis = _run_sniffer(args.sniffer, raw_log, args.latest_trajectory or _get_flag(current_args, "--train-data"), telemetry)
        action = _self_heal(diagnosis, current_args, effective_batch, args, telemetry)
        if action is None or attempt > args.max_retries:
            _emit(telemetry, {"event": "attempt_failed_final", "attempt": attempt, "returncode": rc, "diagnosis": diagnosis})
            _degrade_state(current_args, telemetry, reason="UNRECOVERED_FAILURE")
            return rc
        current_args = action
        _emit(telemetry, {"event": "attempt_retry", "next_attempt": attempt + 1, "args": current_args})


def _run_once(api_script: Path, train_args: list[str], raw_log: Path, telemetry: Path, state: ControllerState, attempt: int) -> int:
    cmd = ["bash", str(api_script), *train_args]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with raw_log.open("a", encoding="utf-8") as log:
        log.write(f"\n===== attempt {attempt} command: {chr(32).join(cmd)} =====\n")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        state.child = proc
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            metrics = _parse_metrics(line)
            if metrics:
                metrics.update({"event": "metric", "attempt": attempt, "time": time.time()})
                _emit(telemetry, metrics)
            if _looks_like_failure(line):
                _emit(telemetry, {"event": "failure_signal", "attempt": attempt, "line": line.strip(), "time": time.time()})
        rc = proc.wait()
        state.child = None
        log.write(f"===== attempt {attempt} exit {rc} =====\n")
    return int(rc)


def _parse_metrics(line: str) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    if m := EPOCH_RE.search(line):
        out["epoch"] = int(m.group("epoch"))
        out["loss"] = _to_float(m.group("loss"))
    elif m := LOSS_RE.search(line):
        out["loss"] = _to_float(m.group("loss"))
    if m := STEP_RE.search(line):
        out["step"] = int(m.group("step"))
    for m in VAL_RE.finditer(line):
        if m.group("rmse_e") is not None:
            out["rmse_e_per_atom_mev"] = _to_float(m.group("rmse_e"))
        if m.group("rmse_f") is not None:
            out["rmse_f_mev_per_A"] = _to_float(m.group("rmse_f"))
        if m.group("val") is not None:
            out["val_mae_or_loss"] = _to_float(m.group("val"))
    return out or None


def _looks_like_failure(line: str) -> bool:
    return bool(re.search(r"cuda.*out of memory|\bnan\b|non[- ]?finite|loss.*diverg|gradient.*explod", line, re.I))


def _run_sniffer(sniffer: str, log_path: Path, trajectory: str | None, telemetry: Path) -> dict[str, Any]:
    cmd = [sys.executable, sniffer, "--log-file", str(log_path)]
    if trajectory:
        cmd.extend(["--trajectory", trajectory])
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=False)
        payload = json.loads(result.stdout) if result.stdout.strip() else {"status": "UNKNOWN", "diagnostics": {}}
    except Exception as exc:
        payload = {"status": "UNKNOWN", "error_class": "SNIFFER_FAILED", "diagnostics": {"error": str(exc)}}
    _emit(telemetry, {"event": "diagnosis", "payload": payload, "time": time.time()})
    return payload


def _self_heal(diagnosis: dict[str, Any], train_args: list[str], effective_batch: int, args: argparse.Namespace, telemetry: Path) -> list[str] | None:
    error_class = diagnosis.get("error_class")
    failure_mode = diagnosis.get("diagnostics", {}).get("failure_mode")
    new_args = list(train_args)
    if error_class == "OOM":
        old_bs = _get_int_flag(new_args, "--batch-size", 1)
        new_bs = max(1, old_bs // 2)
        if new_bs == old_bs:
            return None
        grad_accum = max(1, math.ceil(effective_batch / new_bs))
        _set_flag(new_args, "--batch-size", str(new_bs))
        _set_flag(new_args, "--finetune-mode", "resume")
        os.environ["MAT_AGENT_GRAD_ACCUM_STEPS"] = str(grad_accum)
        _emit(telemetry, {"event": "self_heal", "action": "OOM_REDUCE_BATCH", "old_batch": old_bs, "new_batch": new_bs, "grad_accum_steps": grad_accum})
        return new_args
    if failure_mode == "ATOMIC_COLLISION_IN_PHASE_SPACE":
        train_data = _get_flag(new_args, "--train-data")
        if not train_data:
            return None
        patched = _relax_overlaps(Path(train_data), cutoff=args.relax_overlap_cutoff, fmax=args.relax_fmax, steps=args.relax_steps)
        _set_flag(new_args, "--train-data", str(patched))
        _set_flag(new_args, "--grad-clip", "1.0")
        _set_flag(new_args, "--finetune-mode", "resume")
        _emit(telemetry, {"event": "self_heal", "action": "RELAX_COLLIDING_GEOMETRIES", "patched_train_data": str(patched)})
        return new_args
    return None


def _relax_overlaps(path: Path, cutoff: float, fmax: float, steps: int) -> Path:
    frames = read(str(path), ":")
    if isinstance(frames, Atoms):
        frames = [frames]
    out_frames = []
    for atoms in frames:
        atoms = atoms.copy()
        if _min_distance(atoms) is not None and _min_distance(atoms) < cutoff:
            atoms.calc = PairRepulsionCalculator(cutoff=cutoff)
            dyn = BFGS(atoms, logfile=None)
            dyn.run(fmax=fmax, steps=steps)
            atoms.calc = None
        out_frames.append(atoms)
    patched = path.with_suffix(path.suffix + ".relaxed.extxyz")
    write(str(patched), out_frames, format="extxyz")
    return patched


def _min_distance(atoms: Atoms) -> float | None:
    if len(atoms) < 2:
        return None
    d = atoms.get_all_distances(mic=bool(np.any(atoms.pbc)))
    d = np.asarray(d, dtype=float)
    d[d <= 0.0] = np.inf
    v = float(np.min(d))
    return v if np.isfinite(v) else None


def _degrade_state(train_args: list[str], telemetry: Path, reason: str) -> None:
    out_dir = _default_output_dir(train_args)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(out_dir.rglob("*.model")) + sorted(out_dir.rglob("*.pt"))
    payload = {
        "reason": reason,
        "time": time.time(),
        "output_dir": str(out_dir),
        "candidate_weights": [str(p) for p in candidates],
        "args": train_args,
    }
    (out_dir / "degraded_state.json").write_text(json.dumps(payload, indent=2))
    _emit(telemetry, {"event": "degraded_state", **payload})


def _install_signal_handlers(state: ControllerState, telemetry: Path) -> None:
    def handler(signum, _frame):
        state.interrupted = True
        _emit(telemetry, {"event": "signal", "signal": signum, "time": time.time()})
        if state.child and state.child.poll() is None:
            state.child.terminate()
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def _default_output_dir(args: list[str]) -> Path:
    value = _get_flag(args, "--run-dir")
    return Path(value) if value else ROOT / "playground/runs/runner_controller"


def _get_flag(args: list[str], flag: str) -> str | None:
    for i, item in enumerate(args):
        if item == flag and i + 1 < len(args):
            return args[i + 1]
        if item.startswith(flag + "="):
            return item.split("=", 1)[1]
    return None


def _get_int_flag(args: list[str], flag: str, default: int) -> int:
    value = _get_flag(args, flag)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _set_flag(args: list[str], flag: str, value: str) -> None:
    for i, item in enumerate(args):
        if item == flag and i + 1 < len(args):
            args[i + 1] = value
            return
        if item.startswith(flag + "="):
            args[i] = f"{flag}={value}"
            return
    args.extend([flag, value])


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _emit(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
