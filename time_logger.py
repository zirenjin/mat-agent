"""
time_logger.py — Record per-step timing for MLIP agent eval experiments.

Two usage modes:

(A) Python context manager (preferred for human or agent that runs Python):

    from time_logger import TimingSession

    with TimingSession(
        model="mace", task="inference", runner="human",
        dataset={"name": "demo_100", "n_structures": 100},
    ) as s:
        with s.step("env_setup", category="env_setup", command="pip install mace-torch"):
            os.system("pip install mace-torch")
        with s.step("data_conv", category="data_io"):
            ...
        # automatically saves on context exit
        s.set_outputs(predictions_path="preds.xyz")

(B) Shell CLI mode (for bash scripts / agent that prefers shell):

    python time_logger.py init mace inference human --dataset demo_100
    python time_logger.py step_start env_setup env_setup "pip install mace-torch"
    pip install mace-torch
    python time_logger.py step_end env_setup success
    ...
    python time_logger.py finalize success

Output: results/time_logs/{experiment_id}.json conforming to schema/time_log.schema.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LOG_DIR = Path("results/time_logs")
DEFAULT_STATE_PATH = Path(".time_logger_state.json")  # for CLI mode


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _detect_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {"cpu": platform.processor() or platform.machine()}
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            info["gpu"] = out.stdout.strip().splitlines()[0]
        else:
            info["gpu"] = "none"
    except Exception:
        info["gpu"] = "unknown"
    try:
        import psutil  # type: ignore
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    return info


class TimingSession:
    """Context manager that records every step's wallclock and writes a JSON log."""

    def __init__(
        self,
        model: str,
        task: str,
        runner: str,
        run_id: str | None = None,
        dataset: dict | None = None,
        agent_info: dict | None = None,
        log_dir: Path | str = DEFAULT_LOG_DIR,
        human_notes: str = "",
    ):
        assert model in {"mace", "fairchem", "dpmd"}, model
        assert task in {"inference", "training", "evaluation"}, task
        assert runner in {"human", "agent"}, runner

        self.run_id = run_id or f"run{uuid.uuid4().hex[:6]}"
        self.experiment_id = f"{model}_{task}_{runner}_{self.run_id}"
        self.model = model
        self.task = task
        self.runner = runner
        self.dataset = dataset or {}
        self.agent_info = agent_info or {}
        self.human_notes = human_notes
        self.agent_notes = ""
        self.log_dir = Path(log_dir)
        self.steps: list[dict] = []
        self.outputs: dict = {}
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.success: bool = False
        self.failure_step: str | None = None
        self._t0: float | None = None

    # --- lifecycle -------------------------------------------------------

    def __enter__(self):
        self.started_at = _now_iso()
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finished_at = _now_iso()
        if exc_type is None and self.failure_step is None:
            self.success = True
        else:
            self.success = False
            if exc_type is not None and self.failure_step is None:
                self.failure_step = self.steps[-1]["step_id"] if self.steps else "unknown"
        self.save()
        # don't swallow exceptions
        return False

    # --- step recording --------------------------------------------------

    @contextmanager
    def step(
        self,
        step_id: str,
        category: str,
        command: str = "",
        notes: str = "",
    ):
        assert category in {
            "env_setup", "data_io", "data_conversion", "model_download",
            "inference", "training", "evaluation", "result_io", "other",
        }, category
        started = _now_iso()
        t0 = time.time()
        record: dict[str, Any] = {
            "step_id": step_id,
            "category": category,
            "started_at": started,
            "command": command,
            "notes": notes,
            "status": "success",
        }
        try:
            yield record
        except Exception as e:
            record["status"] = "failure"
            record["error_message"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"
            self.failure_step = step_id
            raise
        finally:
            record["finished_at"] = _now_iso()
            record["duration_seconds"] = round(time.time() - t0, 3)
            self.steps.append(record)

    def mark_step(
        self,
        step_id: str,
        category: str,
        duration_seconds: float,
        status: str = "success",
        command: str = "",
        notes: str = "",
        error_message: str | None = None,
    ):
        """Record a step you timed manually (e.g., a shell command run outside Python)."""
        self.steps.append({
            "step_id": step_id,
            "category": category,
            "started_at": _now_iso(),
            "finished_at": _now_iso(),
            "duration_seconds": round(float(duration_seconds), 3),
            "status": status,
            "command": command,
            "notes": notes,
            "error_message": error_message,
        })

    # --- outputs / notes -------------------------------------------------

    def set_outputs(self, **kwargs):
        self.outputs.update(kwargs)

    def set_metrics(self, **metrics):
        self.outputs.setdefault("metrics", {}).update(metrics)

    def add_agent_note(self, note: str):
        self.agent_notes += (note + "\n")

    # --- persistence -----------------------------------------------------

    def to_dict(self) -> dict:
        if self.finished_at is None:
            self.finished_at = _now_iso()
        duration = None
        if self._t0 is not None:
            duration = round(time.time() - self._t0, 3)
        return {
            "experiment_id": self.experiment_id,
            "runner": self.runner,
            "agent_info": self.agent_info,
            "model": self.model,
            "task": self.task,
            "dataset": self.dataset,
            "hardware": _detect_hardware(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": duration,
            "success": self.success,
            "failure_step": self.failure_step,
            "steps": self.steps,
            "outputs": self.outputs,
            "agent_notes": self.agent_notes.strip(),
            "human_notes": self.human_notes,
        }

    def save(self, path: Path | str | None = None) -> Path:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = Path(path) if path else (self.log_dir / f"{self.experiment_id}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[time_logger] wrote {path}")
        return path


# ============================================================================
# CLI mode — for bash scripts / shells that can't use the context manager
# ============================================================================


def _load_state() -> dict:
    if DEFAULT_STATE_PATH.exists():
        return json.loads(DEFAULT_STATE_PATH.read_text())
    return {}


def _save_state(state: dict):
    DEFAULT_STATE_PATH.write_text(json.dumps(state, indent=2))


def _cli_init(args):
    state = {
        "experiment_id": f"{args.model}_{args.task}_{args.runner}_{args.run_id}",
        "model": args.model,
        "task": args.task,
        "runner": args.runner,
        "run_id": args.run_id,
        "dataset": {"name": args.dataset} if args.dataset else {},
        "started_at": _now_iso(),
        "_t0": time.time(),
        "steps": [],
        "_active_step": None,
        "outputs": {},
        "agent_notes": "",
        "human_notes": args.notes or "",
        "failure_step": None,
    }
    _save_state(state)
    print(f"[time_logger] initialized {state['experiment_id']}")


def _cli_step_start(args):
    state = _load_state()
    state["_active_step"] = {
        "step_id": args.step_id,
        "category": args.category,
        "started_at": _now_iso(),
        "_t0": time.time(),
        "command": args.command or "",
    }
    _save_state(state)
    print(f"[time_logger] step started: {args.step_id}")


def _cli_step_end(args):
    state = _load_state()
    active = state.get("_active_step")
    if not active:
        print("[time_logger] no active step", file=sys.stderr)
        sys.exit(1)
    duration = round(time.time() - active["_t0"], 3)
    record = {
        "step_id": active["step_id"],
        "category": active["category"],
        "started_at": active["started_at"],
        "finished_at": _now_iso(),
        "duration_seconds": duration,
        "status": args.status,
        "command": active["command"],
        "error_message": args.error or None,
        "notes": args.notes or "",
    }
    state["steps"].append(record)
    state["_active_step"] = None
    if args.status == "failure":
        state["failure_step"] = active["step_id"]
    _save_state(state)
    print(f"[time_logger] step ended: {active['step_id']} ({duration}s, {args.status})")


def _cli_set_output(args):
    state = _load_state()
    state["outputs"][args.key] = args.value
    _save_state(state)


def _cli_set_metric(args):
    state = _load_state()
    metrics = state["outputs"].setdefault("metrics", {})
    try:
        metrics[args.key] = float(args.value)
    except ValueError:
        metrics[args.key] = args.value
    _save_state(state)


def _cli_note(args):
    state = _load_state()
    field = "agent_notes" if state.get("runner") == "agent" else "human_notes"
    state[field] = (state.get(field, "") + args.note + "\n").strip()
    _save_state(state)


def _cli_finalize(args):
    state = _load_state()
    state["finished_at"] = _now_iso()
    state["duration_seconds"] = round(time.time() - state["_t0"], 3)
    state["success"] = (args.status == "success")
    state.pop("_t0", None)
    state.pop("_active_step", None)
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = DEFAULT_LOG_DIR / f"{state['experiment_id']}.json"
    # final document — match schema
    final = {
        "experiment_id": state["experiment_id"],
        "runner": state["runner"],
        "agent_info": state.get("agent_info", {}),
        "model": state["model"],
        "task": state["task"],
        "dataset": state.get("dataset", {}),
        "hardware": _detect_hardware(),
        "started_at": state["started_at"],
        "finished_at": state["finished_at"],
        "duration_seconds": state["duration_seconds"],
        "success": state["success"],
        "failure_step": state.get("failure_step"),
        "steps": state["steps"],
        "outputs": state.get("outputs", {}),
        "agent_notes": state.get("agent_notes", ""),
        "human_notes": state.get("human_notes", ""),
    }
    out.write_text(json.dumps(final, indent=2))
    if DEFAULT_STATE_PATH.exists():
        DEFAULT_STATE_PATH.unlink()
    print(f"[time_logger] finalized → {out}")


def _build_cli():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init")
    p_init.add_argument("model", choices=["mace", "fairchem", "dpmd"])
    p_init.add_argument("task", choices=["inference", "training", "evaluation"])
    p_init.add_argument("runner", choices=["human", "agent"])
    p_init.add_argument("--run-id", dest="run_id", default="run1")
    p_init.add_argument("--dataset", default="")
    p_init.add_argument("--notes", default="")
    p_init.set_defaults(func=_cli_init)

    p_ss = sub.add_parser("step_start")
    p_ss.add_argument("step_id")
    p_ss.add_argument("category")
    p_ss.add_argument("command", nargs="?", default="")
    p_ss.set_defaults(func=_cli_step_start)

    p_se = sub.add_parser("step_end")
    p_se.add_argument("step_id")  # for sanity
    p_se.add_argument("status", choices=["success", "failure", "skipped"])
    p_se.add_argument("--error", default=None)
    p_se.add_argument("--notes", default="")
    p_se.set_defaults(func=_cli_step_end)

    p_out = sub.add_parser("set_output")
    p_out.add_argument("key")
    p_out.add_argument("value")
    p_out.set_defaults(func=_cli_set_output)

    p_met = sub.add_parser("set_metric")
    p_met.add_argument("key")
    p_met.add_argument("value")
    p_met.set_defaults(func=_cli_set_metric)

    p_note = sub.add_parser("note")
    p_note.add_argument("note")
    p_note.set_defaults(func=_cli_note)

    p_fin = sub.add_parser("finalize")
    p_fin.add_argument("status", choices=["success", "failure"])
    p_fin.set_defaults(func=_cli_finalize)

    return p


if __name__ == "__main__":
    args = _build_cli().parse_args()
    args.func(args)
