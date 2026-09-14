"""Minimum-viable isolation for candidate inference.

Threat model: the candidate contributes a checkpoint and (optionally) a
``build_calculator`` entrypoint under ``workspace/``. Loading a MACE checkpoint
can execute pickle code, and an entrypoint is arbitrary Python. We must stop that
code from:

  * reading sequestered labels or the DFT reference assets,
  * reading or writing ``benchmark/`` or ``scoring/``,
  * writing anywhere under ``data/``,
  * reaching the network to fetch evaluation truth,
  * shelling out to do any of the above.

Mechanism: candidate code runs in a **separate spawned process** that installs a
``sys.addaudithook`` guard before importing anything candidate-controlled. Only
standardized numeric arrays cross back to the (unsandboxed) parent evaluator,
which is the only place DFT references are read and the scorecard is written.

This is not a container. It does not defend against kernel exploits, ptrace, or
``/proc`` tricks. For untrusted third-party submissions run the whole evaluator
inside a locked-down container as well (see protocol.md "hardened deployment").
Crucially it does NOT restrict what the candidate model *is* -- any architecture,
any framework -- only what its code may touch.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]

DENY_READ = [
    REPO / "data/mace_mof_0/discovery/_sequestered",
    REPO / "data/mace_mof_0/raw",
    REPO / "benchmark",
    REPO / "scoring",
]
# The sandboxed child IS the evaluator's own inference code -- it must import the
# few _lib modules that define the target function. Everything else under
# benchmark/ (protocol, scorecards, split manifests, baselines, integrity,
# aggregation, noise estimate) stays denied.
ALLOW_READ = [
    REPO / "benchmark/evaluation/_lib",
    REPO / "benchmark/evaluation/__init__.py",
]
DENY_WRITE = [
    REPO / "benchmark",
    REPO / "scoring",
    REPO / "data",
    REPO / "tools",
    REPO / "docker",
]


_NET_TOOLS = {"curl", "wget", "nc", "ncat", "netcat", "socat", "ssh", "scp", "sftp",
              "rsync", "ftp", "telnet", "git", "aria2c", "http", "https"}


class SandboxViolation(RuntimeError):
    """Candidate code attempted a forbidden operation."""


def _resolve(p: str) -> Path:
    try:
        return Path(p).resolve()
    except (OSError, ValueError):
        return Path(os.path.abspath(p))


def _under(path: Path, roots: list[Path]) -> bool:
    for r in roots:
        try:
            path.relative_to(r)
            return True
        except ValueError:
            continue
    return False


def _install_guard(extra_write_allow: list[str]) -> None:
    allow_write = [Path(p).resolve() for p in extra_write_allow]

    def hook(event: str, args: tuple) -> None:
        if event == "open":
            path = _resolve(str(args[0]))
            mode = args[1] or ""
            writing = any(c in str(mode) for c in "wax+") or (
                isinstance(args[2], int) and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)
            )
            if _under(path, ALLOW_READ) and not writing:
                return
            if _under(path, DENY_READ) and not _under(path, allow_write):
                raise SandboxViolation(f"blocked read: {path}")
            if writing and _under(path, DENY_WRITE) and not _under(path, allow_write):
                raise SandboxViolation(f"blocked write: {path}")
        elif event in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname"):
            raise SandboxViolation(f"blocked network: {event}")
        elif event in ("subprocess.Popen", "os.exec", "os.posix_spawn", "os.spawn"):
            # Child processes do not inherit this audit hook, so a spawn is only
            # dangerous if it can reach protected data or the network. ML init
            # (torch/e3nn/mace) legitimately shells out to CPU/GPU introspection
            # tools -- allow those, block network/exfil tools and any argv that
            # names a protected path.
            argv = args[1] if len(args) > 1 and args[1] else args[0]
            parts = [str(x) for x in argv] if isinstance(argv, (list, tuple)) else [str(argv)]
            exe = Path(parts[0]).name if parts else ""
            if exe in _NET_TOOLS:
                raise SandboxViolation(f"blocked spawn of network tool: {exe}")
            for p in parts:
                if p.startswith(("/", "./", "../")) or "/" in p:
                    rp = _resolve(p)
                    if _under(rp, DENY_READ) or _under(rp, DENY_WRITE):
                        raise SandboxViolation(f"blocked spawn touching protected path: {p}")

    sys.addaudithook(hook)


def _child(target: Callable[[dict], Any], payload: dict, q: mp.Queue, extra_write_allow: list[str]) -> None:
    try:
        sys.dont_write_bytecode = True   # avoid .pyc writes under denied roots
        _install_guard(extra_write_allow)
        q.put(("ok", target(payload)))
    except SandboxViolation as exc:
        q.put(("violation", str(exc)))
    except BaseException as exc:  # noqa: BLE001  -- report everything back
        import traceback
        q.put(("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


def run_in_sandbox(
    target: Callable[[dict], Any],
    payload: dict,
    *,
    timeout: float = 3600.0,
    write_allow: list[str] | None = None,
) -> Any:
    """Run ``target(payload)`` in a spawned, audit-guarded subprocess."""
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_child, args=(target, payload, q, list(write_allow or [])))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise SandboxViolation(f"candidate inference exceeded {timeout}s and was killed")
    if q.empty():
        raise SandboxViolation(f"candidate inference produced no result (exit {proc.exitcode})")
    status, value = q.get()
    if status == "violation":
        raise SandboxViolation(value)
    if status == "error":
        raise RuntimeError(f"candidate inference failed:\n{value}")
    return value
