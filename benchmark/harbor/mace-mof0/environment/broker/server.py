#!/usr/bin/env python3
"""Compute broker: the only place in this Harbor task that holds Volc
credentials or has the `volc` CLI installed. It is a thin HTTP wrapper
around ``tools/remote_job.py``'s own LOCAL-mode command functions -- this
file adds no submit/poll/collect logic of its own, only a JSON-over-HTTP
front door, so the concurrency/budget gate and the volc-CLI plumbing exist
in exactly one place regardless of which mode (local/client) is calling it.

The agent's own container (`main`) runs `tools/remote_job.py` with
``REMOTE_JOB_BROKER_URL`` set, which makes every subcommand call
HTTP-POST here instead of running locally -- see that file's module
docstring. `main` has neither `volc` nor credentials installed, so this is
a real boundary, not a documented convention the agent could bypass by
shelling out to `volc` directly (there is nothing to shell out to).

Uses only the standard library deliberately, to avoid adding a web
framework dependency to a component whose only job is to gate access to
credentials.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Authoritative for which compute-policy ledger `submit` gates against --
# see do_POST below for why this deliberately overrides whatever `task` the
# client sent, rather than trusting it. Set per-track by this track's own
# environment/docker-compose.yaml (smoke sets it; the standard track leaves
# it unset, so it falls back to mace_mof0_discovery).
BROKER_TASK = os.environ.get("REMOTE_JOB_DEFAULT_TASK", "mace_mof0_discovery")

REPO = Path("/workspace/mat-agent")
_RJ_PATH = REPO / "tools/remote_job.py"
_spec = importlib.util.spec_from_file_location("remote_job", _RJ_PATH)
remote_job = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(remote_job)

# cmd name (URL path, matches tools/remote_job.py's subparser names) ->
# (handler, required arg names, arg types/defaults for optional ones).
_ROUTES: dict[str, dict] = {
    "submit": {
        "func": remote_job.cmd_submit,
        "required": ["conf", "gpu_model", "gpu_count"],
        "defaults": {"task": BROKER_TASK, "run_id": None},
    },
    "status": {
        "func": remote_job.cmd_status,
        "required": ["job_id"],
        "defaults": {},
    },
    "cancel": {
        "func": remote_job.cmd_cancel,
        "required": ["job_id"],
        "defaults": {},
    },
    "wait": {
        "func": remote_job.cmd_wait,
        "required": ["job_id"],
        "defaults": {"poll_seconds": 60, "timeout_seconds": 0},
    },
    "collect": {
        "func": remote_job.cmd_collect,
        "required": ["job_id", "output"],
        "defaults": {},
    },
    "list-active": {
        "func": remote_job.cmd_list_active,
        "required": [],
        "defaults": {},
    },
}


class _Args:
    """Minimal argparse.Namespace look-alike built from a JSON request body."""

    def __init__(self, db: str, **kwargs):
        self.db = db
        for k, v in kwargs.items():
            setattr(self, k, v)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # quieter, timestamped stderr log
        sys.stderr.write(f"[broker] {self.address_string()} {fmt % args}\n")

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler naming
        route_name = self.path.strip("/")
        route = _ROUTES.get(route_name)
        if route is None:
            self._send_json(404, {"error": f"unknown route {route_name!r}"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"invalid JSON body: {exc}"})
            return

        missing = [k for k in route["required"] if body.get(k) in (None, "")]
        if missing:
            self._send_json(400, {"error": f"missing required field(s): {missing}"})
            return

        kwargs = dict(route["defaults"])
        kwargs.update(body)
        if route_name == "submit":
            # Deliberately override, not just default: `task` selects which
            # compute-policy concurrency cap applies (1 GPU for smoke, 4 for
            # standard). If the client's own value were trusted, a request
            # could simply claim --task mace_mof0_discovery from inside the
            # smoke track's tighter-capped environment and submit against the
            # looser standard-track ledger instead -- the same class of gap
            # the whole broker split exists to close for credentials. Which
            # ledger applies is a property of *this broker's own environment*
            # (BROKER_TASK, from REMOTE_JOB_DEFAULT_TASK), not of the request.
            kwargs["task"] = BROKER_TASK
        args = _Args(db=str(remote_job.DEFAULT_DB), **kwargs)

        try:
            result = route["func"](args)
            self._send_json(200, result)
        except SystemExit as exc:
            # cmd_submit raises SystemExit(str) for a policy-gate refusal --
            # that is an expected, meaningful rejection, not a server error.
            self._send_json(409, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 -- always report to the caller, never hang
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)  # noqa: S104 -- compose-internal network only
    print(f"[broker] listening on :{args.port}", file=sys.stderr)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
