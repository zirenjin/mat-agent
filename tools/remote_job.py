#!/usr/bin/env python3
"""Thin, programmable contract for submitting/polling/collecting Volc ML
Platform jobs from an autonomous agent, with a *live* concurrency gate that
``benchmark/compute/account_compute.py`` cannot provide on its own (that
script only sums completed ``run_manifest.json`` files after the fact --
concurrency is a live-state property).

This intentionally reuses, rather than reimplements, the existing
``volc/scripts/submit_and_watch_task.py`` helpers for talking to the ``volc``
CLI (submit / resolve-id / fetch-summary / fetch-logs), and the existing
``tools/task_loop_db.py`` sqlite file (adding one small table of its own,
``remote_jobs``, alongside that module's ``loops``/``loop_events`` tables).
It does not replace either.

Two modes, same CLI surface either way:

- LOCAL mode (default): runs the volc CLI calls and concurrency gate
  directly in this process. This is what the `broker` compose service runs
  (benchmark/harbor/mace-mof0/environment/broker/), which is the only place
  that ever holds Volc credentials.
- CLIENT mode: if the environment variable ``REMOTE_JOB_BROKER_URL`` is set,
  every subcommand is instead POSTed as JSON to that URL and the broker's
  JSON response is printed verbatim -- this process then never touches
  `volc` or credentials at all. This is what runs inside the agent's own
  `main` container (benchmark/harbor/mace-mof0/environment/Dockerfile),
  which intentionally has neither the `volc` binary nor Volc credentials
  installed, so the agent cannot bypass the concurrency/budget gate by
  shelling out to `volc` directly even if it wanted to -- there is nothing
  to shell out to. See benchmark/harbor/README.md ("Compute broker").

Commands:
    python tools/remote_job.py submit --conf PATH --gpu-model NAME --gpu-count N [--run-id ID]
    python tools/remote_job.py status --job-id ID
    python tools/remote_job.py cancel --job-id ID
    python tools/remote_job.py wait --job-id ID [--poll-seconds 60] [--timeout-seconds N]
    python tools/remote_job.py collect --job-id ID --output DIR
    python tools/remote_job.py list-active

All commands print one JSON object to stdout on success and exit non-zero
with a plain-text error on stderr on failure -- this is the machine contract
an agent should parse; do not scrape human-formatted volc CLI text.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / "benchmark/compute/policy.yaml"
DEFAULT_DB = REPO / "playground/loop_db/task_loops.sqlite"

# --- reuse volc/scripts/submit_and_watch_task.py instead of reimplementing
# volc-CLI plumbing (subprocess + JSON parsing) a second time -- but only
# import it lazily, from the LOCAL-mode command handlers below. This process
# also runs as the broker-CLIENT inside the agent's own container (see
# REMOTE_JOB_BROKER_URL below), which has neither the `volc` binary nor
# credentials at all; that mode must not require this module (or `volc`) to
# even be importable, let alone runnable.
saw = None


def _load_saw():
    global saw
    if saw is None:
        saw_path = REPO / "volc/scripts/submit_and_watch_task.py"
        spec = importlib.util.spec_from_file_location("submit_and_watch_task", saw_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        saw = module
    return saw


def _terminal_sets() -> tuple[set[str], set[str], set[str], set[str]]:
    """(TERMINAL_FAIL, TERMINAL_SUCCESS, TERMINAL, KNOWN_RUNNING).

    Ground truth from `volc ml_task get --help` / `volc ml_task list --help`
    on the dev host this was verified against (volc CLI 1.2.52): the full
    status enum is exactly Queue, Staging, Running, Killing, Success, Failed,
    Killed, Initialized -- `list`'s own default non-terminal filter is
    Queue,Staging,Running,Killing (Initialized is also pre-terminal but
    excluded from that default). Success is the *only* success-terminal
    value; Failed/Killed are the fail-terminal ones. We additionally keep
    submit_and_watch_task.py's BAD_STATUSES as a defensive superset (harmless
    if those strings never actually appear; still correctly treated as a
    failure if some other API path ever emits them).
    """
    terminal_fail = set(_load_saw().BAD_STATUSES) | {"Failed", "Killed"}
    terminal_success = {"Success"}
    return (
        terminal_fail,
        terminal_success,
        terminal_fail | terminal_success,
        {"Queue", "Staging", "Running", "Killing", "Initialized"},
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_policy() -> dict:
    import yaml  # type: ignore

    return yaml.safe_load(POLICY.read_text())


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS remote_jobs (
            job_id TEXT PRIMARY KEY,
            run_id TEXT,
            task TEXT NOT NULL,
            conf_path TEXT,
            gpu_model TEXT NOT NULL,
            gpu_count INTEGER NOT NULL,
            a100_equivalent REAL NOT NULL,
            status TEXT NOT NULL,
            started_utc TEXT NOT NULL,
            ended_utc TEXT
        );
        """
    )
    return con


def gpu_equivalent(policy: dict, gpu_model: str, gpu_count: int) -> float:
    factor = policy["common"]["gpu_equivalence"].get(gpu_model, policy["common"]["gpu_equivalence"]["other"])
    return float(gpu_count) * float(factor)


def refresh_and_sum_active(con: sqlite3.Connection, task: str | None = None) -> float:
    """Re-check every non-terminal row against live volc status (regardless
    of task -- a stale row belonging to any track should still get freed),
    then return the A100-equivalent GPU total still active FOR `task` only
    (or across all tasks if `task` is None). Standard and smoke tracks share
    this one sqlite file (both compose files bind-mount the same runs/), so
    without this filter a smoke run's 1-GPU cap could be blocked by an
    unrelated standard-track job's concurrency, or vice versa -- their
    compute-policy budgets are supposed to be entirely separate ledgers."""
    terminal, _, _, known_running = _terminal_sets()
    rows = con.execute("SELECT job_id FROM remote_jobs WHERE ended_utc IS NULL").fetchall()
    for (job_id,) in rows:
        try:
            summary = _load_saw().fetch_task_summary(job_id)
        except Exception as exc:  # noqa: BLE001 -- volc CLI hiccup, don't wedge the gate
            print(f"[remote_job] warning: could not refresh {job_id}: {exc}", file=sys.stderr)
            continue
        status = str(summary.get("Status") or summary.get("State") or "Unknown")
        if status in terminal:
            con.execute(
                "UPDATE remote_jobs SET status=?, ended_utc=? WHERE job_id=?",
                (status, utc_now(), job_id),
            )
        else:
            con.execute("UPDATE remote_jobs SET status=? WHERE job_id=?", (status, job_id))
            if status not in known_running:
                print(
                    f"[remote_job] warning: unrecognized non-terminal status '{status}' for {job_id}; "
                    "treating as still active. Verify against real `volc ml_task get` output and add it "
                    "to TERMINAL/KNOWN_RUNNING in tools/remote_job.py once confirmed.",
                    file=sys.stderr,
                )
    con.commit()
    if task is None:
        total = con.execute(
            "SELECT COALESCE(SUM(a100_equivalent), 0) FROM remote_jobs WHERE ended_utc IS NULL"
        ).fetchone()[0]
    else:
        total = con.execute(
            "SELECT COALESCE(SUM(a100_equivalent), 0) FROM remote_jobs WHERE ended_utc IS NULL AND task=?",
            (task,),
        ).fetchone()[0]
    return float(total)


def cmd_submit(args: argparse.Namespace) -> dict:
    policy = load_policy()
    tpol = policy["tasks"][args.task]
    cap = float(tpol["max_concurrent_a100_equivalent_gpus"])
    requested = gpu_equivalent(policy, args.gpu_model, args.gpu_count)

    con = connect(Path(args.db))
    active = refresh_and_sum_active(con, task=args.task)
    if active + requested > cap + 1e-9:
        raise SystemExit(
            f"refusing to submit: {active:.2f} A100-eq GPUs already active + "
            f"{requested:.2f} requested would exceed the {cap:.0f} A100-eq concurrency cap "
            f"(benchmark/compute/policy.yaml tasks.{args.task}.max_concurrent_a100_equivalent_gpus). "
            "Wait for an active job to finish (see: python tools/remote_job.py list-active) before "
            "submitting another."
        )

    conf_path = Path(args.conf).resolve()
    conf_payload = _load_saw().load_config(conf_path)
    task_name = _load_saw().task_name_from_config(conf_payload)
    _load_saw().submit_config(conf_path)
    job_id = _load_saw().resolve_task_id(task_name)

    con.execute(
        """
        INSERT INTO remote_jobs(job_id, run_id, task, conf_path, gpu_model, gpu_count,
                                 a100_equivalent, status, started_utc, ended_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
        """,
        (
            job_id,
            args.run_id,
            args.task,
            str(conf_path.relative_to(REPO)) if conf_path.is_relative_to(REPO) else str(conf_path),
            args.gpu_model,
            args.gpu_count,
            requested,
            "Submitted",
            utc_now(),
        ),
    )
    con.commit()
    return {
        "job_id": job_id,
        "run_id": args.run_id,
        "task_name": task_name,
        "a100_equivalent": requested,
        "active_a100_equivalent_after_submit": active + requested,
        "concurrency_cap": cap,
    }


def cmd_status(args: argparse.Namespace) -> dict:
    terminal, terminal_success, _, _ = _terminal_sets()
    con = connect(Path(args.db))
    summary = _load_saw().fetch_task_summary(args.job_id)
    status = str(summary.get("Status") or summary.get("State") or "Unknown")
    if status in terminal:
        con.execute(
            "UPDATE remote_jobs SET status=?, ended_utc=COALESCE(ended_utc, ?) WHERE job_id=?",
            (status, utc_now(), args.job_id),
        )
        con.commit()
    return {"job_id": args.job_id, "status": status, "terminal": status in terminal,
             "success": status in terminal_success}


def cmd_cancel(args: argparse.Namespace) -> dict:
    """Cancel a job on Volcengine and free its slot in the concurrency ledger.

    Exists so an abandoned/terminated Harbor trial does not leave an
    untracked GPU job running indefinitely -- Harbor tearing down `main`
    does not itself stop anything already submitted to Volcengine, and
    `list-active` on its own only shows the problem, it doesn't fix it.
    """
    _load_saw().run_cmd(["volc", "ml_task", "cancel", "--id", args.job_id], check=False)
    con = connect(Path(args.db))
    con.execute(
        "UPDATE remote_jobs SET status='Killed', ended_utc=COALESCE(ended_utc, ?) WHERE job_id=?",
        (utc_now(), args.job_id),
    )
    con.commit()
    return {"job_id": args.job_id, "cancelled": True}


def cmd_wait(args: argparse.Namespace) -> dict:
    deadline = time.monotonic() + args.timeout_seconds if args.timeout_seconds else None
    while True:
        result = cmd_status(args)
        print(f"[remote_job] {args.job_id}: status={result['status']}", file=sys.stderr)
        if result["terminal"]:
            return result
        if deadline is not None and time.monotonic() >= deadline:
            raise SystemExit(f"timed out after {args.timeout_seconds}s waiting for {args.job_id} "
                              f"(last status: {result['status']})")
        time.sleep(args.poll_seconds)


def cmd_collect(args: argparse.Namespace) -> dict:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_saw().fetch_task_summary(args.job_id)
    instance = _load_saw().default_instance(summary)
    log_text = _load_saw().fetch_logs(args.job_id, instance, lines=2000)
    log_path = out_dir / f"volc_job_{args.job_id}.log"
    log_path.write_text(log_text)
    # `REPO / "runs"` is genuinely the shared VePFS path here, not an
    # isolated container filesystem: environment/docker-compose.yaml bind-
    # mounts the host repo's runs/ into both the `main` (agent) and `broker`
    # services at this same path (see benchmark/harbor/README.md for how
    # that was confirmed -- harbor.environments.docker.docker.py explicitly
    # documents environment/docker-compose.yaml as a supported task-authored
    # overlay, "Option 2: Task with extra services / overrides", which is
    # what makes this bind mount possible; an earlier draft of this comment
    # wrongly concluded no such mechanism existed at all). A remote volc job
    # writing runs/<run_id>/run_manifest.json therefore really does just
    # appear here, in whichever of the two containers calls this function.
    manifest_hint = None
    con = connect(Path(args.db))
    row = con.execute("SELECT run_id FROM remote_jobs WHERE job_id=?", (args.job_id,)).fetchone()
    if row and row[0]:
        candidate = REPO / "runs" / row[0] / "run_manifest.json"
        manifest_hint = str(candidate.relative_to(REPO))
        if not candidate.is_file():
            print(f"[remote_job] warning: expected {manifest_hint} not found yet -- "
                  "the job may not have reached its manifest-writing step.", file=sys.stderr)
    return {"job_id": args.job_id, "log_path": str(log_path.relative_to(REPO)),
             "expected_run_manifest": manifest_hint}


def cmd_list_active(args: argparse.Namespace) -> dict:
    con = connect(Path(args.db))
    refresh_and_sum_active(con)  # refresh live status for every track; ignore the (unfiltered) return
    rows = con.execute(
        "SELECT job_id, run_id, task, gpu_model, gpu_count, a100_equivalent, status, started_utc "
        "FROM remote_jobs WHERE ended_utc IS NULL ORDER BY task, started_utc"
    ).fetchall()
    active = [
        dict(zip(
            ["job_id", "run_id", "task", "gpu_model", "gpu_count", "a100_equivalent", "status", "started_utc"],
            r,
            strict=True,
        ))
        for r in rows
    ]
    # Reported per-task, not pooled: standard and smoke tracks are separate
    # compute-policy ledgers even though they share this one sqlite file.
    totals_by_task: dict[str, float] = {}
    for row in active:
        totals_by_task[row["task"]] = totals_by_task.get(row["task"], 0.0) + row["a100_equivalent"]
    return {"active_jobs": active, "active_a100_equivalent_by_task": totals_by_task}


def _call_broker(broker_url: str, cmd: str, args: argparse.Namespace) -> dict:
    """POST this subcommand's args to the broker and return its JSON response.

    The wire contract is deliberately dumb: every argparse arg except `db`,
    `cmd`, and `func` (which are local-only / not serializable) goes straight
    into the JSON body, and the broker runs the exact same `cmd_*` function
    this process would have run locally. `--output` for `collect` also
    crosses over -- both `main` and `broker` bind-mount the same host `runs/`
    tree (environment/docker-compose.yaml), so a relative path means the
    same place in both containers.
    """
    payload = {k: v for k, v in vars(args).items() if k not in ("db", "cmd", "func")}
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{broker_url.rstrip('/')}/{cmd}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"broker rejected {cmd!r} ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"could not reach broker at {broker_url!r} for {cmd!r}: {exc.reason}. "
            "Is the `broker` compose service running? (see environment/docker-compose.yaml)"
        ) from exc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit")
    s.add_argument("--conf", required=True, help="volc ml_task config YAML, per volc/task-configs/**")
    s.add_argument("--task", default=os.environ.get("REMOTE_JOB_DEFAULT_TASK", "mace_mof0_discovery"),
                   help="benchmark/compute/policy.yaml task key (env REMOTE_JOB_DEFAULT_TASK overrides "
                        "the default -- set by the smoke task's docker-compose.yaml so a plain "
                        "'submit' without --task still lands in the smoke ledger, not the standard one)")
    s.add_argument("--gpu-model", required=True, help="must match a key in policy.yaml common.gpu_equivalence")
    s.add_argument("--gpu-count", type=int, required=True)
    s.add_argument("--run-id", default=None, help="runs/<run_id> this job writes its manifest+checkpoint under")
    s.set_defaults(func=cmd_submit)

    st = sub.add_parser("status")
    st.add_argument("--job-id", required=True)
    st.set_defaults(func=cmd_status)

    ca = sub.add_parser("cancel")
    ca.add_argument("--job-id", required=True)
    ca.set_defaults(func=cmd_cancel)

    w = sub.add_parser("wait")
    w.add_argument("--job-id", required=True)
    w.add_argument("--poll-seconds", type=int, default=60)
    w.add_argument("--timeout-seconds", type=int, default=0, help="0 = wait indefinitely")
    w.set_defaults(func=cmd_wait)

    c = sub.add_parser("collect")
    c.add_argument("--job-id", required=True)
    c.add_argument("--output", required=True, help="directory to write the fetched container log into")
    c.set_defaults(func=cmd_collect)

    la = sub.add_parser("list-active")
    la.set_defaults(func=cmd_list_active)

    args = ap.parse_args()
    broker_url = os.environ.get("REMOTE_JOB_BROKER_URL")
    result = _call_broker(broker_url, args.cmd, args) if broker_url else args.func(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(str(exc), file=sys.stderr)
        raise
