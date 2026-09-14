#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_FAILURE_MARKERS = [
    "Traceback (most recent call last):",
    "ModuleNotFoundError:",
    "ImportError:",
    "No such file or directory",
    "Permission denied",
    "CUDA out of memory",
    "segmentation fault",
]
BAD_STATUSES = {
    "Abnormal",
    "Cancelled",
    "Canceled",
    "Error",
    "Failed",
    "Killed",
    "Stopped",
    "StopFailed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a Volc task config and watch early task health."
    )
    parser.add_argument("--conf", type=Path, help="Task YAML to submit.")
    parser.add_argument("--task-id", help="Existing task ID to watch.")
    parser.add_argument("--task-name", help="Existing task name to watch.")
    parser.add_argument(
        "--instance",
        help="Instance name for log streaming, defaults to the first live role.",
    )
    parser.add_argument(
        "--success-marker",
        action="append",
        default=[],
        help="Log substring that proves the task entered the expected path. Repeatable.",
    )
    parser.add_argument(
        "--failure-marker",
        action="append",
        default=[],
        help="Extra log substring that should fail fast. Repeatable.",
    )
    parser.add_argument(
        "--min-monitor-seconds",
        type=int,
        default=60,
        help="Minimum monitoring duration before declaring success.",
    )
    parser.add_argument(
        "--max-monitor-seconds",
        type=int,
        default=180,
        help="Maximum monitoring duration before giving up.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=15,
        help="Polling interval for status and log checks.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional short task purpose shown in local monitor output.",
    )
    parser.add_argument(
        "--registry-file",
        type=Path,
        default=None,
        help="Optional Markdown file to append a submission snapshot to. Disabled by default.",
    )
    args = parser.parse_args()

    if not args.conf and not args.task_id and not args.task_name:
        parser.error("pass --conf, --task-id, or --task-name")
    if not args.success_marker:
        parser.error("pass at least one --success-marker")
    if args.min_monitor_seconds < 1:
        parser.error("--min-monitor-seconds must be positive")
    if args.max_monitor_seconds < args.min_monitor_seconds:
        parser.error("--max-monitor-seconds must be >= --min-monitor-seconds")
    return args


def run_cmd(argv: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(argv, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(argv)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def load_config(conf_path: Path) -> dict:
    with conf_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"unexpected YAML payload in {conf_path}")
    return data


def task_name_from_config(payload: dict) -> str:
    for key in ("TaskName", "Name"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise RuntimeError("task config does not define TaskName or Name")


def description_from_config(payload: dict) -> str:
    for key in ("Description",):
        value = payload.get(key)
        if isinstance(value, str):
            return value.strip()
    return ""


def submit_config(conf_path: Path) -> None:
    proc = run_cmd(["volc", "ml_task", "submit", "--conf", str(conf_path)], check=False)
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip(), file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"submit failed for {conf_path}")


def parse_json_output(raw: str):
    raw = raw.strip()
    if not raw:
        return []
    decoder = json.JSONDecoder()
    for idx, char in enumerate(raw):
        if char not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(raw[idx:])
        except json.JSONDecodeError:
            continue
        return payload
    return json.loads(raw)


def fetch_task_list(task_name: str) -> list[dict]:
    proc = run_cmd(
        ["volc", "ml_task", "list", "--name", task_name, "--output", "json"]
    )
    payload = parse_json_output(proc.stdout)
    return payload if isinstance(payload, list) else [payload]


def fetch_task_summary(task_id: str) -> dict:
    proc = run_cmd(["volc", "ml_task", "get", "--id", task_id, "--output", "json"])
    payload = parse_json_output(proc.stdout)
    if isinstance(payload, list):
        if not payload:
            raise RuntimeError(f"no task summary returned for {task_id}")
        return payload[0]
    if isinstance(payload, dict):
        return payload
    raise RuntimeError(f"unexpected task summary shape for {task_id}")


def resolve_task_id(task_name: str, timeout_seconds: int = 90) -> str:
    deadline = time.monotonic() + timeout_seconds
    last_seen = []
    while time.monotonic() < deadline:
        tasks = fetch_task_list(task_name)
        if tasks:
            tasks.sort(key=lambda item: (item.get("Start") or "", item.get("JobId") or ""))
            return tasks[-1]["JobId"]
        last_seen = tasks
        time.sleep(5)
    raise RuntimeError(f"could not resolve task id for {task_name}; last seen: {last_seen}")


def default_instance(summary: dict) -> str:
    role_specs = summary.get("TaskRoleSpecs") or []
    for spec in role_specs:
        if not isinstance(spec, dict):
            continue
        replicas = spec.get("RoleReplicas") or 0
        role_name = spec.get("RoleName")
        if isinstance(role_name, str) and replicas:
            return f"{role_name}_0"
    return "worker_0"


def fetch_logs(task_id: str, instance: str, lines: int = 200) -> str:
    proc = run_cmd(
        [
            "volc",
            "ml_task",
            "logs",
            "--task",
            task_id,
            "--instance",
            instance,
            "--lines",
            str(lines),
        ],
        check=False,
    )
    output = proc.stdout.strip()
    if not output and proc.stderr.strip():
        output = proc.stderr.strip()
    return output


def extract_excerpt(log_text: str, max_lines: int = 20) -> str:
    lines = [line for line in log_text.splitlines() if line.strip()]
    if not lines:
        return "(no logs captured)"
    return "\n".join(lines[-max_lines:])


def append_registry(
    registry_file: Path,
    *,
    task_name: str,
    task_id: str,
    status: str,
    outcome: str,
    conf_path: Optional[Path],
    description: str,
    instance: str,
    success_markers: list[str],
    notes: str,
) -> None:
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    if not registry_file.exists():
        registry_file.write_text(
            "# Submitted Volc Tasks\n\n"
            "Append one entry per submission or verification pass.\n\n",
            encoding="utf-8",
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"## {timestamp} {task_name}",
        "",
        f"- Task ID: `{task_id}`",
        f"- Status snapshot: `{status}`",
        f"- Outcome: `{outcome}`",
        f"- Config: `{conf_path}`" if conf_path else "- Config: `(monitor-only)`",
        f"- Instance: `{instance}`",
        f"- Success markers: `{', '.join(success_markers)}`",
        f"- Description: {description or '(empty)'}",
        f"- Notes: {notes or '(empty)'}",
        "",
    ]
    with registry_file.open("a", encoding="utf-8") as handle:
        handle.write("\n" + "\n".join(lines))


def main() -> int:
    args = parse_args()
    conf_payload = {}
    conf_path = args.conf.resolve() if args.conf else None
    task_name = args.task_name
    description = ""

    if conf_path:
        conf_payload = load_config(conf_path)
        task_name = task_name or task_name_from_config(conf_payload)
        description = description_from_config(conf_payload)
        print(f"Submitting config: {conf_path}")
        submit_config(conf_path)

    if args.task_id:
        task_id = args.task_id
    else:
        if not task_name:
            raise RuntimeError("need a task name to resolve task id")
        print(f"Resolving task ID for: {task_name}")
        task_id = resolve_task_id(task_name)

    print(f"Watching task: {task_id}")
    if args.notes:
        print(f"Notes: {args.notes}")
    start_time = time.monotonic()
    instance = args.instance
    marker_seen = ""
    last_status = "Unknown"
    last_excerpt = "(no logs captured)"
    failure_markers = DEFAULT_FAILURE_MARKERS + args.failure_marker

    while True:
        summary = fetch_task_summary(task_id)
        last_status = str(summary.get("Status") or summary.get("State") or "Unknown")
        task_name = task_name or str(summary.get("JobName") or task_id)
        instance = instance or default_instance(summary)
        log_text = fetch_logs(task_id, instance)
        last_excerpt = extract_excerpt(log_text)

        for marker in args.success_marker:
            if marker and marker in log_text:
                marker_seen = marker
                break

        for marker in failure_markers:
            if marker and marker in log_text:
                if args.registry_file:
                    append_registry(
                        args.registry_file,
                        task_name=task_name,
                        task_id=task_id,
                        status=last_status,
                        outcome=f"failed-fast: {marker}",
                        conf_path=conf_path,
                        description=description,
                        instance=instance,
                        success_markers=args.success_marker,
                        notes=args.notes,
                    )
                print(f"Detected failure marker: {marker}", file=sys.stderr)
                print(last_excerpt, file=sys.stderr)
                return 1

        elapsed = int(time.monotonic() - start_time)
        print(
            f"[{elapsed:>3}s] status={last_status} instance={instance} "
            f"marker={'seen' if marker_seen else 'pending'}"
        )

        if last_status in BAD_STATUSES:
            if args.registry_file:
                append_registry(
                    args.registry_file,
                    task_name=task_name,
                    task_id=task_id,
                    status=last_status,
                    outcome="failed-status",
                    conf_path=conf_path,
                    description=description,
                    instance=instance,
                    success_markers=args.success_marker,
                    notes=args.notes,
                )
            print(last_excerpt, file=sys.stderr)
            return 1

        if elapsed >= args.min_monitor_seconds and marker_seen:
            if args.registry_file:
                append_registry(
                    args.registry_file,
                    task_name=task_name,
                    task_id=task_id,
                    status=last_status,
                    outcome=f"verified: {marker_seen}",
                    conf_path=conf_path,
                    description=description,
                    instance=instance,
                    success_markers=args.success_marker,
                    notes=args.notes,
                )
            print(f"Verified success marker: {marker_seen}")
            print(last_excerpt)
            return 0

        if elapsed >= args.max_monitor_seconds:
            if args.registry_file:
                append_registry(
                    args.registry_file,
                    task_name=task_name,
                    task_id=task_id,
                    status=last_status,
                    outcome="unverified: marker not found before timeout",
                    conf_path=conf_path,
                    description=description,
                    instance=instance,
                    success_markers=args.success_marker,
                    notes=args.notes,
                )
            print(last_excerpt, file=sys.stderr)
            return 1

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
