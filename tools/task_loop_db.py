#!/usr/bin/env python3
"""Record task-loop commits, artifacts, and results in a small SQLite database."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DB = Path("playground/loop_db/task_loops.sqlite")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def current_commit() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS loops (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            objective TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'started',
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            notes TEXT
        );
        CREATE TABLE IF NOT EXISTS loop_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            loop_id INTEGER NOT NULL REFERENCES loops(id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            git_commit TEXT,
            summary TEXT NOT NULL,
            artifact_path TEXT,
            metrics_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_loop_events_loop_id ON loop_events(loop_id);
        CREATE INDEX IF NOT EXISTS idx_loop_events_kind ON loop_events(kind);
        """
    )
    return con


def add_event(
    con: sqlite3.Connection,
    loop_id: int,
    *,
    kind: str,
    summary: str,
    artifact_path: str | None = None,
    metrics: dict[str, Any] | None = None,
    git_commit: str | None = None,
) -> None:
    now = utc_now()
    con.execute(
        """
        INSERT INTO loop_events(loop_id, kind, git_commit, summary, artifact_path, metrics_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            loop_id,
            kind,
            git_commit if git_commit is not None else current_commit(),
            summary,
            artifact_path,
            json.dumps(metrics, sort_keys=True) if metrics is not None else None,
            now,
        ),
    )
    con.execute("UPDATE loops SET updated_at=? WHERE id=?", (now, loop_id))
    con.commit()


def cmd_init(args: argparse.Namespace) -> None:
    with connect(Path(args.db)):
        pass
    print(f"db={args.db}")


def cmd_start(args: argparse.Namespace) -> None:
    now = utc_now()
    with connect(Path(args.db)) as con:
        cur = con.execute(
            "INSERT INTO loops(name, objective, status, started_at, updated_at, notes) VALUES (?, ?, ?, ?, ?, ?)",
            (args.name, args.objective, "started", now, now, args.notes),
        )
        loop_id = int(cur.lastrowid)
        add_event(con, loop_id, kind="start", summary=args.objective, git_commit=current_commit())
    print(f"loop_id={loop_id}")


def cmd_event(args: argparse.Namespace) -> None:
    metrics = json.loads(args.metrics_json) if args.metrics_json else None
    with connect(Path(args.db)) as con:
        add_event(
            con,
            args.loop_id,
            kind=args.kind,
            summary=args.summary,
            artifact_path=args.artifact,
            metrics=metrics,
            git_commit=args.git_commit,
        )
    print(f"event={args.kind}")


def cmd_finish(args: argparse.Namespace) -> None:
    metrics = json.loads(args.metrics_json) if args.metrics_json else None
    with connect(Path(args.db)) as con:
        add_event(
            con,
            args.loop_id,
            kind="result",
            summary=args.summary,
            artifact_path=args.artifact,
            metrics=metrics,
            git_commit=args.git_commit,
        )
        now = utc_now()
        con.execute("UPDATE loops SET status=?, updated_at=? WHERE id=?", (args.status, now, args.loop_id))
        con.commit()
    print(f"status={args.status}")


def cmd_list(args: argparse.Namespace) -> None:
    with connect(Path(args.db)) as con:
        rows = con.execute(
            """
            SELECT loops.id, loops.name, loops.status, loops.updated_at, COUNT(loop_events.id)
            FROM loops
            LEFT JOIN loop_events ON loop_events.loop_id = loops.id
            GROUP BY loops.id
            ORDER BY loops.updated_at DESC
            LIMIT ?
            """,
            (args.limit,),
        ).fetchall()
    for loop_id, name, status, updated_at, events in rows:
        print(f"{loop_id}\t{status}\t{events} events\t{updated_at}\t{name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init").set_defaults(func=cmd_init)

    start = sub.add_parser("start")
    start.add_argument("--name", required=True)
    start.add_argument("--objective", required=True)
    start.add_argument("--notes")
    start.set_defaults(func=cmd_start)

    event = sub.add_parser("event")
    event.add_argument("--loop-id", type=int, required=True)
    event.add_argument("--kind", choices=["architecture_commit", "result_commit", "test", "artifact", "note"], required=True)
    event.add_argument("--summary", required=True)
    event.add_argument("--artifact")
    event.add_argument("--metrics-json")
    event.add_argument("--git-commit")
    event.set_defaults(func=cmd_event)

    finish = sub.add_parser("finish")
    finish.add_argument("--loop-id", type=int, required=True)
    finish.add_argument("--summary", required=True)
    finish.add_argument("--status", choices=["completed", "failed", "blocked"], default="completed")
    finish.add_argument("--artifact")
    finish.add_argument("--metrics-json")
    finish.add_argument("--git-commit")
    finish.set_defaults(func=cmd_finish)

    list_cmd = sub.add_parser("list")
    list_cmd.add_argument("--limit", type=int, default=20)
    list_cmd.set_defaults(func=cmd_list)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
