#!/usr/bin/env python3
"""Shared schema helpers for mat-agent artifacts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON object after validating it can be parsed back."""
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    json.loads(text)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)


@dataclass
class ArtifactRef:
    """Pointer to a file produced or consumed by an agent action."""

    path: str
    kind: str
    description: str = ""


@dataclass
class ExperimentRecord:
    """Generic record for a materials-agent experiment."""

    name: str
    intent: str
    command: list[str] = field(default_factory=list)
    inputs: list[ArtifactRef] = field(default_factory=list)
    outputs: list[ArtifactRef] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    timestamp: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)
