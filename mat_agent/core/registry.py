#!/usr/bin/env python3
"""Registry loading helpers for agent-visible resources."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency fallback
    yaml = None


def load_registry(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON registry file."""
    p = Path(path)
    text = p.read_text()
    if p.suffix in {".json", ".jsonc"}:
        return json.loads(text)
    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML registries")
    data = yaml.safe_load(text)
    return data or {}
