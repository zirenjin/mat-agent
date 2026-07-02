#!/usr/bin/env python3
"""Registry and repository path helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency fallback
    yaml = None


def repo_root(start: str | Path | None = None) -> Path:
    """Find the repository root by walking upward to model/tool directories."""
    cur = Path(start or __file__).resolve()
    if cur.is_file():
        cur = cur.parent
    for parent in [cur, *cur.parents]:
        if (parent / "models").is_dir() and (parent / "tools").is_dir():
            return parent
    raise RuntimeError("could not locate mat-agent repository root")


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


def registry_path(name: str, *, root: str | Path | None = None) -> Path:
    """Return `playground/registry/<name>.yaml` for this repository."""
    base = Path(root) if root else repo_root()
    return base / "playground" / "registry" / f"{name}.yaml"
