#!/usr/bin/env python3
"""Path helpers for the vendored Matbench Discovery source tree."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from mat_agent.registry import repo_root


@dataclass(frozen=True)
class MatbenchDiscoveryPaths:
    """Resolved paths for the optional upstream Matbench Discovery checkout."""

    root: Path

    @property
    def package_dir(self) -> Path:
        return self.root / "matbench_discovery"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    def require(self) -> "MatbenchDiscoveryPaths":
        missing = [path for path in [self.package_dir, self.data_dir] if not path.exists()]
        if missing:
            joined = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Matbench Discovery source is incomplete: {joined}")
        return self


def default_paths(root: str | Path | None = None) -> MatbenchDiscoveryPaths:
    """Return the default vendored Matbench Discovery paths for this repo."""
    base = Path(root) if root else repo_root()
    return MatbenchDiscoveryPaths(base / "third_party" / "matbench-discovery")
