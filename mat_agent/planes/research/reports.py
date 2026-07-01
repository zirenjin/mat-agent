#!/usr/bin/env python3
"""Report writing helpers."""
from __future__ import annotations

from pathlib import Path


def write_markdown_report(path: str | Path, title: str, sections: dict[str, str]) -> None:
    """Write a small Markdown report from named sections."""
    lines = [f"# {title}", ""]
    for name, text in sections.items():
        lines.extend([f"## {name}", "", text.strip(), ""])
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines).rstrip() + "\n")
