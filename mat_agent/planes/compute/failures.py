#!/usr/bin/env python3
"""Runtime failure parsing affordances."""
from __future__ import annotations

from pathlib import Path
import re

PATTERNS = {
    "OOM": r"cuda[^\n]*(out of memory|oom)|outofmemory|cublas.*alloc",
    "NUMERICAL_NAN": r"\bnan\b|\binf\b|non[- ]?finite|floating point exception",
    "DIVERGENCE": r"loss[^\n]*(diverg|explod)|gradient[^\n]*(explod|overflow)",
}


def classify_log(path: str | Path) -> dict[str, object]:
    """Classify common ML runtime failures from a text log."""
    p = Path(path)
    text = p.read_text(errors="replace") if p.exists() else ""
    hits = [name for name, pattern in PATTERNS.items() if re.search(pattern, text, flags=re.I)]
    return {"path": str(path), "status": "failed" if hits else "unknown_or_ok", "signals": hits}
