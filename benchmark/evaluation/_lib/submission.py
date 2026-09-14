"""Parse and validate a submission descriptor against the frozen contract.

A submission the evaluator will act on must satisfy every rule here. Anything
else raises ``SubmissionError`` and the evaluator refuses to score it.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FRAME_INDEX = REPO / "benchmark/data/discovery_frame_index.json"

ALLOWED_CKPT_ROOTS = ("runs", "workspace", "data/mace_mof_0/models")
FORBIDDEN_ROOTS = ("benchmark", "scoring", "docker", "tools")
ALLOWED_DTYPES = ("float32", "float64")
ALLOWED_INFERENCE_KINDS = ("mace_calculator", "entrypoint")
EXTERNAL_INPUT_FIELDS = {"kind", "id", "source_url", "license", "sha256", "used_for", "leakage_argument"}


class SubmissionError(ValueError):
    """Raised for any submission that violates the contract."""


def _under(path: Path, roots: tuple[str, ...]) -> bool:
    try:
        rel = path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return False  # outside the repo entirely -> not under any repo root
    return any(rel == r or rel.startswith(r + "/") for r in roots)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_external_inputs(path: Path) -> list[dict]:
    if not path.is_file():
        raise SubmissionError(f"external_inputs declared but not found: {path}")
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SubmissionError(f"external_inputs is not valid JSON: {exc}") from exc
    items = doc.get("inputs")
    if not isinstance(items, list):
        raise SubmissionError("external_inputs.inputs must be a list")
    for i, item in enumerate(items):
        missing = EXTERNAL_INPUT_FIELDS - set(item)
        if missing:
            raise SubmissionError(f"external_inputs.inputs[{i}] missing fields: {sorted(missing)}")
        if item["kind"] not in ("dataset", "pretrained_weights"):
            raise SubmissionError(f"external_inputs.inputs[{i}].kind invalid: {item['kind']!r}")
        if not str(item["leakage_argument"]).strip():
            raise SubmissionError(f"external_inputs.inputs[{i}].leakage_argument is empty")
    return items


def _check_training_data(path: Path) -> dict:
    """If the agent declares the exact frames it trained on, verify none are sequestered."""
    if not FRAME_INDEX.is_file():
        raise SubmissionError("frozen frame index missing; run benchmark/data/build_discovery_splits.py")
    idx = json.loads(FRAME_INDEX.read_text())
    dev_hashes = {row[3] for row in idx["dev_train"]} | {row[3] for row in idx["dev_eval"]}
    seq_hashes = set(idx["sequestered_ml_frame_sha256"])
    if not path.is_file():
        raise SubmissionError(f"training_data declared but not found: {path}")

    frame_hashes, blocks, _buf = [], 0, []
    lines = path.read_text().splitlines(keepends=True)
    i = 0
    while i < len(lines):
        na = int(lines[i].strip())
        raw = "".join(lines[i:i + na + 2])
        frame_hashes.append(hashlib.sha256(raw.encode()).hexdigest())
        i += na + 2
        blocks += 1
    leaked = seq_hashes.intersection(frame_hashes)
    if leaked:
        raise SubmissionError(
            f"training_data contains {len(leaked)} sequestered-ML frame(s) -- R2 violation"
        )
    outside = [h for h in frame_hashes if h not in dev_hashes]
    return {
        "declared_training_frames": blocks,
        "frames_matching_dev_index": blocks - len(outside),
        "frames_outside_dev_index": len(outside),
        "note": "frames outside the dev index are allowed only if covered by a declared external input (R4)",
    }


def load_and_validate(submission_path: str | Path, *, is_baseline: bool = False) -> dict:
    p = Path(submission_path)
    if not p.is_file():
        raise SubmissionError(f"submission file not found: {p}")
    try:
        sub = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise SubmissionError(f"submission is not valid JSON: {exc}") from exc

    if "checkpoint" not in sub:
        raise SubmissionError("submission.checkpoint is required")
    ckpt = (REPO / sub["checkpoint"]) if not Path(sub["checkpoint"]).is_absolute() else Path(sub["checkpoint"])
    if not ckpt.is_file():
        raise SubmissionError(f"checkpoint not found: {ckpt}")
    if _under(ckpt, FORBIDDEN_ROOTS):
        raise SubmissionError(f"checkpoint must not live under {FORBIDDEN_ROOTS}: {sub['checkpoint']}")
    if not is_baseline and not _under(ckpt, ALLOWED_CKPT_ROOTS):
        raise SubmissionError(f"checkpoint must be under one of {ALLOWED_CKPT_ROOTS}: {sub['checkpoint']}")

    inf = sub.get("inference")
    if not isinstance(inf, dict):
        raise SubmissionError("submission.inference must be an object")
    if inf.get("kind") not in ALLOWED_INFERENCE_KINDS:
        raise SubmissionError(f"inference.kind must be one of {ALLOWED_INFERENCE_KINDS}")
    if inf.get("dtype", "float64") not in ALLOWED_DTYPES:
        raise SubmissionError(f"inference.dtype must be one of {ALLOWED_DTYPES}")
    if inf["kind"] == "entrypoint":
        ep = inf.get("entrypoint", "")
        if not re.match(r"^[\w./-]+\.py:[A-Za-z_]\w*$", ep):
            raise SubmissionError("inference.entrypoint must be 'workspace/<file>.py:function'")
        mod_path = REPO / ep.split(":")[0]
        if not _under(mod_path, ("workspace",)):
            raise SubmissionError("inference.entrypoint module must live under workspace/")
        if not mod_path.is_file():
            raise SubmissionError(f"inference.entrypoint module not found: {mod_path}")

    if not is_baseline and not sub.get("training_recipe_entrypoint"):
        raise SubmissionError("training_recipe_entrypoint is required for the reproducibility check")

    external = []
    if sub.get("external_inputs"):
        ext_path = (REPO / sub["external_inputs"]) if not Path(sub["external_inputs"]).is_absolute() else Path(sub["external_inputs"])
        external = _validate_external_inputs(ext_path)

    training_data_report = None
    if sub.get("training_data"):
        td = (REPO / sub["training_data"]) if not Path(sub["training_data"]).is_absolute() else Path(sub["training_data"])
        training_data_report = _check_training_data(td)

    return {
        "raw": sub,
        "checkpoint_path": ckpt,
        "checkpoint_sha256": sha256_file(ckpt),
        "inference": inf,
        "external_inputs": external,
        "training_data_report": training_data_report,
        "is_baseline": is_baseline,
    }
