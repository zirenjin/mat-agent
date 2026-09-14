#!/usr/bin/env bash
# Establish the human baseline with the FROZEN evaluator.
#
# Run inside the evaluation Docker image (torch + mace-torch==0.3.9 + phonopy +
# spglib + ase). Needs the sequestered split built and ~1 GPU.
#
#   docker build --build-arg MAT_AGENT_INSTALL_MODE=full -f docker/Dockerfile -t mat-agent:full .
#   docker run --gpus all -v "$PWD":/workspace/mat-agent -w /workspace/mat-agent \
#       mat-agent:full bash benchmark/baselines/mace_mof0/run_baseline.sh
#
# Produces:
#   benchmark/baselines/mace_mof0/baseline_scores.json    <- BINDING baseline (mofs_v2)
#   benchmark/baselines/mace_mof0/baseline_context.json   <- foundation / v1 / v2 side by side
#   benchmark/evaluation/noise_estimate.json              <- measured tolerances
# then: python benchmark/integrity/verify_frozen.py --update && commit all three.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
SUB=benchmark/baselines/mace_mof0/submissions
OUT="${BASELINE_OUT_DIR:-runs/_baseline}"
mkdir -p "$OUT"

test -f data/mace_mof_0/discovery/_sequestered/sequestered_ml.xyz \
  || { echo "run: python benchmark/data/build_discovery_splits.py"; exit 2; }

# Resumable: skip a model whose scorecard is already present (long GPU job).
for name in foundation mofs_v1 mofs_v2; do
  if [ -f "$OUT/$name/scorecard.json" ]; then
    echo "=== $name already scored -> skip ($OUT/$name/scorecard.json)"
    continue
  fi
  echo "=== scoring $name ==="
  python benchmark/evaluation/evaluate_mof0.py score \
    --submission "$SUB/$name.json" --baseline --label "$name" \
    --out-dir "$OUT/$name" --stages ab
  echo "BASELINE_MODEL_DONE $name"
done

cp "$OUT/mofs_v2/scorecard.json" benchmark/baselines/mace_mof0/baseline_scores.json

python - <<'PY'
import json, pathlib
out = pathlib.Path("runs/_baseline")
ctx = {n: json.loads((out/n/"scorecard.json").read_text()) for n in ("foundation","mofs_v1","mofs_v2")}
pathlib.Path("benchmark/baselines/mace_mof0/baseline_context.json").write_text(
    json.dumps({"arms": {n: {"stage_a": c.get("stage_a"), "stage_b": c.get("stage_b")}
                         for n, c in ctx.items()},
                "binding_baseline": "mofs_v2",
                "selection_rule": "strongest published human model on the frozen evaluator; "
                                  "mofs_v2 supersedes mofs_v1 and the untuned foundation"},
               indent=2, sort_keys=True) + "\n")
print("wrote baseline_context.json")
PY

if [ ! -f benchmark/evaluation/noise_estimate.json ]; then
  echo "=== noise estimate (3x Stage B on mofs_v2) ==="
  python benchmark/evaluation/estimate_noise.py \
    --submission "$SUB/mofs_v2.json" --baseline --repeats 3 --stages b --out-dir "$OUT/_noise"
fi

echo
echo "BASELINE_RUN_OK  baseline_scores.json + baseline_context.json + noise_estimate.json written"
echo "NEXT: python benchmark/integrity/verify_frozen.py --update  &&  git add -A benchmark/ && git commit"
