#!/usr/bin/env bash
# SMOKE-track verifier wrapper. Identical scoring path to the standard
# track's benchmark/harbor/mace-mof0/tests/test.sh (same frozen evaluator,
# same scientific_score.py -- no separate scoring logic for smoke, so a
# smoke run's number means the same thing it would mean anywhere else) with
# two differences: it reads the mace_mof0_discovery_smoke compute ledger, and
# it additionally writes plumbing_status.json -- the objectively-checkable
# subset of the smoke success checklist (benchmark/harbor/mace-mof0-smoke/README.md).
# The scientific reward is NEVER gated by plumbing status here; they are
# reported side by side, never merged into one number.
#
# See benchmark/harbor/mace-mof0/tests/test.sh for the full commentary on the
# reward file path/schema and the sequestered-data fatal check -- unchanged
# here, not re-explained.
set -euo pipefail

REPO="${MAT_AGENT_ROOT:-/workspace/mat-agent}"
cd "$REPO"

LOG_DIR="${HARBOR_VERIFIER_LOG_DIR:-/logs/verifier}"
mkdir -p "$LOG_DIR"

PLUMBING="$LOG_DIR/plumbing_status.json"
plumbing_set() {
    # Merge one key=value (value must be valid JSON: true/false/a number/a quoted string).
    python3 -c "
import json, sys
path, key, value = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
try:
    doc = json.load(open(path))
except FileNotFoundError:
    doc = {}
doc[key] = value
json.dump(doc, open(path, 'w'), indent=2, sort_keys=True)
" "$PLUMBING" "$1" "$2"
}

write_reward() {
    local value="$1" reason="$2"
    python3 -c "import json,sys; json.dump({'reward': float(sys.argv[1])}, open(sys.argv[2], 'w'))" \
        "$value" "$LOG_DIR/reward.json"
    printf '%s\n' "$value" > "$LOG_DIR/reward.txt"
    printf '%s\n' "$reason" > "$LOG_DIR/reward_reason.txt"
    echo "[test.sh:smoke] reward=$value reason=$reason"
}

plumbing_set harbor_agent_started true
plumbing_set verifier_started true

SUBMISSION="runs/SUBMISSION/submission.json"
if [ ! -f "$SUBMISSION" ]; then
    plumbing_set submission_created false
    plumbing_set artifact_transferred false
    plumbing_set verifier_completed true
    plumbing_set reward_created true
    write_reward 0.0 "no submission at $SUBMISSION (smoke: non-gating, but submission_created=false is a real plumbing miss)"
    exit 0
fi
plumbing_set submission_created true
plumbing_set artifact_transferred true
cp "$SUBMISSION" "$LOG_DIR/submission.json"

SEQ_FILE="data/mace_mof_0/discovery/_sequestered/sequestered_ml.xyz"
if [ ! -f "$SEQ_FILE" ]; then
    echo "[test.sh:smoke] FATAL: $SEQ_FILE not present -- see the standard track's test.sh for why this is fatal, not a soft skip." >&2
    plumbing_set verifier_completed false
    write_reward 0.0 "verifier environment missing sequestered data -- infrastructure fault, not a real score"
    exit 1
fi

echo "[test.sh:smoke] preflight..."
if ! python3 benchmark/integrity/preflight.py --submission "$SUBMISSION" --json > "$LOG_DIR/preflight.json"; then
    plumbing_set verifier_completed true
    plumbing_set reward_created true
    write_reward 0.0 "preflight rejected the submission (see $LOG_DIR/preflight.json) -- smoke: non-gating"
    exit 0
fi

echo "[test.sh:smoke] scoring..."
EVAL_OUT="runs/SUBMISSION/eval"
BASELINE_SCORECARD="benchmark/baselines/mace_mof0/baseline_scores.json"
# KNOWN, PRE-EXISTING GAP (AUTONOMOUS_RESEARCH_READY.md blockers B1/B3, not
# introduced by this task): this file has never been generated -- it needs a
# real, never-yet-run, multi-hour Stage-B (phonopy) pass on a GPU, which is
# out of scope for a smoke attempt. Score WITHOUT --compare-to-baseline when
# it's missing, so a smoke run still validates preflight+Stage-A/B plumbing
# and produces a real scorecard.json, instead of hard-failing on a data-
# availability gap that has nothing to do with whether Harbor/broker/Volc
# plumbing worked.
COMPARE_ARGS=()
if [ -f "$BASELINE_SCORECARD" ]; then
    COMPARE_ARGS=(--compare-to-baseline --baseline-scorecard "$BASELINE_SCORECARD")
else
    echo "[test.sh:smoke] WARNING: $BASELINE_SCORECARD missing (never generated -- see AUTONOMOUS_RESEARCH_READY.md B1/B3)." >&2
    echo "  Scoring without a baseline comparison: scorecard.json will exist, verdict.json will not." >&2
    plumbing_set baseline_scorecard_available false
fi
if [ -f "$BASELINE_SCORECARD" ]; then
    plumbing_set baseline_scorecard_available true
fi

if ! python3 benchmark/evaluation/evaluate_mof0.py score \
        --submission "$SUBMISSION" \
        --out-dir "$EVAL_OUT" \
        --stages ab \
        "${COMPARE_ARGS[@]}" \
        > "$LOG_DIR/evaluate_mof0.stdout" 2>&1; then
    cp "$LOG_DIR/evaluate_mof0.stdout" "$LOG_DIR/evaluate_mof0.error.log" 2>/dev/null || true
    plumbing_set verifier_completed true
    plumbing_set reward_created true
    write_reward 0.0 "evaluator run failed (see $LOG_DIR/evaluate_mof0.stdout) -- smoke: non-gating, but worth reading why"
    exit 0
fi

for f in scorecard.json verdict.json; do
    [ -f "$EVAL_OUT/$f" ] && cp "$EVAL_OUT/$f" "$LOG_DIR/$f"
done

python3 benchmark/compute/account_compute.py --task mace_mof0_discovery_smoke --json \
    > "$LOG_DIR/compute_ledger.json" 2>/dev/null || true

plumbing_set verifier_completed true
plumbing_set reward_created true

if [ -f "$EVAL_OUT/verdict.json" ]; then
    python3 benchmark/evaluation/scientific_score.py --verdict "$EVAL_OUT/verdict.json" --json \
        > "$LOG_DIR/scientific_score.json"
    SCORE="$(python3 -c "import json; print(json.load(open('$LOG_DIR/scientific_score.json'))['scientific_score'])")"
    BEATS="$(python3 -c "import json; print(json.load(open('$LOG_DIR/scientific_score.json'))['beats_baseline'])")"
    write_reward "$SCORE" "scientific_score=$SCORE beats_baseline=$BEATS -- smoke: this is recorded, NOT required to be high"
else
    write_reward 0.0 "scorecard.json produced but no verdict.json (no baseline_scores.json to compare against -- see baseline_scorecard_available in $PLUMBING). scorecard.json itself is real; read it directly for smoke purposes."
fi
