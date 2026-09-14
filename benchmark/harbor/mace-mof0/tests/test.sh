#!/usr/bin/env bash
# Harbor verifier wrapper. Deliberately thin: it validates the submission,
# runs the frozen evaluator, and translates its verdict into the machine
# interface Harbor's runtime reads -- it re-implements no physics metric and
# makes no scientific judgment of its own. The full scorecard (not just the
# scalar reward) is preserved under /logs/verifier/ as the primary scientific
# result; the scalar is only the harness-required interface.
#
# Reward path/schema confirmed by reading harbor==0.23.0's actual installed
# source directly (harbor/verifier/verifier.py + harbor/models/trial/paths.py),
# not documentation: /logs/verifier/reward.json is read first if present
# (bare top-level JSON number, OR a {"key": number, ...} dict for multiple
# named rewards -- every value must be a finite int/float or the whole file
# is rejected, so no extra non-numeric keys); /logs/verifier/reward.txt (a
# single bare number) is the fallback. This script writes both, in the
# single-scalar form: benchmark/evaluation/scientific_score.py's
# baseline-relative, hard-gated scalar (0.0 unless verdict.beats_baseline is
# true; otherwise the mean per-MOF improvement margin) -- see that file for
# why it is not just beats_baseline as a 1.0/0.0 flag. The full
# scorecard/verdict/reason, which are NOT numeric, are written as separate
# files alongside, never inside the reward file itself.
#
# data/mace_mof_0/discovery/_sequestered/** and data/mace_mof_0/raw/phonons/**
# being available here depends on task.toml's [verifier].environment_mode
# actually being honored as "separate" with [verifier.environment] pointing
# at mat-agent-harbor-verifier:latest (see environment/Dockerfile.verifier) --
# that field's *meaning* is confirmed from harbor/models/task/config.py and
# harbor/models/task/verifier_mode.py source, but this task has not yet been
# run end-to-end, so whether the wiring is actually correct in practice is
# not yet confirmed. If those paths are missing here, this script fails
# loudly rather than silently scoring nothing.
set -euo pipefail

REPO="${MAT_AGENT_ROOT:-/workspace/mat-agent}"
cd "$REPO"

LOG_DIR="${HARBOR_VERIFIER_LOG_DIR:-/logs/verifier}"
mkdir -p "$LOG_DIR"

write_reward() {
    # reward.json's values (or reward.txt's single value) must ALL be finite
    # numbers -- Harbor's verifier rejects the whole file otherwise (confirmed
    # in verifier/verifier.py's _parse_reward_json). The human-readable reason
    # goes to stdout and to reward_reason.txt, never inside the reward file.
    local value="$1" reason="$2"
    python3 -c "import json,sys; json.dump({'reward': float(sys.argv[1])}, open(sys.argv[2], 'w'))" \
        "$value" "$LOG_DIR/reward.json"
    printf '%s\n' "$value" > "$LOG_DIR/reward.txt"
    printf '%s\n' "$reason" > "$LOG_DIR/reward_reason.txt"
    echo "[test.sh] reward=$value reason=$reason"
}

SUBMISSION="runs/SUBMISSION/submission.json"
if [ ! -f "$SUBMISSION" ]; then
    write_reward 0.0 "no submission at $SUBMISSION"
    exit 0
fi
cp "$SUBMISSION" "$LOG_DIR/submission.json"

SEQ_FILE="data/mace_mof_0/discovery/_sequestered/sequestered_ml.xyz"
if [ ! -f "$SEQ_FILE" ]; then
    echo "[test.sh] FATAL: $SEQ_FILE not present in this environment -- the verifier" >&2
    echo "  cannot score without the sequestered split. This means either" >&2
    echo "  [verifier].environment_mode in task.toml is not actually giving the" >&2
    echo "  verifier stage a different/hidden-data-bearing environment from the" >&2
    echo "  agent, or the verifier image build did not include it. Fix the Harbor" >&2
    echo "  environment wiring -- do not relax this check to 'succeed anyway'." >&2
    write_reward 0.0 "verifier environment missing sequestered data -- infrastructure fault, not a real score"
    exit 1
fi

echo "[test.sh] preflight..."
if ! python3 benchmark/integrity/preflight.py --submission "$SUBMISSION" --json > "$LOG_DIR/preflight.json"; then
    write_reward 0.0 "preflight rejected the submission (see $LOG_DIR/preflight.json)"
    exit 0
fi

echo "[test.sh] scoring..."
EVAL_OUT="runs/SUBMISSION/eval"
BASELINE_SCORECARD="benchmark/baselines/mace_mof0/baseline_scores.json"
# KNOWN, PRE-EXISTING GAP (AUTONOMOUS_RESEARCH_READY.md blockers B1/B3): this
# file has never been generated (needs a real, never-yet-run Stage-B/phonopy
# pass on GPU). Score WITHOUT --compare-to-baseline when it's missing rather
# than hard-failing -- a real scorecard.json still means something even with
# no verdict.json, and a candidate should not be scored 0.0 for a data-
# availability gap that predates it.
COMPARE_ARGS=()
if [ -f "$BASELINE_SCORECARD" ]; then
    COMPARE_ARGS=(--compare-to-baseline --baseline-scorecard "$BASELINE_SCORECARD")
else
    echo "[test.sh] WARNING: $BASELINE_SCORECARD missing (never generated -- see AUTONOMOUS_RESEARCH_READY.md B1/B3). Scoring without a baseline comparison." >&2
fi
if ! python3 benchmark/evaluation/evaluate_mof0.py score \
        --submission "$SUBMISSION" \
        --out-dir "$EVAL_OUT" \
        --stages ab \
        "${COMPARE_ARGS[@]}" > "$LOG_DIR/evaluate_mof0.stdout" 2>&1; then
    cp "$LOG_DIR/evaluate_mof0.stdout" "$LOG_DIR/evaluate_mof0.error.log" 2>/dev/null || true
    write_reward 0.0 "evaluator run failed (see $LOG_DIR/evaluate_mof0.stdout)"
    exit 0
fi

for f in scorecard.json verdict.json; do
    [ -f "$EVAL_OUT/$f" ] && cp "$EVAL_OUT/$f" "$LOG_DIR/$f"
done

python3 benchmark/compute/account_compute.py --task mace_mof0_discovery --json \
    > "$LOG_DIR/compute_ledger.json" 2>/dev/null || true

if [ -f "$EVAL_OUT/verdict.json" ]; then
    python3 benchmark/evaluation/scientific_score.py --verdict "$EVAL_OUT/verdict.json" --json \
        > "$LOG_DIR/scientific_score.json"
    SCORE="$(python3 -c "import json; print(json.load(open('$LOG_DIR/scientific_score.json'))['scientific_score'])")"
    BEATS="$(python3 -c "import json; print(json.load(open('$LOG_DIR/scientific_score.json'))['beats_baseline'])")"
    write_reward "$SCORE" "scientific_score=$SCORE beats_baseline=$BEATS (see $LOG_DIR/scientific_score.json, verdict.json, scorecard.json)"
else
    write_reward 0.0 "scorecard.json produced but no verdict.json (no baseline_scores.json to compare against). scorecard.json itself is real; read it directly."
fi
