# MACE-MP-MOF0 Discovery — SMOKE track

Validates the autonomous-research plumbing (Harbor -> Claude Code -> compute
broker -> Volc -> back -> Claude Code -> verifier) end-to-end on a short,
cheap real trajectory. See `instruction.md` for what the agent is told, and
`benchmark/harbor/mace-mof0/README.md` for everything about the shared
architecture (compute broker, bind mounts, hidden-data isolation) that
applies identically here — this file only covers what's smoke-specific.

Reuses the standard track's images unchanged (`mat-agent-harbor:latest`,
`mat-agent-harbor-verifier:latest`, and the broker built from
`benchmark/harbor/mace-mof0/environment/broker/`) — build via
`benchmark/harbor/mace-mof0/environment/build.sh`; there is no separate
build script here. Credentials: same
`benchmark/harbor/mace-mof0/export_volc_env.sh`.

```bash
bash benchmark/harbor/mace-mof0/environment/build.sh
eval "$(bash benchmark/harbor/mace-mof0/export_volc_env.sh)"
harbor run -p benchmark/harbor/mace-mof0-smoke -a claude-code -m <model>
```

## Budget

`benchmark/compute/policy.yaml`, `tasks.mace_mof0_discovery_smoke` — a
completely separate ledger from the standard track's `mace_mof0_discovery`
(different `--task` key everywhere: `account_compute.py`,
`tools/remote_job.py`'s concurrency gate). Nothing spent here counts against,
or is constrained by, the standard track's 50 A100-hour budget.

| | |
|---|---|
| Agent wall-clock | 90 min (`task.toml` `[agent].timeout_sec = 5400`) |
| Total compute | ~2 A100-equivalent GPU-hours |
| Per-experiment cap | ~1.5 A100-equivalent GPU-hours |
| Concurrency | 1 A100-equivalent GPU |
| Scored hidden evaluations | 1 (the final verifier run) |
| Scientific improvement | **non-gating** — recorded, not required |

## Plumbing success checklist

The scientific score (`scientific_score.json`, `reward.json`) is a completely
separate signal from the checklist below — never conflate them. A smoke
attempt with `scientific_score = 0.0` and every box below checked is a full
success; the reverse (an accidentally-good score with boxes unchecked) is not.

Objectively checkable from artifacts alone (`tests/test.sh` writes these to
`/logs/verifier/plumbing_status.json`):

- [ ] `harbor_agent_started`
- [ ] `submission_created`
- [ ] `artifact_transferred`
- [ ] `verifier_started`
- [ ] `verifier_completed`
- [ ] `reward_created`
- [ ] `baseline_scorecard_available` (informational — see "Known gap" below;
      `false` here does not itself indicate a smoke failure)

Requires reading the agent's own transcript/trajectory (no test.sh-visible
artifact proves these; fill in by hand when writing up a run):

- [ ] `remote_job_1_submitted`
- [ ] `remote_job_1_completed`
- [ ] `result_1_observed_by_agent`
- [ ] `second_decision_conditioned_on_result_1` — **the critical one**: did
      the agent's hypothesis/decision B actually change because of what
      experiment A returned, or was it a second idea it always intended to
      run regardless of A's outcome? A trajectory can satisfy every other
      box and still fail this one.
- [ ] `remote_job_2_submitted_or_explicitly_skipped_for_valid_reason`

## Known gap: `baseline_scores.json`

`benchmark/baselines/mace_mof0/baseline_scores.json` (the frozen evaluator's
Stage A/B scorecard for the baseline checkpoint) has never been generated in
this repo — it needs a real, never-yet-executed, multi-hour Stage-B
(relax + phonopy across 4 MOFs) GPU pass
(`AUTONOMOUS_RESEARCH_READY.md` blockers B1/B3), which is out of scope for a
smoke attempt to also be the first one to run. `tests/test.sh` in this track
(and, now, the standard track's) detects this and scores WITHOUT
`--compare-to-baseline` when the file is absent, so a smoke run still
produces a real `scorecard.json` and exercises the full preflight -> Stage A
-> Stage B -> reward path; only `verdict.json` and hence a nonzero
`scientific_score` require the missing file. This is a real, pre-existing
gap to close before any run's `beats_baseline`/`scientific_score` can mean
anything — track it separately from smoke plumbing success.
