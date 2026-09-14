# MACE-MP-MOF0 Task

Master overview tying together the three pieces of this benchmark that
otherwise live in separate files: the historical reproduction record, the
open-ended discovery task, and the Harbor autonomous-agent integration. Each
section below is a summary with a pointer to its authoritative source —
this file restates nothing that would drift out of sync with those.

## Reproduction (historical)

Unchanged, kept as evidence, not rewritten: `docs/task_cards/mace_mof0_reproduction_task_card.md`
is the scripted walkthrough; `benchmark/baselines/mace_mof0/reproduced_baseline_manifest.json`
records the actual completed run
(`mof0_paper_reproduce_force_mh_lr_20260829T124117Z`) this Task.md's
Experiment section below builds on — checkpoint/config SHA256, exact
training config, and the metrics showing it reproduces (slightly exceeds)
the published recipe on the released test split.

## Experiment

### Research Question

> Given a strong reproduced MACE-MP-MOF0 baseline, a fixed dataset/evaluation
> protocol, and a fixed scientific-compute budget, can an autonomous research
> agent discover a training or adaptation strategy that improves held-out MOF
> performance while preserving broad physical-property accuracy?

The agent is not asked to reproduce the paper — that is already done (see
Reproduction above) and frozen as the starting point. It is asked to find an
improvement on top of a working, honestly-benchmarked baseline, under a
compute budget that rules out "just train longer/bigger."

### Baseline and Warm Start

One checkpoint serves both roles, deliberately: `data/mace_mof_0/baseline/mof0_reproduced_baseline.model`
(in a Harbor agent container) / `data/mace_mof_0/runs/mof0_paper_reproduce_force_mh_lr_20260829T124117Z/models/mof0_paper_reproduce.model`
(in this repo checkout) — this repo's own from-scratch reproduction, not the
externally-published `mofs_v2.model`. Full provenance, hashes, and the
reproduced-vs-published comparison:
`benchmark/baselines/mace_mof0/reproduced_baseline_manifest.json`.
Warm-starting from it is the *expected default* (re-deriving it from the raw
foundation on every attempt wastes budget on reproduction, not research);
starting from the raw foundation `mace_agnesi_medium.model` instead is still
permitted when a hypothesis specifically calls for it. Read-only in the
agent's container (`chmod 444`, defense-in-depth, not absolute — see the
manifest's `immutability_enforcement`). Full contract:
`tasks/mace_mof0_discovery/task_card.md` §2.

### Agent Action Space

Unchanged from the pre-Harbor discovery task design — already broad and
already un-menu'd: `tasks/mace_mof0_discovery/task_card.md` §3 ("What is
open"). Architecture, message-passing, readouts, loss formulation/weighting,
optimizer, LR schedule, staged training, regularization, curriculum,
multi-head/adapter strategies, replay/rehearsal mixture, data
weighting/augmentation within the allowed pool, ensembling, target choice —
no required ablation list, no suggested method ranking, no hypothesis
steered by the instructions. `benchmark/harbor/mace-mof0/instruction.md`
(the Harbor-specific operational notes shown to the agent) deliberately adds
no method guidance on top of this.

### Frozen Benchmark Components

Unchanged and non-editable by the agent (enforced by
`benchmark/integrity/verify_frozen.py`, not just documentation):
`benchmark/evaluation/**`, `scoring/**`, compute-policy implementation
(`benchmark/compute/account_compute.py`), integrity manifests, the
reproduced-baseline checkpoint + its training config (see the manifest's
`provenance_rules`), hidden/sequestered data definitions, and — new for the
Harbor integration — the Harbor task infrastructure itself (`task.toml` is
now frozen too), the compute broker (`environment/broker/`), and the
verifier's separate environment. Full list: `tasks/mace_mof0_discovery/task_card.md`
§2 and §10; architecture: `benchmark/harbor/README.md`.

### Objective and Evaluation

The frozen evaluator (`benchmark/evaluation/evaluate_mof0.py` +
`_lib/aggregate.py`) produces a **hard-gated verdict**
(`beats_baseline` = physics-primary AND no-ML-regression AND reproducible
AND compute-legal — task_card.md §6), not a single scalar. It never did
produce one; `benchmark/evaluation/scientific_score.py` is a new, thin,
additional layer (no new metric, no changed scoring logic in the frozen
files) deriving exactly one scalar from that same verdict for Harbor's
reward interface: `0.0` unless `beats_baseline` is true, otherwise the mean
per-MOF physics-improvement margin. See that file's docstring for why this
satisfies baseline-relative normalization, no trivial cross-metric
compensation, and no reliance on non-reproducible metrics. This is what
`benchmark/harbor/mace-mof0/tests/test.sh` writes to
`/logs/verifier/reward.json`.

**Known gap:** `benchmark/baselines/mace_mof0/baseline_scores.json` (the
frozen evaluator's own scorecard for the baseline, needed to actually
compute `beats_baseline`) has never been generated — it requires a real,
never-yet-executed, multi-hour Stage-B (phonopy) GPU pass
(`AUTONOMOUS_RESEARCH_READY.md` blockers B1/B3). Until it exists, `test.sh`
scores without a baseline comparison (real `scorecard.json`, no
`verdict.json`, reward `0.0`) rather than failing outright — see that
script's comments and `benchmark/harbor/mace-mof0-smoke/README.md`'s "Known
gap" section.

### Smoke Track

Validates plumbing (Harbor -> Claude Code -> compute broker -> Volc -> back
-> verifier), not science. `benchmark/harbor/mace-mof0-smoke/`
(`README.md` for the full plumbing-success checklist and known gaps).
Budget: `benchmark/compute/policy.yaml` → `tasks.mace_mof0_discovery_smoke`
— 90 min agent wall-clock, ~2 A100-eq GPU-hours total, 1 GPU concurrency, 1
scored evaluation. Scientific improvement is explicitly non-gating.

### Standard Track

`benchmark/harbor/mace-mof0/` (task.toml: 16h agent wall-clock).
Compute-policy numbers (`benchmark/compute/policy.yaml` →
`tasks.mace_mof0_discovery`: 50 total A100-h, 12/experiment, 4-GPU
concurrency) are **TBD/proposed, not calibrated** — kept only as
placeholders from the pre-warm-start design, explicitly not preserved for
false consistency. The one real data point: the full reproduction run
alone cost ~127 GPU-hours (4 GPUs × ~31.7h wall-clock) — more than this
entire total budget — but a standard-track attempt no longer needs to repeat
that (it warm-starts from the checkpoint instead); what a meaningful
*adaptation* run from that warm start costs has not yet been measured.
Calibrate from real warm-start fine-tune timing before treating a
standard-track attempt as properly budgeted — see `policy.yaml`'s comment on
this entry for the exact figures and caveat.
