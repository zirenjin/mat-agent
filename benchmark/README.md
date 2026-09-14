# benchmark/

Immutable benchmark infrastructure for the `mat-agent` autonomous ML research
tasks. Nothing in this directory is part of the agent's editable surface. It is
the "rules of the game": the agent may read it, must obey it, and must not change
it.

## Layout

```text
benchmark/
  data/          frozen discovery split definition: manifest + per-frame hash index
                 (builder: benchmark/data/build_discovery_splits.py; xyz files land
                  under data/mace_mof_0/discovery/, gitignored)
  integrity/     frozen-file manifest + verifier, integrity rules, preflight gate
  compute/       compute policy, compute accounting, run-manifest writer
  evaluation/    the single evaluator: protocol, runner (evaluate_mof0.py),
                 _lib/ (submission validation, sandbox, Stage A/B, release adapters,
                 aggregation, noise), output schema, noise estimator
  baselines/     frozen human-baseline descriptors, run script, scorecard
```

## Run order to reach "ready"

1. `python benchmark/data/build_discovery_splits.py` — needs the ddmms release under `data/mace_mof_0/`.
2. `bash benchmark/baselines/mace_mof0/run_baseline.sh` — inside the eval Docker image, on GPU. Produces `baseline_scores.json`, `baseline_context.json`, `noise_estimate.json`.
3. `python benchmark/integrity/verify_frozen.py --update` — pin the new baseline/noise files; commit.
4. `python benchmark/integrity/preflight.py` — must print `preflight OK`.

## Immutability tiers (whole repo)

| Tier | Meaning | Examples |
| --- | --- | --- |
| **editable** | agent may create/modify freely | `workspace/`, `runs/`, agent-authored configs |
| **readable / immutable** | agent reads, CI rejects changes | `benchmark/**`, `scoring/**`, `data/mace_mof_0/manifest.json`, `data/registry.yaml`, `docker/**`, infra tools in the frozen manifest, `tasks/*/task_card.md` |
| **hidden / evaluator-only** | agent never sees the labels; evaluator holds them | sequestered ML holdout, sequestered physics/DFT reference values, baseline scorecards' provenance |

The boundary between "editable" and "readable/immutable" is enforced by
`benchmark/integrity/verify_frozen.py` (run in CI and in task preflight). The
boundary to "hidden" is enforced by *process*: the evaluator runs scoring; the
agent only ever submits a checkpoint + inference entrypoint.

## Key design decision

The evaluator consumes **only a submitted checkpoint and a declared inference
function**, never the agent's training code. This is what lets the task leave
architecture / loss / optimizer / fine-tuning / replay completely open while
keeping the evaluator, data truth, and baseline outside the agent's reach.

## Adding a new task

1. `tasks/<task_id>/task_card.md` — constrain the *question*, not the *solution*.
2. `benchmark/evaluation/` entry for the task's protocol + runner.
3. `benchmark/baselines/<task_id>/` — frozen baseline artifact + scorecard.
4. `benchmark/compute/policy.yaml` — a stanza for the task's budget.
5. Regenerate `benchmark/integrity/frozen_manifest.json`
   (`python benchmark/integrity/verify_frozen.py --update`).
