# MACE-MP-MOF0 Autonomous Discovery — Harbor task

You are running as an autonomous research agent inside a Harbor task container
for the `mat-agent` repository, checked out at `/workspace/mat-agent`.

**Your actual research assignment is `tasks/mace_mof0_discovery/task_card.md`
in that checkout — read it now, in full, before doing anything else.** It is
the frozen, authoritative statement of the scientific problem, what you may and
may not modify, the data/compute/evaluation contract, and what "beating the
baseline" means. This file only adds the operational notes specific to running
inside this Harbor container; it does not restate or override the task card.

The reproduced-baseline checkpoint the task card's §2 tells you to warm-start
from by default is at
`/workspace/mat-agent/data/mace_mof_0/baseline/mof0_reproduced_baseline.model`
in this container, read-only (`chmod 444`) — see
`benchmark/baselines/mace_mof0/reproduced_baseline_manifest.json` for its full
provenance and the metrics that show it reproduces the published recipe.

## How training actually runs

GPU training does **not** happen inside this container. This container is
where you reason, edit `workspace/`, and orchestrate — actual training and
inference jobs are submitted to Volcengine ML Platform (`volc ml_task`) and run
remotely on GPU nodes, the same way they always have in this repo:

- **You do not have the `volc` CLI or Volcengine credentials in this
  container, and that is not a mistake.** They live only in a separate
  `broker` service you cannot reach except through `tools/remote_job.py`
  (which auto-detects it via `REMOTE_JOB_BROKER_URL` and talks to it over
  HTTP) or `tools/task_loop_db.py`. Do not go looking for a way to call
  `volc` directly -- there is nothing installed to call, on purpose: it is
  the mechanism that makes the concurrency/budget cap below a real limit
  instead of a rule you're just being asked to follow.
- Existing scaffolding is unchanged and still yours to read/use:
  `volc/task-configs/mace_mof0/*.yaml` (as templates -- see below),
  `tools/task_loop_db.py`, `playground/registry/*.yaml`,
  `playground/loop_db/task_loops.sqlite`.
- **Submit exclusively through `tools/remote_job.py`:**
  ```
  python tools/remote_job.py submit --conf workspace/my_run.yaml \
      --gpu-model A100-40GB --gpu-count 4 --run-id <run_id>
  python tools/remote_job.py wait --job-id <job_id> --poll-seconds 120
  python tools/remote_job.py collect --job-id <job_id> --output runs/<run_id>
  python tools/remote_job.py list-active   # see what's currently counted against the cap
  ```
  It refuses a submission outright if it would exceed the concurrency cap,
  rather than letting you find out after the fact -- and unlike in earlier
  versions of this task, that refusal is enforced by the broker (which owns
  the only copy of the credentials), not just by this local wrapper.
- **The config you pass to `--conf` must be under `workspace/`, not
  `volc/task-configs/`.** Copy a template from `volc/task-configs/mace_mof0/`
  into `workspace/` and edit it there (or author a new one from scratch) before
  submitting. This isn't a style preference: `workspace/` and `runs/` are the
  only two directories shared between this container and the separate broker
  container that actually reads the file and talks to Volcengine --
  `volc/task-configs/` as it exists in this container is invisible to the broker.
- This container has normal outbound network access for everything else
  (research, docs, package installs) -- only the Volc path is walled off.
- `runs/<run_id>/run_manifest.json` and the checkpoint a submitted job writes
  really do just appear under your own `runs/` once the job reaches that
  point -- `runs/` is bind-mounted from the same host path the job's remote
  GPU node writes to, so there is nothing to separately fetch for those. Use
  `python tools/remote_job.py wait`/`status` to know when a job has reached a
  terminal state before you go looking.

## Budget

All figures are in `benchmark/compute/policy.yaml` (`tasks.mace_mof0_discovery`)
and are the single source of truth if anything here ever drifts from it:

- **Agent wall-clock: 16h** (`[agent].timeout_sec = 57600` in `task.toml`).
  This is your real-time ceiling across every iteration combined, not a
  compute quantity — treat it as the outer bound on the hypothesis ->
  experiment -> analysis -> iterate loop from the task card §8. There is no
  mid-run checkpoint/resume: everything you want kept must land under
  `runs/`, `workspace/`, or `playground/` before the container is torn down.
- **Total compute: 50 A100-equivalent GPU-hours** for the whole task.
- **Per-experiment cap: 12 A100-equivalent GPU-hours** (soft cap; above it
  needs a one-line justification in the run manifest and still counts).
- **Concurrency: 4 A100-equivalent GPUs at a time**, enforced live by the
  `broker` service `tools/remote_job.py submit` talks to (see above) — a real
  boundary you cannot bypass by finding some other way to call `volc`,
  because there is no other way to call `volc` from where you're running.
- **Scored hidden evaluations: 3 total**, including the one the verifier runs
  against your final `runs/SUBMISSION/submission.json`. Self-evaluation on
  `dev_eval.xyz` is unlimited and doesn't count; spend the other 2 deliberately.
- Verifier/grading compute does not count against your 50h.

## Credentials

You don't have any, and won't get any by any supported path -- see "How
training actually runs" above. The Volc AK/SK live only inside the separate
`broker` container, reconstructed there from environment variables at its
own startup (`environment/entrypoint.sh`). Submitting/polling/collecting
Volc jobs goes through `tools/remote_job.py`, which forwards your requests
to that broker over the network; it never hands you the credentials
themselves.

## What you cannot see, and why `preflight.py` is not yours to run

This container never contains `data/mace_mof_0/discovery/_sequestered/**` or
`data/mace_mof_0/raw/phonons/**` (sequestered-ML and sequestered-physics, per
the task card's data-contract table) -- not "don't read them," they are
structurally absent from this image. `data/mace_mof_0/external/` (unrelated
upstream download staging) is also absent. Everything else the data-contract
table marks as yours is present.

One consequence: `benchmark/integrity/preflight.py` checks that the
sequestered file exists at a fixed path, because it was written for the
benchmark's original single-environment design where the agent and the
evaluator shared a filesystem. In this Harbor task they don't, so running
`preflight.py` yourself will fail on that check for a structural reason, not
because anything is wrong with your submission. Don't try to work around it
and don't treat that failure as meaningful. To sanity-check your submission
descriptor's shape yourself (this part does not need sequestered data), call
the same validator the evaluator uses directly:
```
python3 -c "
import sys; sys.path.insert(0, '.')
from benchmark.evaluation._lib.submission import load_and_validate
load_and_validate('runs/SUBMISSION/submission.json')
print('shape OK')
"
```
Leave `preflight.py`'s full gate and the actual scoring run to the verifier
stage, which runs separately after your container exits and does have the
sequestered data.

## What to hand back

Exactly what the task card's §9 ("Deliverables") asks for, culminating in
`runs/SUBMISSION/submission.json` per `benchmark/evaluation/protocol.md`. The
grading step for this Harbor task runs the same frozen evaluator the task card
describes against whatever is at `runs/SUBMISSION/submission.json` when your
run ends — nothing else is scored, and running the evaluator yourself for a
reported number is still out of contract per the task card.
