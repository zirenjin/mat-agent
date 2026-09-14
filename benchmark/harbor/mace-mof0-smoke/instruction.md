# MACE-MP-MOF0 Autonomous Discovery — SMOKE track

This is the **smoke track**: a short, cheap, real run of the same autonomous
research loop as the standard `mace_mof0_discovery` Harbor task, run here to
validate the plumbing (you, the compute broker, Volcengine, and the
verifier all actually work together), not to produce a publishable result.

**Read `tasks/mace_mof0_discovery/task_card.md` first, in full**, exactly as
you would for the standard track — the scientific problem, data contract,
and submission format are identical. Everything below only changes the
*scale* and *how success is judged for this run specifically*.

## What is different from the standard track

- **Wall-clock: 90 minutes** (`[agent].timeout_sec = 5400`), not 16 hours.
- **Compute: ~2 A100-equivalent GPU-hours total**, ~1.5 per experiment, 1 GPU
  at a time (`benchmark/compute/policy.yaml`'s `mace_mof0_discovery_smoke`
  entry — a completely separate ledger from the standard track's, so nothing
  you do here touches that budget, and nothing there constrains you here).
- **You are not expected to beat the baseline.** Scientific improvement
  (`beats_baseline`, `scientific_score`) is recorded but is explicitly
  **non-gating** for this run — a smoke attempt that produces a valid,
  honestly-reported submission with a low or zero score is a full success.
  What matters is whether the loop below actually happened.
- **Use a tiny slice of the problem on purpose**: a small subset of
  `dev_train.xyz`, a drastically shortened training budget (far fewer epochs
  than any real fine-tune), so a real remote experiment finishes in minutes,
  not hours. The resulting checkpoint is expected to be scientifically weak —
  that is fine and correct for this track.

## The one thing that actually matters here

Run a **real** two-step research loop, not a single predetermined submission:

1. Read the baseline (`data/mace_mof_0/baseline/mof0_reproduced_baseline.model`
   — see task card §2) and the training pipeline.
2. Form hypothesis A (anything small and fast to test — this is a plumbing
   check, not where your best idea needs to go).
3. Implement it in `workspace/`, submit a real, tiny job:
   `python tools/remote_job.py submit --conf workspace/<your-config>.yaml --gpu-model <model> --gpu-count 1 --run-id <run_id>`
4. **Block on it with `python tools/remote_job.py wait --job-id <job_id>`
   and let that one call do the waiting.** Do not repeatedly call `status`
   in a manual polling loop across many of your own turns — `wait` already
   polls internally and returns once the job reaches a terminal state (or
   times out); spending your own turns re-checking status instead wastes
   your 90-minute budget on nothing.
5. Look at the real result (`collect`, the run manifest, logs).
6. Form hypothesis/decision B **based on what you actually saw in step 5** —
   not a second idea you had planned all along regardless of the outcome.
   If you decide a second experiment isn't worth it given the result and the
   remaining budget, that is a valid decision too, as long as it's a decision
   made *because of* the result, and say so explicitly.
7. If you run a second experiment, the same submit/wait/collect contract
   applies.
8. Pick a final candidate (even a weak one) and write
   `runs/SUBMISSION/submission.json` per the task card's contract.

A trajectory that submits one experiment, doesn't wait for or look at its
real result, and immediately writes a submission does not demonstrate
anything this track exists to check — do not do that even if you're confident
about your first idea.

## Credentials, isolation, budget mechanics

All identical to the standard track (separate `broker` service holds Volc
credentials, `runs/`/`workspace/` are bind-mounted, sequestered data is
absent from your image, `tools/remote_job.py` is your only path to
Volcengine) — see `benchmark/harbor/README.md` for the full explanation.
The only operational difference is that this task's
`environment/docker-compose.yaml` sets `REMOTE_JOB_DEFAULT_TASK` so your
`submit` calls land in the smoke ledger automatically.
