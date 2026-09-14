# Task Card: MACE-MP-MOF0 Autonomous Discovery

## 1. Scientific problem

MACE-MP-MOF0 is a foundation MLIP (`MACE-MP-0b medium`) fine-tuned on PBE+D3
metal-organic-framework (MOF) data so that it can drive high-throughput phonon
calculations of MOFs. The published model improves MOF lattice dynamics over the
untuned foundation, but it still leaves measurable errors in phonon spectra,
imaginary-mode pathologies, thermal expansion, and bulk moduli relative to DFT.

**Your goal:** starting from this repo's own reproduced MACE-MP-MOF0 checkpoint
(`data/mace_mof_0/baseline/mof0_reproduced_baseline.model` — a working
reproduction of the published recipe, matching or slightly exceeding it on
released-test-set force/energy error; see §2), produce a fine-tuned MOF
potential that generalizes to MOF lattice dynamics *better than that
checkpoint*, under a fixed data / compute / evaluation / integrity contract,
using whatever modeling or training methodology you can justify.

You decide the method. The benchmark only fixes the problem and the rules.

## 2. What is fixed (you may not change these)

| Element | Fixed specification | Where enforced |
| --- | --- | --- |
| Training data | The MACE-MOF0 public release pool, minus the sequestered framework holdout. See §4. | `benchmark/integrity/`, `data/mace_mof_0/manifest.json` |
| Starting checkpoints | Two, both fixed: the raw foundation `data/mace_mof_0/models/mace_agnesi_medium.model` (MACE-MP-0b medium), and the reproduced baseline `data/mace_mof_0/baseline/mof0_reproduced_baseline.model` (this repo's own from-scratch reproduction of the MACE-MP-MOF0 recipe — see `benchmark/baselines/mace_mof0/reproduced_baseline_manifest.json`). **Warm-starting from the reproduced baseline is the expected default** — it already matches the published recipe, so re-deriving it from the raw foundation on every attempt wastes budget on reproduction rather than research. Starting from the raw foundation instead is still permitted when your hypothesis specifically calls for it (e.g. a from-scratch ablation), not forbidden by default. No other pretrained weight is allowed. | integrity manifest, `reproduced_baseline_manifest.json` |
| Compute budget | See `benchmark/compute/policy.yaml`. Total, per-experiment cap, and model-size cap all apply. | `benchmark/compute/account_compute.py` |
| Evaluator | `benchmark/evaluation/` — you never run scoring yourself for a reported number. You submit a checkpoint + inference entrypoint; the evaluator produces the scorecard. | `benchmark/evaluation/protocol.md` |
| Baseline | The reproduced checkpoint above (`mof0_reproduced_baseline.model`), scored by the same evaluator — not the published `mofs_v2.model` (context-only; see the manifest). "Beating it" is defined in §6. | `benchmark/baselines/mace_mof0/reproduced_baseline_manifest.json`, `submissions/reproduced_v1.json` |
| Integrity rules | `benchmark/integrity/constraints.md` (short). | `benchmark/integrity/verify_frozen.py` in CI / preflight |

`benchmark/**`, `scoring/**`, `data/mace_mof_0/manifest.json`, `data/registry.yaml`,
`docker/**`, and the infra tools listed in the frozen manifest are **read-only
reference**. `verify_frozen.py` will fail your submission if any of them changed.

## 3. What is open (you are expected to explore these yourself)

Architecture, message-passing design, readouts, loss function and its weighting,
optimizer, LR schedule, EMA, fine-tuning strategy (full / partial / adapter /
multi-head / distillation / anything), replay/rehearsal strategy and mixture,
regularization, data weighting or reweighting *within the allowed pool*,
augmentation of allowed structures, curriculum, ensembling for the submitted
model, uncertainty use, and target choice (energy/force/stress/Hessian/phonon-aware
objectives, etc.).

There is **no required ablation list and no suggested method ranking.** Do not
treat this section as a menu to work through. Form your own hypothesis about why
the published model underperforms on MOF lattice dynamics and pursue it.

## 4. Data contract

Provenance and checksums: `data/mace_mof_0/manifest.json` (frozen).

| Split | Path | You may | Notes |
| --- | --- | --- | --- |
| `dev-train` | `data/mace_mof_0/discovery/dev_train.xyz` | train, reweight, subset, augment | Released train ∪ valid frames, minus sequestered frameworks. Labels visible. |
| `dev-eval` | `data/mace_mof_0/discovery/dev_eval.xyz` | evaluate yourself, unlimited | Your own model-selection set. Labels visible. Not part of the reported score. |
| `sequestered-ML` | held by evaluator | nothing directly | Frames from framework families removed from `dev-*`, grouped by framework. Reported. |
| `sequestered-physics` | `data/mace_mof_0/raw/phonons/*`, DFT refs | nothing directly | MOF-5, UiO-66, MOF-74, MIL-53 phonon/DOS/NTE/bulk-modulus DFT references. The headline target. Reported. |

The released `test.xyz` labels are already public; therefore repeated tuning
against them is *test-set overfitting* and is out of contract. Use `dev-eval`
for iteration. Scored evaluations against the sequestered endpoints are
**budgeted** (see §5) and logged.

External data or pretrained weights other than the two fixed starting
checkpoints in §2: **not allowed by default.** If you have a scientific reason to introduce some (e.g. a
public MOF dataset for rehearsal), you must declare it first per
`benchmark/integrity/constraints.md`; an undeclared external source invalidates
the submission.

## 5. Compute and evaluation budget

Read `benchmark/compute/policy.yaml`. In summary:

- Every training or inference run writes a run manifest under `runs/<run_id>/`
  with measured GPU-seconds, GPU model, and the resolved config.
- `benchmark/compute/account_compute.py` sums the ledger. Exceeding the total or
  per-experiment cap invalidates later runs.
- Model parameter count is capped (policy file) so the task cannot be won by
  scaling model size far beyond the baseline. A larger model within the cap is
  allowed if you report the compute it costs.
- Scored evaluations against sequestered endpoints are limited (policy file).
  Plan your model-selection on `dev-eval`; spend scored evaluations on
  candidates you already believe in.

Compute efficiency (score per GPU-hour, and score at the baseline's own compute)
is reported alongside the raw score. A win that costs 10x the baseline's compute
is reported as such.

## 6. What "surpass the baseline" means

The evaluator emits a **scorecard**, not a single scalar. The reproduced
baseline checkpoint (§2) is scored identically and its scorecard is frozen in
`benchmark/baselines/mace_mof0/baseline_scores.json`.

You surpass the baseline when **all** of the following hold on the sequestered
endpoints:

1. **Physics primary** — the per-MOF phonon-physics aggregate (phonon DOS MAE,
   imaginary-mode count, unit-cell error, space-group retention, bulk-modulus
   relative error, and thermal-expansion relative error where the release
   supports it — the exact endpoint set and availability matrix are in
   `benchmark/evaluation/protocol.md`) is better than the baseline's on
   ≥ ⌈¾⌉ of the scored MOFs, with no MOF `worse` (a single catastrophic
   endpoint is enough to make a MOF `worse`, regardless of the average).
2. **No ML regression** — sequestered-ML energy/force/stress errors are within
   the protocol tolerance of the baseline or better (you may not trade away
   force-field fidelity for a phonon-only fit).
3. **Reproducible** — a re-run of your submitted training recipe from the fixed
   foundation reproduces the submitted checkpoint's scorecard within the
   protocol's run-to-run tolerance.
4. **Compute-legal** — the run ledger validates against
   `benchmark/compute/policy.yaml`.

These four conditions are the actual, binding definition of "surpass the
baseline" — nothing below changes them. `benchmark/evaluation/scientific_score.py`
additionally derives one scalar from this same scorecard for harness/reward
purposes (Harbor's verifier reward, `benchmark/harbor/README.md` §Objective):
0.0 unless all four conditions above hold, otherwise the mean per-MOF physics
improvement margin. Do not target that scalar directly as if it were a
different, softer objective than the four conditions — it is the same
pass/fail gate with a magnitude attached once it already passes, not a
substitute you can improve by trading away any one of the four conditions.

A result that improves one endpoint while breaking another is *not* a win; report
it honestly as a partial result.

## 7. Methodological improvement vs hyperparameter tuning

Report your result in one of two tiers, with evidence:

- **Methodological** — the improvement comes from a change that plausibly
  generalizes beyond this dataset (a new objective, architecture change,
  fine-tuning or replay scheme, etc.). Provide an argument and, where feasible,
  one transfer check (e.g. it also helps on a held-out framework family, or on a
  second property it was not tuned for).
- **Tuning** — the improvement is from search over LR / batch / epochs / loss
  weights / EMA within the published recipe family. Still a valid submission,
  but label it as such.

Paper-worthy submissions are expected to be in the first tier.

## 8. Required workflow

Work as an autonomous researcher, not from a checklist:

1. **Hypothesis** — write down, in `runs/<run_id>/hypothesis.md`, what you think
   limits the published model and what change should help and why.
2. **Experiment** — implement it in `workspace/`, train within budget, log the
   run manifest.
3. **Analysis** — evaluate on `dev-eval`; inspect failure modes (use
   `tools/anomaly_sniffer.py`, per-framework and per-property breakdowns).
   Write what you learned.
4. **Iterate** — next hypothesis follows from the analysis, not from a plan
   fixed in advance.
5. **Submit** — when a candidate beats `dev-eval` and your own physics proxy,
   spend a scored evaluation (§5). Keep going or stop.

## 9. Deliverables

Under `runs/` and a final `runs/SUBMISSION/`:

1. `submission.json` — pointer to the submitted checkpoint, the inference
   entrypoint, declared external data (if any), and the training recipe entry
   point for the reproducibility check.
2. The evaluator scorecard for the submitted model and the diff against
   `baseline_scores.json`.
3. The compute ledger (`account_compute.py --report`).
4. A research log: the hypothesis→experiment→analysis chain across iterations,
   including negative results.
5. A short report: what limitation you targeted, what change produced the
   result, tier (§7), transfer evidence, and honest limitations.
6. All large artifacts stay under gitignored `runs/` paths.

## 10. Out of contract (automatic invalidation)

- Editing anything in the frozen manifest (evaluator, scorers, data manifests,
  integrity, compute policy, baseline artifacts, Docker environment).
- Reading or fitting against sequestered labels outside the evaluator.
- Tuning against released `test.xyz`.
- Undeclared external data or pretrained weights.
- Starting from a checkpoint other than the two fixed in §2 (the raw
  foundation or the reproduced baseline).
- Reporting a number the evaluator did not produce.
- Hard-coding sequestered structures, per-structure corrections keyed to the
  evaluation set, or unit/subset changes not visible in the run config.
