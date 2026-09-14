# Integrity constraints

Short list. These exist to keep results comparable and honest, not to limit
scientific choices. If a rule here seems to block a legitimate research idea,
that is a bug in the rule — raise it, do not route around it.

## R1 — Frozen infrastructure

Files listed in `frozen_manifest.json` must not change. `verify_frozen.py`
checks this. Covers: the evaluator, all scorers, data manifests, the compute
policy, baseline artifacts, the Docker environment, and the task card.

## R2 — Label isolation

Sequestered labels (ML holdout, physics/DFT reference values) are read only by
the evaluator. The agent never loads them, copies them, or fits against them.
Reported numbers are only those the evaluator produced.

## R3 — No test-set tuning

The released `test.xyz` has public labels. Use `dev_eval.xyz` for model
selection. Iterating against `test.xyz` numbers is test-set overfitting.

## R4 — Declared inputs only

Start from the fixed foundation checkpoint. Train only on the allowed data pool.
Any other pretrained weights or external datasets must be declared **before use**
in `runs/<run_id>/external_inputs.json` with: source URL, license, checksum,
what it is used for, and why it does not leak sequestered content. Undeclared
external inputs invalidate the submission.

## R5 — Honest compute

Every run writes a manifest with real measured GPU-seconds and GPU model
(`benchmark/compute/policy.yaml` §"run manifest"). Do not under-report,
split runs to dodge the per-experiment cap, or resume across manifests to hide
cost.

## R6 — No evaluation-set-specific hacks

No hard-coded sequestered structures, no per-structure correction terms keyed to
the evaluation systems, no silent unit/subset/threshold changes. Any
transformation applied to inputs must be visible in the submitted run config and
applied uniformly.

## R7 — Reproducibility

The submitted training recipe must re-run from the fixed foundation and
reproduce the submitted scorecard within the protocol's run-to-run tolerance.

## external_inputs.json schema

```json
{
  "inputs": [
    {
      "kind": "dataset | pretrained_weights",
      "id": "short-name",
      "source_url": "https://...",
      "license": "SPDX id or text",
      "sha256": "...",
      "used_for": "e.g. rehearsal mixture",
      "leakage_argument": "why this cannot contain sequestered MOF-5/UiO-66/MOF-74/MIL-53 phonon references"
    }
  ]
}
```
