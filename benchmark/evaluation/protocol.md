# MACE-MP-MOF0 Discovery — evaluation protocol

Frozen. `benchmark/evaluation/evaluate_mof0.py` is the **only** thing that
produces a reportable number. A submission that reports a metric the evaluator
did not emit is out of contract. Every metric value is produced by shelling out
to the frozen scorers in `scoring/` (`_lib/metrics.run_scorer`); nothing in the
evaluator re-implements a metric.

## Submission the evaluator accepts

```json
{
  "checkpoint": "runs/<id>/model/final.model",
  "inference": {
    "kind": "mace_calculator",                          // or "entrypoint"
    "entrypoint": "workspace/infer.py:build_calculator", // required iff kind == entrypoint
    "dtype": "float64",
    "head": "pbe_d3"
  },
  "training_recipe_entrypoint": "workspace/train_final.sh",
  "training_data": "runs/<id>/dev_train_used.xyz",       // optional; enables the leakage check
  "external_inputs": "runs/<id>/external_inputs.json",   // required iff any external data/weights used
  "notes": "free text"
}
```

`submission.py` refuses anything that: has no checkpoint; puts the checkpoint
under `benchmark/`/`scoring/`/`tools/`/`docker/`; uses an inference kind other
than `mace_calculator`/`entrypoint`; points an entrypoint outside `workspace/`;
uses a dtype other than `float32`/`float64`; omits `training_recipe_entrypoint`;
declares malformed `external_inputs`; or declares `training_data` containing a
sequestered-ML frame hash.

The evaluator never imports the agent's training code. Candidate inference
(checkpoint load + all energy/force/stress calls + relaxation + phonopy force
evaluations) runs inside `_lib/sandbox.run_in_sandbox`: a spawned subprocess with
a `sys.addaudithook` guard that blocks reads of the sequestered split and the
DFT reference assets, writes under `benchmark/`/`scoring/`/`data/`, network, and
process spawning. Only standardized numeric arrays cross back to the parent,
which is the only place DFT references are read and the scorecard is written.
This constrains what candidate *code* may touch, never what the model *is*.

## Stage A — ML generalization (sequestered framework holdout)

`data/mace_mof_0/discovery/_sequestered/sequestered_ml.xyz` — whole MOF systems
(`system_name`) held out of `dev_*`, so this measures cross-framework transfer,
not frame memorization. Candidate sees geometry only.

Scored with `scoring/machine_learning/{rmse,mae}.py`:

| Metric | Unit | Pooling |
| --- | --- | --- |
| energy RMSE / MAE | meV/atom | per-frame E/N, then over frames |
| force RMSE / MAE | meV/Å | pooled over all components / frames |
| stress RMSE / MAE | meV/Å³ | pooled over Voigt components (ASE convention) |

Also reported per framework under anonymized ids (`fw_<6hex>`) with metal node
and size — the held-out system names stay in the evaluator-only
`_sequestered/holdout_systems.json`.

## Stage B — MOF lattice dynamics

For each of **MOF-5, UiO-66, MOF-74, MIL-53**: relax the released `POSCAR`
(`FrechetCellFilter` + FIRE, `fmax = 1e-4 eV/Å`, ≤ 500 steps) → finite-
displacement phonons (phonopy, displacement 0.01 Å, per-axis supercell so each
lattice vector ≥ 8 Å capped at 2, mesh 12³) → EOS scan (isotropic strain
`[-2,-1,0,1,2]%`, ions relaxed at fixed cell) → Birch–Murnaghan fit.

Endpoint availability is fixed by what the ddmms release actually contains
(`_lib/release_adapters.AVAILABILITY`):

| endpoint | MOF-5 | UiO-66 | MOF-74 | MIL-53 | DFT reference |
| --- | :-: | :-: | :-: | :-: | --- |
| `phonon_dos_mae` | ✓ | ✓ | ✓ | ✓ | `phonons/DFT_DOS.zip` (grids aligned by the evaluator) |
| `phonon_frequency_rmse` | — | — | — | — | **no DFT band structure released → always `null`** |
| `imaginary_mode_count` | ✓ | ✓ | ✓ | ✓ | candidate phonopy run; baseline count is the reference |
| `unit_cell_length_percent_error` | ✓ | ✓ | ✓ | ✓ | `optimized structures.zip/DFT/*_geom-opt.json` |
| `space_group_match` | ✓ | ✓ | ✓ | ✓ | same (spglib, symprec 1e-3) |
| `bulk_modulus_relative_error` | ✓ | ✓ | ✓ | ✓ | `BulkModulus_data.zip` (`dft (GPa)` column) |
| `thermal_expansion_relative_error` | ✓ | ✓ | — | — | `NegativeThermalExpansion.csv`; QHA, **off unless `--with-qha`** |

Project-specific release parsing lives only in `_lib/release_adapters.py`. An
endpoint with no fair DFT reference is `null` with a `reason`, never a proxy.

## Aggregation

Per endpoint, ratio vs the baseline `r = candidate / baseline` (lower-is-better
metrics). `space_group_match`: `r = 1` if match state unchanged, `10` (and
**catastrophic**) if the baseline retained it and the candidate lost it, `0.5` if
recovered. `imaginary_mode_count`: `r = (cand + 0.5)/(base + 0.5)`, catastrophic
if the baseline had ≤ 0 and the candidate ≥ 2, or `r > 3`.

Per-MOF **aggregate** = geometric mean of *available* endpoint ratios (nulls
dropped for candidate and baseline alike).

**Catastrophic override:** if any single endpoint ratio exceeds
`catastrophic_ratio` (≥ 2.0; set from measured noise — see below), that MOF is
`worse` regardless of the geometric mean. This stops a single blown endpoint
from being averaged away.

MOF status: `better` if `aggregate < 1 - tol_mof` and no catastrophic endpoint;
`worse` if `aggregate > 1 + tol_mof` or any catastrophic endpoint; else `equal`;
`incomplete` if no endpoint scored.

## Tolerances

`_lib/aggregate._tolerances` reads `benchmark/evaluation/noise_estimate.json`
when present (produced by `estimate_noise.py` from ≥ 3 perturbed-seed repeats of
the baseline). Until then it uses the **provisional** protocol defaults
`tol_mof = 0.03`, `tol_ml = 0.05`, `tol_repro = 0.05` and the scorecard marks the
source `"protocol_default (provisional)"`.

## Verdict (`beats_baseline`)

`true` iff **all** hold (mirrors task card §6):

1. **Physics primary** — Stage-B `better` on ≥ ⌈¾ · N_scored⌉ MOFs and **no** MOF
   `worse` (catastrophic override included).
2. **No ML regression** — no Stage-A overall metric worse than baseline by more
   than `tol_ml`.
3. **Reproducible** — a re-run of `training_recipe_entrypoint` from the fixed
   foundation reproduces the scorecard within `tol_repro`. If not run,
   `beats_baseline = false` (`reasons: "reproducibility re-run not performed"`).
4. **Compute-legal** — `benchmark/compute/account_compute.py` returns `ok: true`.

`tier` is `not_a_win` when the verdict fails, else `unverified` — methodological
vs tuning (task card §7) is a human call on the research log, not something the
evaluator decides.

## Output

`scorecard.json` (see `scorecard_schema.json`) plus, with
`--compare-to-baseline`, `verdict.json`. The baseline scorecard at
`benchmark/baselines/mace_mof0/baseline_scores.json` is produced by the identical
code path (`--baseline`) and frozen.

## Why not "lowest force RMSE wins"

Force RMSE on in-distribution frames is the most overfittable proxy and a weak
predictor of lattice dynamics. The headline is downstream MOF physics; Stage A
only guards against trading away force-field fidelity to chase it.
