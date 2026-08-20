# MatterTune Bash API

This document is the normative agent-facing API specification for using MatterTune inside `mat-agent`. The public interface is Bash-first. MatterTune Python configs remain the internal source of truth, but YAML/JSON config files are not the v1 agent-facing API.

## 1. API Principles

| Principle | Requirement |
| --- | --- |
| Bash-first | Autonomous agents call shell commands with typed arguments. They are not required to author MatterTune config files. |
| Typed adapter | `tools/mattertune_train.py` maps Bash arguments into `MatterTunerConfig` and backbone configs. |
| Internal source of truth | MatterTune `MatterTunerConfig`, backbone configs, `capabilities()`, and `parity_status()` define the internal semantics. |
| No escape hatch | Public v1 commands MUST NOT expose generic dotted-path mutation such as `--override a.b.c=value` or `--set a.b.c=value`. |
| Model-local surface | Only parameters supported by a model protocol may appear in that model subcommand's `--help`. |
| Protected data isolation | Training commands may read train and validation data only. Protected test/OOD data belongs to `evaluate.sh`. |
| Capability refusal | Unsupported, undeclared, or unsafe protocol requests MUST fail before training. |

## 2. Command Model

| Command | Responsibility | Data visibility |
| --- | --- | --- |
| `train.sh MODEL ...` | Adaptation / fine-tuning. | train, validation |
| `inference.sh MODEL ...` | Run a checkpoint on unlabeled structures and write model outputs. | unlabeled structures |
| `evaluate.sh MODEL ...` | Run a checkpoint on labeled benchmark data and compute metrics / benchmark scores. | labeled evaluation split, including protected test/OOD when authorized |

The v1 Bash adapter subcommands are `deepmd`, `sevennet`, `chgnet`, and `mace` where supported by the command. This is the agent-facing CLI coverage, not the full MatterTune backbone registry. Dry-run execution plans MUST report `mattertune_backbone_name`, `capability_source`, `capability_errors`, `parity_status`, `parity_source`, and `parity_errors` so agents can distinguish Bash adapter support from MatterTune capability/parity metadata.

## 3. Common Artifact / Path Semantics

| Artifact or path | Semantics |
| --- | --- |
| `--checkpoint` | Input model checkpoint or trusted upstream checkpoint identifier as defined by the model protocol. |
| `--train-data` | Training split. Required for `train.sh`; must be readable and must not be mutated. |
| `--val-data` | Validation split. Optional for `train.sh`; must be readable and must not be mutated. |
| `--run-dir` | Required output directory. Must be inside the agent sandbox, normally `playground/runs/<run_id>`. |
| `--resume-from` | Explicit checkpoint path to resume from. Boolean resume flags are not part of the API. |
| `--manifest` | Optional explicit manifest path. Default is `<run-dir>/execution_plan.json` for dry-run planning and `<run-dir>/run_manifest.json` for training. |

## 4. `train.sh` Common Arguments

| Argument | Required | Meaning |
| --- | --- | --- |
| `--checkpoint PATH_OR_ID` | yes | Model checkpoint according to the selected model protocol. |
| `--train-data PATH` | yes | Training dataset path. |
| `--val-data PATH` | no | Validation dataset path. |
| `--run-dir DIR` | yes | Agent sandbox output directory. |
| `--properties LIST` | yes | Comma-separated requested properties, e.g. `energy,forces` or `energy,forces,stresses`. |
| `--batch-size N` | no | DataLoader batch size. |
| `--num-workers N|auto` | no | DataLoader worker count. |
| `--seed N` | no | Reproducibility seed for protocol elements that need one. |
| `--learning-rate LR` | protocol-dependent | Optimizer learning-rate override. It is not a universal scientific default. |
| `--max-steps N` | no | Step budget. |
| `--checkpoint-every-n-epochs N` | no | Periodic checkpoint cadence. Default: `100`; must be >= 1. |
| `--resume-from PATH` | no | Explicit checkpoint to resume from. |
| `--dry-run` | no | Build and validate an execution plan without starting training. |
| `--json` | no | Emit JSON instead of stable `key=value`. |
| `--manifest PATH` | no | Explicit manifest path. |

`train.sh` MUST NOT accept `--test-data`.

## 5. DeepMD Training Protocols

DeepMD has two distinct protocol families.

### Single-Branch

```bash
train.sh deepmd \
  --mode single \
  --checkpoint model.pt \
  --train-data train.extxyz \
  --run-dir playground/runs/run_id \
  --properties energy,forces \
  --learning-rate 1e-4
```

| Argument | Required | Meaning |
| --- | --- | --- |
| `--mode single` | yes | Use a single-task or single-branch DeepMD checkpoint. |
| `--branch BRANCH` | conditional | Explicit branch for multi-task checkpoints evaluated as a single branch. Not required for genuinely single-task checkpoints. |

### Multi-Head / Multi-Domain

```bash
train.sh deepmd \
  --mode multihead \
  --checkpoint dpa31.pt \
  --train-data train.extxyz \
  --run-dir playground/runs/run_id \
  --properties energy,forces,stresses \
  --resume-head MP_traj \
  --copy-head new_task=MP_traj \
  --random-head task_x \
  --head-seed task_x=42 \
  --private-fitting-head task_x \
  --domain catalyst=MP_traj \
  --domain polymer=new_task \
  --sampling-weight catalyst=0.5 \
  --loss-weight polymer=2.0 \
  --trainable-scope head_only \
  --learning-rate 1e-4
```

| Argument | Required | Meaning |
| --- | --- | --- |
| `--mode multihead` | yes | Use official DeePMD multi-head / multi-task adaptation semantics. |
| `--resume-head HEAD` | one head declaration required | Resume an existing pretrained head. Repeatable. |
| `--copy-head NEW=SOURCE` | one head declaration required | Initialize `NEW` from existing `SOURCE`. Repeatable. |
| `--random-head HEAD` | one head declaration required | Randomly initialize a new head. Repeatable. |
| `--head-seed HEAD=N` | required for each random head | Seed for a random head. |
| `--private-fitting-head HEAD` | required for each random head | Give the random head a private fitting net so it cannot overwrite shared pretrained fitting parameters. |
| `--domain DOMAIN=HEAD` | yes | Route a data domain to a head. Repeatable. |
| `--sampling-weight DOMAIN=WEIGHT` | no | Sampling share for a domain. This is separate from loss weighting. |
| `--loss-weight DOMAIN=WEIGHT` | no | Objective coefficient for a domain. This is separate from sampling weight. |
| `--trainable-scope full|head_only` | no | Train full selected topology or only head-private parameters. |

`--head-init` is not part of the API because initialization strategy is per head, not global.

## 6. SevenNet Training Protocols

```bash
train.sh sevennet \
  --continual-mode replay-ewc \
  --checkpoint 7net-0 \
  --train-data train.extxyz \
  --run-dir playground/runs/run_id \
  --properties energy,forces,stresses \
  --replay-data replay.extxyz \
  --replay-batch-size 8 \
  --replay-ratio 0.25 \
  --fisher fisher.pt \
  --reference reference.pt \
  --lambda-ewc 10.0 \
  --learning-rate 1e-4
```

| Argument | Required | Meaning |
| --- | --- | --- |
| `--continual-mode none|replay|ewc|replay-ewc` | yes | Select the SevenNet adaptation protocol family explicitly. |
| `--replay-data PATH` | required for `replay`, `replay-ewc` | Replay dataset artifact. |
| `--replay-batch-size N` | required for `replay`, `replay-ewc` | Replay batch size. |
| `--replay-ratio FLOAT` | required for `replay`, `replay-ewc` | Fraction/share of replay data consumed by the protocol. |
| `--fisher PATH` | required for `ewc`, `replay-ewc` | Precomputed Fisher artifact. |
| `--reference PATH` | required for `ewc`, `replay-ewc` | Reference parameter artifact. |
| `--lambda-ewc FLOAT` | required for `ewc`, `replay-ewc` | EWC coefficient. |

Replay semantics MUST follow SevenNet reEWC behavior: replay batches receive separate optimizer steps rather than being folded into the current-task loss. Fisher estimation is an artifact-preparation operation, not a SevenNet native training flag.

## 7. CHGNet Training Protocols

```bash
train.sh chgnet \
  --objective efsm \
  --checkpoint 0.3.0 \
  --train-data train.extxyz \
  --run-dir playground/runs/run_id \
  --properties energy,forces,stresses,magnetic_moments \
  --atomref-mode preserve \
  --energy-basis per_atom \
  --learning-rate 1e-4
```

| Argument | Required | Meaning |
| --- | --- | --- |
| `--objective ef|efs|efm|efsm` | yes | CHGNet native objective family: energy/forces, plus optional stress and magnetic moments. |
| `--atomref-mode preserve|network_only|atomref_only|joint` | yes | Parameters allowed to move in CHGNet AtomRef adaptation. |
| `--energy-basis per_atom|total` | yes | Energy loss basis. `per_atom` matches CHGNet native objective parity. |

CHGNet magnetic moments use `unsigned_magnitude` semantics. Signed magnetic moments MUST NOT be exposed as a selectable Bash option.

## 8. MACE Training Protocols

MatterTune MACE Tier D currently means real MatterTune execution for single-head/full fine-tuning with explicit objective, trainer, and EMA controls. It does not claim native MACE multi-head replay parity.

```bash
train.sh mace \
  --checkpoint /mnt/checkpoints/mace.model \
  --trust-checkpoint \
  --head-mode single_head \
  --train-data train.extxyz \
  --val-data val.extxyz \
  --run-dir playground/runs/run_id \
  --properties energy,forces,stresses \
  --energy-weight 1 \
  --forces-weight 10 \
  --stress-weight 100 \
  --learning-rate 0.005 \
  --batch-size 8 \
  --max-epochs 2500 \
  --ema \
  --ema-decay 0.995
```

| Argument | Required | Meaning |
| --- | --- | --- |
| `--checkpoint PATH` | yes | Trusted local MACE checkpoint file. Named checkpoint download/resolution is not the canonical path. |
| `--trust-checkpoint` | yes | Acknowledge trusted local pickle / `torch.load` checkpoint semantics. |
| `--head-mode single_head|multi_head` | yes | Select MACE head protocol family. Non-dry-run execution is Tier D only for `single_head`. |
| `--head HEAD` | required for `multi_head` | Target head for multi-head checkpoints in planning. Multi-head training execution is refused until native parity is implemented. |
| `--energy-weight FLOAT` | no | Energy loss coefficient; default `1.0`. |
| `--forces-weight FLOAT` | no | Forces loss coefficient; default `1.0`. |
| `--stress-weight FLOAT` | no | Stress loss coefficient; default `1.0`. |
| `--max-epochs N` | no | Epoch budget. Required for paper-like epoch-based recipes unless `--max-steps` is used. |
| `--accelerator NAME` | no | Lightning accelerator, default `auto`. |
| `--devices auto|all|N|LIST` | no | Lightning devices, e.g. `auto`, `1`, or `0,1,2,3`. |
| `--strategy NAME` | no | Lightning strategy, default `auto`. |
| `--precision NAME` | no | Lightning precision, default `32-true`. |
| `--energy-key KEY` | no | EXTXYZ `Atoms.info` key to expose as canonical ASE potential energy label, e.g. `dft_energy`. |
| `--forces-key KEY` | no | EXTXYZ `Atoms.arrays` key to expose as canonical ASE forces label, e.g. `dft_forces`. |
| `--stress-key KEY` | no | EXTXYZ `Atoms.info` key to expose as canonical ASE stress label, e.g. `dft_stress`. |
| `--ema` | no | Apply MatterTune EMA recipe. |
| `--ema-decay FLOAT` | required with `--ema` by protocol convention | EMA decay; default `0.995`. |
| `--lora` | no | Apply MatterTune LoRA recipe. |
| `--lora-rank N` | required when `--lora` is used | LoRA rank. |
| `--lora-alpha N` | required when `--lora` is used | LoRA alpha. |

The Bash API MUST NOT execute MACE replay, pseudo-label replay, or native multi-head replay until those protocol families are represented by MatterTune capability/parity metadata and stable adapter code. Dry-run may carry a multi-head plan, but non-dry-run multi-head execution MUST fail before training.

## 9. `inference.sh`

| Argument | Required | Meaning |
| --- | --- | --- |
| `MODEL` | yes | Model family: `deepmd`, `sevennet`, `chgnet`, or `mace`. |
| `--checkpoint PATH` | yes | Trained checkpoint to load. |
| `--data PATH` | yes | Unlabeled structures. |
| `--output PATH` | yes | Prediction artifact path. |
| `--properties LIST` | yes | Properties to emit. |
| `--batch-size N` | no | Inference batch size. |
| `--num-workers N|auto` | no | DataLoader worker count. |
| `--device DEVICE` | no | Runtime device. |
| `--manifest PATH` | no | Inference manifest. |

`inference.sh` MUST NOT compute benchmark metrics.

## 10. `evaluate.sh`

| Argument | Required | Meaning |
| --- | --- | --- |
| `MODEL` | yes | Model family. |
| `--checkpoint PATH` | yes | Trained checkpoint to evaluate. |
| `--eval-data PATH` | yes | Labeled evaluation data, including protected test/OOD when authorized. |
| `--split train|val|test|ood` | yes | Evaluation split identity. |
| `--scorer NAME` | yes | Scorer registered by `scoring/registry.yaml`. |
| `--output PATH` | yes | Score report path. |
| `--manifest PATH` | no | Evaluation manifest. |

`score.sh` may exist as an internal scoring helper, but it is not the MatterTune lifecycle API.

## 11. Output Artifacts / Run Manifest

| Artifact | Producer | Required contents |
| --- | --- | --- |
| `execution_plan.json` | `train.sh --dry-run` | Parsed protocol, path checks, checkpoint identity, requested properties, capability requirements, resolved config preview. |
| `run_manifest.json` | `train.sh` | Command argv, resolved protocol, MatterTune config identity, capability/parity metadata, data identities, checkpoint outputs, resume source. |
| prediction artifact | `inference.sh` | Canonical prediction keys such as `pred_energy`, `pred_forces`, `pred_stress`, and model-specific metadata. |
| evaluation report | `evaluate.sh` | Scorer name, split identity, metric values, input artifact identities, capability/parity metadata. |

Run manifests MUST expose support/parity metadata obtained from MatterTune capability sources. The API spec does not hard-code current verification status values.

## 11.1 Checkpoint Preservation

MatterTune training launched through the adapter MUST preserve enough checkpoints for long-loop analysis and rollback:

- The primary checkpoint callback writes to `<run-dir>/checkpoints` with `filename=periodic-epoch={epoch:04d}-step={step}`.
- `save_last=True` is always enabled, so short tests that run fewer than the periodic cadence still preserve `last.ckpt` at train end.
- `save_top_k=-1` is always enabled on the primary callback, so every periodic checkpoint selected by the cadence is retained.
- `every_n_epochs` defaults to `100` and is controlled by `--checkpoint-every-n-epochs`.
- If a validation split is present, a second checkpoint callback writes `best-val_loss-epoch={epoch:04d}-step={step}` with `monitor=val_loss`, `mode=min`, `save_top_k=1`, and `every_n_epochs=1`. This preserves the actual best validation checkpoint independently of the periodic archive cadence.

## 12. Exit Codes

| Code | Meaning |
| --- | --- |
| `0` | Command completed successfully. |
| `2` | Bad arguments, invalid protocol combination, missing files, or sandbox violation. |
| `64` | Command exists but the requested lifecycle action is not implemented in the installed runtime. |
| other | Underlying Python, Docker, Lightning, dependency, or MatterTune failure. |

## 13. Safety / Capability Refusal

| Refusal class | Requirement |
| --- | --- |
| Wrong model flag | A flag from another model protocol MUST be rejected by argparse before config construction. |
| Protected data in training | `train.sh` MUST reject `--test-data` and any equivalent protected split input. |
| Unsupported capability | Requested capabilities must be checked against MatterTune capability/parity metadata before training. |
| Unsafe checkpoint | MACE requires explicit `--trust-checkpoint` for local pickle checkpoint loading; named-download semantics are not canonical. |
| Ambiguous resume | Boolean resume is illegal; callers must pass `--resume-from PATH`. |
| CHGNet signed magmom | Signed magnetic-moment training requests MUST be refused; only unsigned magnitude semantics are valid. |
| DeepMD random shared fitting net | Random heads MUST declare a seed and private fitting net. |
