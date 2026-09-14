# Submitted Volc Tasks

Append one entry per submission or verification pass.


## 2026-08-27T03:19:12Z mof0_paper_reproduce

- Task ID: `t-20260827111448-64nbt`
- Status snapshot: `Staging`
- Outcome: `unverified: marker not found before timeout`
- Config: `/GenSIvePFS/users/public/mat-agent/volc/task-configs/mace_mof0/mof0-paper-reproduce.yaml`
- Instance: `worker_0`
- Success markers: `Using foundation model for multiheads finetuning with Materials Project data, MACE_MOF0_MULTIHEAD_REPLAY_CONFIGURED`
- Description: Original requested task name: mof0 paper reproduce. Native MACE MACE-MP-MOF0 paper reproduction on 4 x A100 with MP replay and 100-step checkpoint archives. Image: gensi-cn-beijing.cr.volces.com/ai4science/mlip_agent:v1. Run ID: mof0_paper_reproduce_20260827T031355Z
- Notes: native MACE MACE-MP-MOF0 reproduction, 4xA100 batch 48, MP replay, 100-step checkpoints

## 2026-08-27T03:21:09Z mof0_paper_reproduce

- Task ID: `t-20260827111448-64nbt`
- Status snapshot: `Running`
- Outcome: `failed-fast: No such file or directory`
- Config: `(monitor-only)`
- Instance: `worker_0`
- Success markers: `Using foundation model for multiheads finetuning with Materials Project data, MACE_MOF0_MULTIHEAD_REPLAY_CONFIGURED`
- Description: (empty)
- Notes: continue monitoring corrected mlip_agent image task

## 2026-08-27T03:29:41Z mof0_paper_reproduce

- Task ID: `t-20260827112838-j6zf8`
- Status snapshot: `Failed`
- Outcome: `failed-fast: MACE_MOF0_ENV_SELECTION_FAILED`
- Config: `/GenSIvePFS/users/public/mat-agent/volc/task-configs/mace_mof0/mof0-paper-reproduce.yaml`
- Instance: `worker_0`
- Success markers: `Using foundation model for multiheads finetuning with Materials Project data, MACE_MOF0_MULTIHEAD_REPLAY_CONFIGURED`
- Description: Original requested task name: mof0 paper reproduce. Native MACE MACE-MP-MOF0 paper reproduction on 4 x A100 with MP replay and 100-step checkpoint archives. Image: gensi-cn-beijing.cr.volces.com/ai4science/mlip_agent:v1. Run ID: mof0_paper_reproduce_20260827T032738Z
- Notes: native MACE MACE-MP-MOF0 reproduction, 4xA100 batch 48, MP replay, 100-step checkpoints; auto-select MACE Python env

## 2026-08-27T03:32:26Z mof0_paper_reproduce

- Task ID: `t-20260827113133-wfp55`
- Status snapshot: `Running`
- Outcome: `failed-fast: MACE_MOF0_ENV_SELECTION_FAILED`
- Config: `/GenSIvePFS/users/public/mat-agent/volc/task-configs/mace_mof0/mof0-paper-reproduce.yaml`
- Instance: `worker_0`
- Success markers: `Using foundation model for multiheads finetuning with Materials Project data, MACE_MOF0_MULTIHEAD_REPLAY_CONFIGURED`
- Description: Original requested task name: mof0 paper reproduce. Native MACE MACE-MP-MOF0 paper reproduction on 4 x A100 with MP replay and 100-step checkpoint archives. Image: gensi-cn-beijing.cr.volces.com/ai4science/mlip_agent:v1. Run ID: mof0_paper_reproduce_20260827T033111Z
- Notes: native MACE MACE-MP-MOF0 reproduction, 4xA100 batch 48, MP replay, 100-step checkpoints; core CUDA MACE env selection

## 2026-08-27T03:44:24Z mof0_env_diagnostic

- Task ID: `t-20260827114009-npwh8`
- Status snapshot: `Running`
- Outcome: `verified: MACE_MOF0_DIAGNOSTIC_DONE`
- Config: `/GenSIvePFS/users/public/mat-agent/volc/task-configs/mace_mof0/mof0-env-diagnostic.yaml`
- Instance: `worker_0`
- Success markers: `MACE_MOF0_DIAGNOSTIC_DONE`
- Description: Diagnostic-only check for Python/conda/MACE/torch CUDA environment in mlip_agent:v1 before MACE-MOF0 training. Run ID: mof0_env_diagnostic_20260827T033844Z
- Notes: diagnostic-only Python/conda/MACE/torch CUDA probe for mlip_agent:v1

## 2026-08-27T04:11:46Z mof0_env_diagnostic

- Task ID: `t-20260827120929-sgc49`
- Status snapshot: `Queue`
- Outcome: `verified: MACE_MOF0_BOOTSTRAP_DIAGNOSTIC_OK`
- Config: `/GenSIvePFS/users/public/mat-agent/volc/task-configs/mace_mof0/mof0-env-diagnostic.yaml`
- Instance: `worker_0`
- Success markers: `MACE_MOF0_BOOTSTRAP_DIAGNOSTIC_OK, MACE_MOF0_DIAGNOSTIC_DONE`
- Description: Diagnostic-only bootstrap check: mace-torch first, phonopy==2.29.0 with scikit-build-core<0.10 constraint. Run ID: mof0_env_bootstrap_diagnostic_20260827T040905Z
- Notes: diagnostic bootstrap: mace-torch first, phonopy with scikit-build-core constraint

## 2026-08-27T04:24:30Z mof0_paper_reproduce

- Task ID: `t-20260827122255-9rrqn`
- Status snapshot: `Running`
- Outcome: `failed-fast: MACE_MOF0_PREFLIGHT_FAILED`
- Config: `/GenSIvePFS/users/public/mat-agent/volc/task-configs/mace_mof0/mof0-paper-reproduce.yaml`
- Instance: `worker_0`
- Success markers: `MACE_MOF0_MULTIHEAD_REPLAY_CONFIGURED, MACE_MOF0_PREFLIGHT_OK, Using foundation model for multiheads finetuning with Materials Project data`
- Description: Original requested task name: mof0 paper reproduce. Native MACE MACE-MP-MOF0 paper reproduction on 4 x A100 with runtime bootstrap for mace-torch==0.3.9 and phonopy==2.29.0, MP replay, and 100-step checkpoint archives. Image: gensi-cn-beijing.cr.volces.com/ai4science/mlip_agent:v1. Run ID: mof0_paper_reproduce_20260827T042207Z
- Notes: simplified launch: mlip_agent:v1 runtime bootstrap, native MACE MACE-MP-MOF0 4xA100 batch48 MP replay step100 checkpoints

## 2026-08-27T04:28:11Z mof0_paper_reproduce

- Task ID: `t-20260827122655-bpqx9`
- Status snapshot: `Running`
- Outcome: `failed-fast: Traceback (most recent call last):`
- Config: `/GenSIvePFS/users/public/mat-agent/volc/task-configs/mace_mof0/mof0-paper-reproduce.yaml`
- Instance: `worker_0`
- Success markers: `MACE_MOF0_MULTIHEAD_REPLAY_CONFIGURED, MACE_MOF0_PREFLIGHT_OK, Using foundation model for multiheads finetuning with Materials Project data`
- Description: Original requested task name: mof0 paper reproduce. Native MACE MACE-MP-MOF0 paper reproduction on 4 x A100 with runtime bootstrap for mace-torch==0.3.9 and phonopy==2.29.0, MP replay, and 100-step checkpoint archives. Image: gensi-cn-beijing.cr.volces.com/ai4science/mlip_agent:v1. Run ID: mof0_paper_reproduce_20260827T042554Z
- Notes: simplified launch: mlip_agent:v1 runtime bootstrap, trusted checkpoint load fix, native MACE MACE-MP-MOF0 4xA100 batch48 MP replay step100 checkpoints

## 2026-08-27T04:43:19Z mof0_paper_reproduce

- Task ID: `t-20260827124046-qxf7c`
- Status snapshot: `Staging`
- Outcome: `verified: MACE_MOF0_TRAINING_STARTED`
- Config: `/GenSIvePFS/users/public/mat-agent/volc/task-configs/mace_mof0/mof0-paper-reproduce.yaml`
- Instance: `worker_0`
- Success markers: `MACE_MOF0_TRAINING_STARTED, Using foundation model for multiheads finetuning with Materials Project data`
- Description: Original requested task name: mof0 paper reproduce. Native MACE MACE-MP-MOF0 paper reproduction on 4 x A100 with runtime bootstrap for mace-torch==0.3.9 and phonopy==2.29.0, global PyTorch weights-only override, startup smoke, MP replay, and 100-step checkpoint archives. Image: gensi-cn-beijing.cr.volces.com/ai4science/mlip_agent:v1. Run ID: mof0_paper_reproduce_20260827T043940Z
- Notes: hardened launch after full startup debug: global torch load override, bootstrap, preflight, patch smoke, native MACE 4xA100 batch48

## 2026-08-27T04:49:11Z mof0_paper_reproduce

- Task ID: `t-20260827124714-5kmnz`
- Status snapshot: `Running`
- Outcome: `failed-fast: Traceback (most recent call last):`
- Config: `/GenSIvePFS/users/public/mat-agent/volc/task-configs/mace_mof0/mof0-paper-reproduce.yaml`
- Instance: `worker_0`
- Success markers: `Using foundation model`
- Description: Original requested task name: mof0 paper reproduce. Native MACE MACE-MP-MOF0 paper reproduction on 4 x A100 with runtime bootstrap for mace-torch==0.3.9 and phonopy==2.29.0, global PyTorch weights-only override, launcher=torchrun, startup smoke, MP replay, and 100-step checkpoint archives. Image: gensi-cn-beijing.cr.volces.com/ai4science/mlip_agent:v1. Run ID: mof0_paper_reproduce_20260827T044538Z
- Notes: launcher=torchrun after distributed Slurm root-cause fix; require native MACE foundation-model log

## 2026-08-28T05:58:56Z mof0_paper_reproduce

- Task ID: `t-20260828135623-xbdjz`
- Status snapshot: `Running`
- Outcome: `verified: Using foundation model`
- Config: `/GenSIvePFS/users/public/mat-agent/volc/task-configs/mace_mof0/mof0-paper-reproduce.yaml`
- Instance: `worker_0`
- Success markers: `Using foundation model`
- Description: Original requested task name: mof0 paper reproduce. Native MACE MACE-MP-MOF0 paper reproduction on 4 x A100 with runtime bootstrap for mace-torch==0.3.9 and phonopy==2.29.0, global PyTorch weights-only override, torchrun compatibility patch, startup smoke, MP replay, and 100-step checkpoint archives. Image: gensi-cn-beijing.cr.volces.com/ai4science/mlip_agent:v1. Run ID: mof0_paper_reproduce_20260828T055530Z
- Notes: torchrun env patch for MACE 0.3.9 Slurm-only init; native MACE 4xA100 batch48

## 2026-08-28T06:00:32Z mof0_paper_reproduce

- Task ID: `t-20260828135623-xbdjz`
- Status snapshot: `Failed`
- Outcome: `failed-fast: Traceback (most recent call last):`
- Config: `(monitor-only)`
- Instance: `worker_0`
- Success markers: `Starting training, Epoch, MACE_MOF0_STEP_CHECKPOINT_SAVED`
- Description: (empty)
- Notes: follow-up monitor after confirmed multi-head MP replay startup

## 2026-08-28T06:40:40Z mof0_paper_reproduce

- Task ID: `t-20260828142531-mx6x6`
- Status snapshot: `Running`
- Outcome: `unverified: marker not found before timeout`
- Config: `/GenSIvePFS/users/public/mat-agent/volc/task-configs/mace_mof0/mof0-paper-reproduce.yaml`
- Instance: `worker_0`
- Success markers: `MACE_MOF0_MP_REPLAY_CACHE_READY`
- Description: Original requested task name: mof0 paper reproduce. Native MACE MACE-MP-MOF0 paper reproduction on 4 x A100 with runtime bootstrap for mace-torch==0.3.9 and phonopy==2.29.0, global PyTorch weights-only override, torchrun compatibility patch, rank0-only MP replay generation, predownloaded MP replay cache, startup smoke, and 100-step checkpoint archives. Image: gensi-cn-beijing.cr.volces.com/ai4science/mlip_agent:v1. Run ID: mof0_paper_reproduce_20260828T062316Z
- Notes: MP replay cache predownload plus rank0-only replay subset patch; native MACE 4xA100 batch48
