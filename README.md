# mat-agent

`mat-agent` is a compact benchmark workspace for testing agents on physically
constrained materials ML tasks. The central workflow is: expose an agent to
approved train/validation data, let it fine-tune MLIPs through MatterTune, record
budgeted attempts, and score predictions on protected splits.

## Canonical Layout

Only these top-level directories are part of the active design:

```text
playground/        agent sandbox: registries, runs, cache
tools/             reusable infrastructure tools
skills/            agent operating instructions
benchmark_tasks/   benchmark task specs, warm starts, baselines, task packages
data/              dataset manifests, local data, prep scripts, checkpoints
scoring/           score registries, metric code, external scorer adapters
mattertune/        MatterTune checkout used as the MLIP/fine-tuning backend
docker/            unified Docker image and Bash lifecycle entrypoints
docs/              public design docs; only mattertune_bash_api.md is versioned
```

## What Goes Where

- Put agent-visible run outputs under `playground/runs/<run_id>/`.
- Put agent-visible registries under `playground/registry/`.
- Put reusable infrastructure only once under `tools/`.
- Put agent operating instructions under `skills/`.
- Put benchmark definitions under `benchmark_tasks/tasks/<task_id>/`.
- Put scoring wrappers under `scoring/adapters/<score_family>/` and shared metrics under `scoring/core/`.
- Treat `mattertune/` as external model source. Agents may configure MatterTune, not patch MLIP architecture code.

## Docker

Build the full MatterTune runtime with the fast China mirror default:

```bash
docker build --build-arg MAT_AGENT_INSTALL_MODE=full -f docker/Dockerfile -t mat-agent:full .
```

The Dockerfile also supports `MAT_AGENT_INSTALL_MODE=smoke` for fast structural checks.

The shell contract for MatterTune is documented in [docs/mattertune_bash_api.md](docs/mattertune_bash_api.md).

## Boundary

Agents can use train/validation data and MatterTune-supported fine-tuning options
such as multi-heads, replay, and LoRA where exposed by the Bash API. Agents must
not modify canonical raw/test/OOD datasets, and must not edit MatterTune backbone
architecture source during benchmark episodes.
