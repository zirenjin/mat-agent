# mat-agent Architecture

`mat-agent` is a small Materials CodeAct framework. Its job is to expose stable
affordances to an agent, not to implement a full agent runtime or prescribe a
paper workflow.

## Layers

1. `models/*/api`: stable wrappers for train, inference, and evaluation.
2. `playground/registry`: factual registries for available models, datasets, and tools.
3. `mat_agent`: lightweight Python helpers for registry access, data inspection,
   model API command construction, failure classification, logging, and Matbench
   Discovery path resolution.
4. `mat_agent/scaffolds`: prompt-level CodeAct operating contracts.
5. `skills`: flat Markdown operating instructions for common workflows.

## Agent Loop

The framework expects a code-acting agent to use this loop:

1. Inspect registries and local artifacts.
2. State the hypothesis or operational goal.
3. Execute through model APIs or small helper utilities.
4. Inspect logs, metrics, and produced artifacts.
5. Revise the next action based on evidence.
6. Record commands, artifacts, and conclusions.

This borrows the useful discipline of ReAct, but keeps execution CodeAct-first
because materials workflows are dominated by code, files, containers, datasets,
and GPU jobs.
