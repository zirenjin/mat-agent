# Matbench Discovery Skill

Use this skill when an agent needs the upstream Matbench Discovery source as a
standard materials-discovery evaluation interface.

## Source

The upstream source snapshot lives at:

```text
third_party/matbench-discovery
```

It is copied from the official repository:

```text
https://github.com/janosh/matbench-discovery
```

The framework includes the upstream core source, scripts, tests, and project
metadata. Large upstream assets such as model submissions, data dumps, paper
artifacts, and generated site files should stay outside this framework repo
unless a downstream project explicitly vendors them.

## Boundary

This skill exposes Matbench Discovery as an integration affordance. It does not
define this repository's benchmark tasks, hidden labels, scaffold ablations, or
paper-specific scoring.

## Typical Use

Import or inspect the upstream package from `third_party/matbench-discovery`,
then log inputs, outputs, and metrics through the Research Plane.


## Data And Model Assets

The framework may include the upstream `data/` snapshot needed for standard
Matbench Discovery loading, training, and inference workflows. Before a long run,
resolve and record the actual dataset path and checksum through the Research
Plane.

Model submission folders may be used as metadata for model selection, but large
prediction files (`*.csv.gz`) are not required for agent-side training and
inference workflows unless a downstream evaluation explicitly asks for official
submission comparisons.
