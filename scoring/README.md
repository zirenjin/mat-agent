# Scoring

`scoring/` is the only top-level place for score computation.

```text
registry.yaml                 score family registry
core/                         shared metrics and normalized result types
adapters/<score_family>/      thin wrappers around external or benchmark-specific scorers
matbench-discovery/           vendored Matbench Discovery source
schemas/                      score output schemas
```

Adapters should emit normalized JSON and keep upstream source separate from local wrapper glue.
