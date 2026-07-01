# Materials CodeAct Scaffold

The scaffold is designed around reusable affordances rather than prescribed
scientific workflows.

## Data Plane

Reusable data inspection, validation, conversion, splitting, and geometry summary
helpers.

## Compute Plane

Reusable runtime inspection, smoke-run, failure parsing, and checkpoint discovery
helpers.

## MLIP Plane

Command builders and contracts around `models/<model>/api/{train,inference,evaluate}.sh`.

## Research Plane

Generic experiment logging, evidence comparison, and report generation. This
plane is not a benchmark scorer.
