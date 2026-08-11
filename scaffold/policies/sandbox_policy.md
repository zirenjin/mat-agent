# Sandbox Policy

The playground is the smallest active range for an agent. It may create runs, logs, caches, derived configs, and derived train/validation artifacts. It may not modify canonical datasets, hidden tests, OOD splits, scoring source, or MLIP architecture code.

A benchmark may expose a second sandbox for data editing. In that layer, train and validation data may be derived or cleaned if the task allows it. Canonical `test` and `ood` paths remain read-only and should be unavailable during optimization.

