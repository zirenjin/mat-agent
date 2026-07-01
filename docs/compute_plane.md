# Compute Plane

The Compute Plane exposes generic runtime affordances:

- inspect GPU state
- clamp commands into smoke-run settings
- classify common runtime failures from logs
- discover checkpoints

Cluster-specific queue policies and experiment-specific launchers should live in
downstream deployment repositories.
