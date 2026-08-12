# Matbench Discovery Adapter

This adapter hosts Matbench Discovery integration code. The upstream source
snapshot is vendored under `upstream/`; thin mat-agent wrappers should call the
standalone metrics in `scoring/machine_learning/` or `scoring/physics/` rather
than duplicating metric implementations.
