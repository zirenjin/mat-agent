"""Evaluator-internal library for the MACE-MP-MOF0 Discovery task.

Only ``benchmark/evaluation/`` code imports this package. It holds the pieces that
must stay outside the agent's reach: submission validation, the sequestered-data
reader, the candidate-inference sandbox, Stage A / Stage B runners, the ddmms
release adapters, aggregation, and noise estimation.

Metric *values* are always produced by shelling out to the frozen scorers in
``scoring/`` (see ``metrics.run_scorer``); nothing here re-implements a metric.
"""
