# Matbench Discovery Data Boundary

This framework vendors the upstream Matbench Discovery source interface and may
carry the upstream `data/` snapshot needed to run standard data loading and
training/inference workflows.

Do not vendor official model prediction archives or paper-specific generated
artifacts here. Those belong in downstream experiment repositories when a paper
run explicitly needs them.

For WBM-style workflows, record the resolved dataset path, checksum, split, and
label provenance in the Research Plane before training or evaluation.
