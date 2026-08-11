# MLIP Policy

Agents may use MatterTune-supported fine-tuning mechanisms such as hyperparameter changes, multi-head fine-tuning, replay, and layer freezing. Agents may not patch backbone architecture code during a playground task.

Allowed changes should be expressed as configs or command-line flags, not source edits under `models/src/mattertune/backbones/`.

