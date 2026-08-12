# Logging Skill

Use this skill to keep agent experiments auditable without duplicating logging
systems.

## Canonical Tools

- `tools/time_logger.py` records timing events.
- `tools/feedback_logger.py` records human or evaluator feedback.
- `tools/run_logged_docker.sh` wraps Docker execution when command logs are needed.

Run-local logs should live under `playground/runs/<run_id>/`. Do not create a
second logging helper for the same responsibility; extend the canonical tool if a
new field is required.
