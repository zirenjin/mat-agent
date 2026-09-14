# Thin Harbor adapter layer on top of the existing, frozen mat-agent runtime
# (docker/Dockerfile). This does NOT reimplement the MatterTune/mace/phonopy
# install -- it starts FROM the already-built image and only adds what an
# autonomous agent trial needs: the repo's benchmark/task scaffolding,
# tools/remote_job.py's HTTP-client mode, and every supported *agent CLI*
# (Claude Code, Codex -- see below). This image is agent-agnostic: which one
# actually runs is chosen at `harbor run -a ...` time, not baked into which
# image you use. Adding a third (Gemini CLI, OpenHands, ...) is one more RUN
# step here plus one more class in benchmark/harbor/agents/preinstalled.py,
# never a change to this task or any other Dockerfile in this repo.
#
# Deliberately does NOT install the volc CLI or any Volc credentials -- see
# environment/broker/ (Dockerfile + server.py) and
# environment/docker-compose.yaml. Those live in exactly one place, a
# separate `broker` service `main` (this image) talks to over the compose
# network, so this container has nothing to shell out to even if an agent
# tried to bypass tools/remote_job.py's concurrency/budget gate directly.
#
# Build (from repo root, NOT this directory -- see environment/build.sh which
# wraps exactly this):
#   docker build -f docker/Dockerfile --build-arg MAT_AGENT_INSTALL_MODE=full -t mat-agent:full .
#   docker build -f benchmark/harbor/mace-mof0/environment/Dockerfile \
#       --build-arg GIT_SHA="$(git rev-parse --short HEAD)" \
#       -t mat-agent-harbor:"$(git rev-parse --short HEAD)" -t mat-agent-harbor:latest .
FROM mat-agent:full

ARG GIT_SHA=unknown
ENV MAT_AGENT_ROOT=/workspace/mat-agent \
    GIT_SHA=${GIT_SHA}
WORKDIR ${MAT_AGENT_ROOT}

# --- agent CLIs, installed at BUILD time, not runtime -----------------------
# harbor==0.23.0's own runtime install() step for BaseInstalledAgent
# subclasses (confirmed for both ClaudeCode and Codex) gates on
# `command -v <binary>` via environment.exec(), which was found unreliable in
# this sandbox on two separate real `harbor run` trials -- it reported
# success even when the binary was never installed, silently skipping the
# real install and leaving the agent with nothing to invoke. Pre-installing
# here and pairing it with benchmark/harbor/agents/preinstalled.py's
# PreinstalledAgentMixin (verifies presence, never attempts install) removes
# that runtime step entirely instead of trying to make it reliable. See
# benchmark/harbor/README.md "Known blockers" for the full diagnosis.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# The curl-based bootstrap.sh Harbor's own ClaudeCode.install() prefers
# (downloads.claude.ai) timed out from this build context (this network
# reaches npm/PyPI/nodesource mirrors fine, just not that specific host) --
# npm install is the documented alternative (it's what Harbor's own install()
# uses on Alpine) and does not depend on that host at all.
ARG CLAUDE_CODE_VERSION=2.1.270
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}
RUN claude --version

ARG CODEX_CLI_VERSION=latest
RUN npm install -g @openai/codex@${CODEX_CLI_VERSION}
RUN codex --version

# --- explicit allowlist of what the AGENT container may contain ---------
# Deliberately NOT a `COPY . .`: this is the hidden-data isolation boundary.
# data/mace_mof_0/discovery/_sequestered/** and data/mace_mof_0/raw/phonons/**
# (the sequestered-ML and sequestered-physics splits, task_card.md SS4) are
# NEVER referenced below and therefore cannot end up in this image by
# accident, regardless of what data/mace_mof_0/ grows on the host over time.
COPY benchmark/ ${MAT_AGENT_ROOT}/benchmark/
COPY tasks/ ${MAT_AGENT_ROOT}/tasks/
COPY docs/mattertune_bash_api.md ${MAT_AGENT_ROOT}/docs/mattertune_bash_api.md
# volc/task-configs/ as read-only starting-point templates only -- this
# image never installs the volc CLI or credentials (see header comment), so
# volc/scripts/submit_and_watch_task.py (which shells out to `volc`) is
# deliberately never copied here; it lives only in environment/broker/'s
# image. A config the agent wants to actually submit must be copied or
# authored under workspace/ first (bind-mounted, broker-visible) --
# volc/task-configs/ baked in here is invisible to the separate broker
# container. See instruction.md.
COPY volc/task-configs/ ${MAT_AGENT_ROOT}/volc/task-configs/
COPY playground/registry/ ${MAT_AGENT_ROOT}/playground/registry/
COPY data/registry.yaml ${MAT_AGENT_ROOT}/data/registry.yaml
COPY data/mace_mof_0/manifest.json ${MAT_AGENT_ROOT}/data/mace_mof_0/manifest.json
# The fixed foundation checkpoint -- task_card.md SS2 fixes this as the only
# pretrained *foundation* weight a submission's training_recipe_entrypoint
# may declare; mofs_v1/v2 are baseline artifacts scored by the verifier, not
# weights on offer to the agent, and data/mace_mof_0/external/ (31GB of
# upstream download staging, not part of the SS4 data contract at all) is
# intentionally never copied.
COPY data/mace_mof_0/models/mace_agnesi_medium.model ${MAT_AGENT_ROOT}/data/mace_mof_0/models/mace_agnesi_medium.model
# The reproduced-baseline checkpoint -- read-only warm start for this Harbor
# task (benchmark/baselines/mace_mof0/reproduced_baseline_manifest.json).
# chmod 444 is defense-in-depth, not an absolute guarantee (a root process in
# this container could still chmod it back) -- see that manifest's
# immutability_enforcement.honest_limitation. The durable check is the
# recorded checkpoint_sha256; re-hash before trusting a run that touched this.
COPY data/mace_mof_0/runs/mof0_paper_reproduce_force_mh_lr_20260829T124117Z/models/mof0_paper_reproduce.model \
     ${MAT_AGENT_ROOT}/data/mace_mof_0/baseline/mof0_reproduced_baseline.model
RUN chmod 444 ${MAT_AGENT_ROOT}/data/mace_mof_0/baseline/mof0_reproduced_baseline.model
COPY data/mace_mof_0/raw/data_training/ ${MAT_AGENT_ROOT}/data/mace_mof_0/raw/data_training/
COPY data/mace_mof_0/discovery/dev_train.xyz ${MAT_AGENT_ROOT}/data/mace_mof_0/discovery/dev_train.xyz
COPY data/mace_mof_0/discovery/dev_eval.xyz ${MAT_AGENT_ROOT}/data/mace_mof_0/discovery/dev_eval.xyz

# workspace/ and runs/ are bind-mounted from the host at container start
# (environment/docker-compose.yaml) -- these mkdirs only matter if that
# mount is ever removed and this image runs standalone.
RUN mkdir -p ${MAT_AGENT_ROOT}/workspace ${MAT_AGENT_ROOT}/runs ${MAT_AGENT_ROOT}/playground/loop_db

CMD ["/bin/bash"]
