# Compute broker: the only container in this task that ever holds Volc
# credentials or has the `volc` CLI. See server.py's module docstring and
# benchmark/harbor/README.md ("Compute broker") for why this exists as a
# separate service instead of living in the agent's own `main` container.
#
# Built from repo root as build context (see ../../build.sh) so it can COPY
# the same volc/scripts/ helpers and benchmark/compute/policy.yaml the agent
# side (tools/remote_job.py) already depends on -- no logic is duplicated
# for a second time here, only re-hosted behind server.py's HTTP shim.
FROM mat-agent:full

ARG GIT_SHA=unknown
ENV MAT_AGENT_ROOT=/workspace/mat-agent \
    GIT_SHA=${GIT_SHA}
WORKDIR ${MAT_AGENT_ROOT}

# Vendored at build time from the operator's local ~/.volc/bin/volc -- see
# ../build.sh. Never committed (../.gitignore covers .volc-bin/ for both
# this and the agent Dockerfile).
COPY benchmark/harbor/mace-mof0/environment/.volc-bin/volc /usr/local/bin/volc
RUN chmod +x /usr/local/bin/volc

COPY tools/remote_job.py ${MAT_AGENT_ROOT}/tools/remote_job.py
COPY volc/scripts/submit_and_watch_task.py ${MAT_AGENT_ROOT}/volc/scripts/submit_and_watch_task.py
COPY benchmark/compute/policy.yaml ${MAT_AGENT_ROOT}/benchmark/compute/policy.yaml
COPY benchmark/harbor/mace-mof0/environment/broker/server.py ${MAT_AGENT_ROOT}/broker_server.py
COPY benchmark/harbor/mace-mof0/environment/entrypoint.sh /usr/local/bin/harbor-entrypoint.sh
RUN chmod +x /usr/local/bin/harbor-entrypoint.sh

RUN mkdir -p ${MAT_AGENT_ROOT}/runs ${MAT_AGENT_ROOT}/playground/loop_db

ENTRYPOINT ["/usr/local/bin/harbor-entrypoint.sh"]
CMD ["python3", "/workspace/mat-agent/broker_server.py", "--port", "8765"]
