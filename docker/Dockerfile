FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ARG PIP_INDEX_URL=https://mirrors.ivolces.com/pypi/simple/
ARG PIP_TRUSTED_HOST=mirrors.ivolces.com
ARG MAT_AGENT_INSTALL_MODE=smoke

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} \
    MAT_AGENT_ROOT=/workspace/mat-agent

WORKDIR ${MAT_AGENT_ROOT}

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY mattertune/ /workspace/mat-agent/mattertune/
COPY tools/ /workspace/mat-agent/tools/
COPY scoring/ /workspace/mat-agent/scoring/
COPY docker/entrypoints/ /usr/local/bin/

RUN if [ "$MAT_AGENT_INSTALL_MODE" = "smoke" ]; then \
      test -f /workspace/mat-agent/mattertune/src/mattertune/__init__.py; \
    elif [ "$MAT_AGENT_INSTALL_MODE" = "core" ]; then \
      python -m pip install --prefer-binary Cython wheel setuptools numpy && \
      python -m pip install --prefer-binary --no-build-isolation -e mattertune; \
    elif [ "$MAT_AGENT_INSTALL_MODE" = "full" ]; then \
      python -m pip install --prefer-binary Cython wheel setuptools numpy && \
      python -m pip install --prefer-binary --no-build-isolation -e "mattertune[train,data,logging,neighborlist]"; \
    else \
      echo "Unknown MAT_AGENT_INSTALL_MODE=$MAT_AGENT_INSTALL_MODE" >&2; exit 2; \
    fi

ENV PYTHONPATH=/workspace/mat-agent:/workspace/mat-agent/mattertune/src
CMD ["/bin/bash"]
