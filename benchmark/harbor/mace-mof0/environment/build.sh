#!/usr/bin/env bash
# Build the Harbor environment images for the mace-mof0 task, all from the
# same git SHA.
#
#   1. docker/Dockerfile -> mat-agent:full (existing, frozen runtime)
#   2. environment/Dockerfile -> mat-agent-harbor:<sha>+:latest
#      (AGENT env, referenced by task.toml [environment].docker_image --
#      Harbor expects this prebuilt, it does not build it. No volc CLI, no
#      credentials, sequestered data structurally excluded.)
#   3. environment/Dockerfile.verifier -> mat-agent-harbor-verifier:<sha>+:latest
#      (VERIFIER env, task.toml [verifier.environment].docker_image, same
#      "prebuilt, Harbor doesn't build it" contract. Includes the
#      sequestered split; no volc CLI, no credentials.)
#
# Builds from a STAGED context (environment/stage_context.sh), not repo
# root directly -- see that script's header comment for why: this repo's
# working directory is ~789GB, almost all irrelevant to these images, and
# every `docker build .` from repo root hung/failed on this host's Docker
# (24.0.9, no buildx -- legacy builder only, which walks the whole context
# client-side before applying .dockerignore). Confirmed fixed empirically:
# the identical Dockerfile that failed against repo root built successfully
# in ~30s against the staged context. Each Dockerfile's own COPY allowlist
# (the real hidden-data isolation boundary) is completely unaffected -- only
# where Docker looks for those same files changed, not what ends up in any
# image.
#
# The `broker` service (environment/docker-compose.yaml,
# environment/broker/Dockerfile) is NOT pre-built here -- it has no
# `docker_image` in task.toml, only a `build:` block in the compose overlay
# (context now also the staged dir, see that file), so Harbor builds it
# itself via `docker compose build` when `harbor run` brings the agent
# environment up. That means both the vendored volc CLI AND the staged
# context must still be present when THAT later build happens -- this
# script deliberately does not clean up either afterward. Re-run this
# script (cheap: staging is hardlinks, ~1s) if repo files change between
# building and running.
#
# Usage:
#   bash benchmark/harbor/mace-mof0/environment/build.sh
#   MAT_AGENT_INSTALL_MODE=smoke bash benchmark/harbor/mace-mof0/environment/build.sh   # fast, no MatterTune deps
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

GIT_SHA="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then
    echo "[build] WARNING: working tree has uncommitted changes -- the image will not exactly" >&2
    echo "        reproduce from git alone. Committing before building is strongly recommended" >&2
    echo "        for any run whose result you intend to report." >&2
fi

VOLC_BIN_SRC="${VOLC_BIN:-$HOME/.volc/bin/volc}"
VOLC_BIN_DST="$REPO_ROOT/benchmark/harbor/mace-mof0/environment/.volc-bin/volc"
if [ ! -x "$VOLC_BIN_SRC" ]; then
    echo "[build] ERROR: volc CLI not found at $VOLC_BIN_SRC (set VOLC_BIN=/path/to/volc)" >&2
    exit 1
fi
mkdir -p "$(dirname "$VOLC_BIN_DST")"
cp "$VOLC_BIN_SRC" "$VOLC_BIN_DST"
chmod +x "$VOLC_BIN_DST"
echo "[build] vendored volc CLI to $VOLC_BIN_DST (kept for the broker's later compose build -- not cleaned up)"

echo "[build] staging build context..."
bash "$REPO_ROOT/benchmark/harbor/mace-mof0/environment/stage_context.sh"
STAGE_DIR="${MAT_AGENT_BUILD_STAGE_DIR:-$(dirname "$REPO_ROOT")/mat-agent-harbor-buildctx}"

echo "[build] stage 1/3: mat-agent:full (docker/Dockerfile, MAT_AGENT_INSTALL_MODE=${MAT_AGENT_INSTALL_MODE:-full})"
docker build \
    --build-arg MAT_AGENT_INSTALL_MODE="${MAT_AGENT_INSTALL_MODE:-full}" \
    -f "$STAGE_DIR/docker/Dockerfile" -t mat-agent:full "$STAGE_DIR"

echo "[build] stage 2/3: mat-agent-harbor:${GIT_SHA} (agent env: no volc CLI/credentials, sequestered data excluded)"
docker build \
    --build-arg GIT_SHA="$GIT_SHA" \
    -f "$STAGE_DIR/benchmark/harbor/mace-mof0/environment/Dockerfile" \
    -t "mat-agent-harbor:${GIT_SHA}" -t mat-agent-harbor:latest "$STAGE_DIR"

echo "[build] stage 3/4: mat-agent-harbor-verifier:${GIT_SHA} (standard-track verifier, includes sequestered data + baked-in tests/test.sh)"
docker build \
    --build-arg GIT_SHA="$GIT_SHA" \
    --build-arg TESTS_SRC_DIR=benchmark/harbor/mace-mof0/tests \
    -f "$STAGE_DIR/benchmark/harbor/mace-mof0/environment/Dockerfile.verifier" \
    -t "mat-agent-harbor-verifier:${GIT_SHA}" -t mat-agent-harbor-verifier:latest "$STAGE_DIR"

echo "[build] stage 4/4: mat-agent-harbor-verifier-smoke:${GIT_SHA} (smoke-track verifier -- same image otherwise, different baked-in tests/test.sh: Harbor's separate-verifier-mode never uploads tests/ at runtime, see Dockerfile.verifier's header comment, so the two tracks' genuinely different test.sh cannot share one image)"
docker build \
    --build-arg GIT_SHA="$GIT_SHA" \
    --build-arg TESTS_SRC_DIR=benchmark/harbor/mace-mof0-smoke/tests \
    -f "$STAGE_DIR/benchmark/harbor/mace-mof0/environment/Dockerfile.verifier" \
    -t "mat-agent-harbor-verifier-smoke:${GIT_SHA}" -t mat-agent-harbor-verifier-smoke:latest "$STAGE_DIR"

echo "[build] done."
echo "  mat-agent-harbor:${GIT_SHA}                 (also :latest) -- task.toml [environment].docker_image"
echo "  mat-agent-harbor-verifier:${GIT_SHA}         (also :latest) -- standard track's [verifier.environment].docker_image"
echo "  mat-agent-harbor-verifier-smoke:${GIT_SHA}   (also :latest) -- smoke track's [verifier.environment].docker_image"
echo "  broker: not built here -- Harbor builds it from environment/docker-compose.yaml + environment/broker/Dockerfile at 'harbor run' time, using the same staged context at $STAGE_DIR."
echo "  this script's stdout is the record of which git SHA the :latest tags currently point to."
