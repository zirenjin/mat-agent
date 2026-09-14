#!/usr/bin/env bash
# Populate a small, separate build context directory via hardlinks (cp -al:
# instant, zero extra disk, only works within one filesystem -- the staging
# dir must be a sibling of the repo, not /tmp, if the repo lives on a
# network filesystem like this one does).
#
# WHY THIS EXISTS: this repo's working directory is ~789GB (data/ alone is
# ~739GB, almost all unrelated to the Harbor task -- other datasets, caches,
# old run outputs). Every `docker build .` from repo root -- including the
# EXISTING docker/Dockerfile's mat-agent:full, which predates this task --
# hung/failed on this host with a generic `Error response from daemon:
# request err`, reproducible even for a trivial two-line Dockerfile. This
# host's Docker (24.0.9, no buildx/BuildKit plugin available -- confirmed by
# testing DOCKER_BUILDKIT=1, which errors asking for buildx) only has the
# legacy builder, which walks and stat()s the ENTIRE context directory
# client-side to apply .dockerignore BEFORE sending anything -- so a huge
# working tree is slow/fails regardless of how small the final filtered
# context would be, and regardless of network conditions (the .dockerignore
# negation logic itself was verified correct with small synthetic trees
# earlier -- this is a distinct, structural problem, not a bug in that logic).
#
# The fix: don't ask Docker to walk the real repo root at all. Stage an
# explicit allowlist (hardlinked, so staging ~450MB across ~200k small files
# takes seconds, not a copy) into a small sibling directory, and point
# `docker build`/`docker compose build` at THAT instead. Every Dockerfile's
# own COPY list (the actual hidden-data isolation boundary, see
# environment/Dockerfile's header comment) is completely unchanged and still
# just as narrow -- this only changes what directory Docker has to look at
# to find those same files, not what any image ends up containing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
STAGE_DIR="${MAT_AGENT_BUILD_STAGE_DIR:-$(dirname "$REPO_ROOT")/mat-agent-harbor-buildctx}"

echo "[stage_context] staging into $STAGE_DIR (hardlinks from $REPO_ROOT)"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

stage() {
    local rel="$1"
    local src="$REPO_ROOT/$rel"
    local dst="$STAGE_DIR/$rel"
    if [ ! -e "$src" ]; then
        echo "[stage_context] WARNING: $src does not exist, skipping" >&2
        return
    fi
    mkdir -p "$(dirname "$dst")"
    cp -al "$src" "$dst"
}

# Same allowlist as the four Dockerfiles' own COPY instructions,
# union'd -- each Dockerfile still only COPYs its own subset of this from
# the staged context, so staging a superset here does not widen what ends up
# in any image. See environment/Dockerfile, Dockerfile.verifier,
# broker/Dockerfile, and ../../../../docker/Dockerfile for the real,
# per-image allowlists this must stay a superset of.
stage mattertune
stage tools
stage scoring
stage docker
stage benchmark
stage tasks
stage docs/mattertune_bash_api.md
stage volc/task-configs
stage volc/scripts
stage playground/registry
stage data/registry.yaml
stage data/mace_mof_0/manifest.json
stage data/mace_mof_0/models
stage data/mace_mof_0/raw
stage data/mace_mof_0/discovery
stage "data/mace_mof_0/runs/mof0_paper_reproduce_force_mh_lr_20260829T124117Z/models/mof0_paper_reproduce.model"

echo "[stage_context] done: $(du -sh "$STAGE_DIR" | cut -f1) staged"
