#!/usr/bin/env bash
# Reconstructs ~/.volc/{credentials,config,access_token} from base64 env vars
# at container start, then execs whatever command Harbor/the agent asked for.
#
# Used ONLY by environment/broker/Dockerfile now -- the agent's own `main`
# image (environment/Dockerfile) never gets Volc credentials or the volc
# CLI at all (see that Dockerfile's header comment and
# environment/docker-compose.yaml).
#
# The env vars are populated on the HOST (never printed, never committed) by
# benchmark/harbor/mace-mof0/export_volc_env.sh, and reach the `broker`
# compose service via plain Docker Compose ${VAR} interpolation in
# environment/docker-compose.yaml — see that file and instruction.md's
# "Credentials" section.
set -euo pipefail

mkdir -p "$HOME/.volc"
chmod 700 "$HOME/.volc"

if [ -n "${VOLC_CREDENTIALS_B64:-}" ]; then
    printf '%s' "$VOLC_CREDENTIALS_B64" | base64 -d > "$HOME/.volc/credentials"
    chmod 600 "$HOME/.volc/credentials"
fi

if [ -n "${VOLC_CONFIG_B64:-}" ]; then
    printf '%s' "$VOLC_CONFIG_B64" | base64 -d > "$HOME/.volc/config"
    chmod 600 "$HOME/.volc/config"
fi

if [ -n "${VOLC_ACCESS_TOKEN_B64:-}" ]; then
    printf '%s' "$VOLC_ACCESS_TOKEN_B64" | base64 -d > "$HOME/.volc/access_token"
    chmod 600 "$HOME/.volc/access_token"
fi

# Never let the b64 blobs linger in the container's own environment/process list.
unset VOLC_CREDENTIALS_B64 VOLC_CONFIG_B64 VOLC_ACCESS_TOKEN_B64

exec "$@"
