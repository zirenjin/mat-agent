#!/usr/bin/env bash
# Run this on the HOST, in the shell you'll invoke `harbor run` from — NOT
# inside any tool/agent session whose output gets logged or transcribed.
# It reads your local ~/.volc credentials and prints `export VAR=...` lines
# for VOLC_CREDENTIALS_B64 / VOLC_CONFIG_B64 / VOLC_ACCESS_TOKEN_B64.
#
# Usage:
#   eval "$(bash benchmark/harbor/mace-mof0/export_volc_env.sh)"
#   harbor run -p benchmark/harbor/mace-mof0 -a claude-code -m <model>
#
# task.toml pulls these into the container via [environment].env
# ${VOLC_CREDENTIALS_B64} / ${VOLC_CONFIG_B64} / ${VOLC_ACCESS_TOKEN_B64}
# templating; environment/entrypoint.sh writes them back to ~/.volc/* files
# at container start. Nothing here is written to disk or logged — it only
# ever lives in this shell's environment and the child `harbor run` process.
set -euo pipefail

VOLC_DIR="${VOLC_DIR:-$HOME/.volc}"

emit() {
    local var_name="$1" path="$2"
    if [ -f "$path" ]; then
        printf 'export %s=%s\n' "$var_name" "$(base64 -w0 < "$path")"
    else
        echo "# WARNING: $path not found, $var_name not set" >&2
    fi
}

emit VOLC_CREDENTIALS_B64 "$VOLC_DIR/credentials"
emit VOLC_CONFIG_B64 "$VOLC_DIR/config"
emit VOLC_ACCESS_TOKEN_B64 "$VOLC_DIR/access_token"
