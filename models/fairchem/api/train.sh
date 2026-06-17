#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 << 'USAGE'
Usage: bash models/fairchem/api/train.sh

FairChem/UMA training is not implemented in mat-agent.
Supported FairChem interfaces are:
    bash models/fairchem/api/inference.sh --model-path <ckpt.pt> --data <input.extxyz> --output <pred.extxyz> --head <oc20|omat|omol|odac> --device <cpu|cuda>
    bash models/fairchem/api/evaluate.sh --predictions <pred.extxyz> --truth <truth.extxyz> --output <verdict.json>
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

usage
echo "ERROR: FairChem/UMA training is unsupported in mat-agent." >&2
exit 2
