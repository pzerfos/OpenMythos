#!/usr/bin/env bash
# Inner runner for a single NoPE ablation variant.
# Invoked by bsub_nope_ablation.sh; expects conda env openmythos activated
# and CWD to be the repo root.
#
# Positional arg:
#   $1 -- variant name: "baseline" | "scoped" | "partial"
#
# Required env:
#   GPUS_PER_NODE -- number of GPUs per node (passed by bsub wrapper)

set -euo pipefail

: "${GPUS_PER_NODE:?GPUS_PER_NODE must be set by the bsub wrapper}"
VARIANT="${1:?variant positional arg required (baseline|scoped|partial)}"

echo "==============================================="
echo "NoPE ablation variant: ${VARIANT}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "==============================================="

torchrun --nproc_per_node="${GPUS_PER_NODE}" \
    training/1b_poc_fineweb.py --variant "${VARIANT}"
