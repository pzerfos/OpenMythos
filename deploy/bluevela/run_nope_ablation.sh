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

# Derive a unique rendezvous port per LSF job. torchrun's default (29500) can
# collide when a previous torchrun on the same node is still in TIME_WAIT or
# when LSF schedules multiple jobs close together. LSB_JOBID is unique per
# job on the cluster, so mapping it into [29500, 30499] avoids collisions
# within a realistic running-job window without needing dynamic port lookup.
MASTER_PORT=$(( 29500 + ${LSB_JOBID:-$$} % 1000 ))

echo "==============================================="
echo "NoPE ablation variant: ${VARIANT}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "MASTER_PORT=${MASTER_PORT} (derived from LSB_JOBID=${LSB_JOBID:-unset})"
echo "==============================================="

torchrun \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    training/1b_poc_fineweb.py --variant "${VARIANT}"
