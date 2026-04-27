#!/usr/bin/env bash
# Submit OpenMythos 1B 10B-token scale-up run to BlueVela LSF.
# 2 nodes × 8 GPUs (H100 80GB) = 16 GPUs, preemptable queue.
# Resumes automatically from the latest checkpoint in OUTPUT_DIR/checkpoints.
#
# Usage:
#   bash deploy/bluevela/bsub_1b_10b.sh
#
# Required environment variables:
#   CLEARML_API_HOST, CLEARML_API_ACCESS_KEY, CLEARML_API_SECRET_KEY, HF_TOKEN
#
# Optional environment variables:
#   CLEARML_PROJECT  -- ClearML project (default: granite-mythos)
#   EXPERIMENT_NAME  -- ClearML task name (default: 1b-10b-tokens)
#   OUTPUT_DIR       -- checkpoint dir (default: /u/pzerfos/data/granite-mythos/output/experiments)
#   TARGET_TOKENS    -- token budget in billions (default: 10)

set -euo pipefail

REQUIRED_VARS=(CLEARML_API_HOST CLEARML_API_ACCESS_KEY CLEARML_API_SECRET_KEY HF_TOKEN)
MISSING=()
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then MISSING+=("$var"); fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "ERROR: Missing required environment variables:"
    for var in "${MISSING[@]}"; do echo "  - $var"; done
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-/u/pzerfos/data/granite-mythos/output/experiments}"
TARGET_TOKENS="${TARGET_TOKENS:-10}"
CLEARML_PROJECT="${CLEARML_PROJECT:-granite-mythos}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-1b-10b-tokens}"
DATASET_PATH="${DATASET_PATH:-/proj/datasets/pzerfos/fineweb-edu-100B/sample/100BT}"

NUM_NODES=2
GPUS_PER_NODE=8
MASTER_PORT=29500
QUEUE=preemptable
BSUB_GROUP=grp_preemptable
JOB_NAME="pz-mythos-1b-10b"

umask 0002
DATE=$(date "+%Y-%m-%d-%H-%M")
LOG_DIR="${OUTPUT_DIR}/errs_and_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${DATE}.log"
ERR_FILE="${LOG_DIR}/${DATE}.err"

echo "========================================="
echo "  OpenMythos 1B — 10B Token Scale-Up"
echo "========================================="
echo "  Nodes:         $NUM_NODES"
echo "  GPUs/node:     $GPUS_PER_NODE  (total: $((NUM_NODES * GPUS_PER_NODE)))"
echo "  Target tokens: ${TARGET_TOKENS}B"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Dataset:       $DATASET_PATH"
echo "  ClearML:       $CLEARML_PROJECT / $EXPERIMENT_NAME"
echo "  Log:           $LOG_FILE"
echo "  Err:           $ERR_FILE"
echo "========================================="

bsub \
    -J "${JOB_NAME}" \
    -q "${QUEUE}" \
    -o "${LOG_FILE}" \
    -e "${ERR_FILE}" \
    -n "${NUM_NODES}" \
    -gpu "num=${GPUS_PER_NODE}/task:mode=exclusive_process" \
    -G "${BSUB_GROUP}" \
    blaunch \
    PYTHONUNBUFFERED=1 \
    CLEARML_API_HOST="${CLEARML_API_HOST}" \
    CLEARML_API_ACCESS_KEY="${CLEARML_API_ACCESS_KEY}" \
    CLEARML_API_SECRET_KEY="${CLEARML_API_SECRET_KEY}" \
    HF_TOKEN="${HF_TOKEN}" \
    CLEARML_PROJECT="${CLEARML_PROJECT}" \
    EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    TARGET_TOKENS="${TARGET_TOKENS}" \
    DATASET_PATH="${DATASET_PATH}" \
    MASTER_PORT="${MASTER_PORT}" \
    NUM_NODES="${NUM_NODES}" \
    GPUS_PER_NODE="${GPUS_PER_NODE}" \
    bash -c "
        source \$(conda info --base)/etc/profile.d/conda.sh
        conda activate openmythos
        cd ${REPO_DIR}
        MASTER_ADDR=\$(echo \$LSB_HOSTS | awk '{print \$1}')
        echo \"Rank info: MASTER_ADDR=\${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NNODES=${NUM_NODES} GPUS_PER_NODE=${GPUS_PER_NODE}\"
        torchrun \
            --nnodes=${NUM_NODES} \
            --nproc_per_node=${GPUS_PER_NODE} \
            --rdzv_backend=c10d \
            --rdzv_endpoint=\${MASTER_ADDR}:${MASTER_PORT} \
            training/1b_poc_fineweb.py
    "
