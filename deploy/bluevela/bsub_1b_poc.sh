#!/usr/bin/env bash
# Submit OpenMythos 1B PoC training job to BlueVela LSF.
#
# Usage:
#   bash deploy/bluevela/bsub_1b_poc.sh
#
# Required environment variables:
#   CLEARML_API_HOST, CLEARML_API_ACCESS_KEY, CLEARML_API_SECRET_KEY, HF_TOKEN
#
# Optional environment variables:
#   CLEARML_PROJECT  -- ClearML project (default: granite-mythos)
#   EXPERIMENT_NAME  -- ClearML task name (default: 1b-poc-fineweb-10B)
#   OUTPUT_DIR       -- output directory (default: /u/pzerfos/data/granite-mythos/output/experiments)
#   NUM_GPUS         -- GPUs to request (default: 2)
#   TARGET_TOKENS    -- token budget in billions (default: 10)

set -euo pipefail

# ---------------------------------------------------------------------------
# Validate required environment variables
# ---------------------------------------------------------------------------
REQUIRED_VARS=(CLEARML_API_HOST CLEARML_API_ACCESS_KEY CLEARML_API_SECRET_KEY HF_TOKEN)
MISSING=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        MISSING+=("$var")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "ERROR: Missing required environment variables:"
    for var in "${MISSING[@]}"; do
        echo "  - $var"
    done
    exit 1
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-/u/pzerfos/data/granite-mythos/output/experiments}"
NUM_GPUS="${NUM_GPUS:-4}"
TARGET_TOKENS="${TARGET_TOKENS:-10}"
CLEARML_PROJECT="${CLEARML_PROJECT:-granite-mythos}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-1b-poc-fineweb-10B}"
DATASET_PATH="${DATASET_PATH:-/proj/datasets/pzerfos/fineweb-edu-100B/sample/100BT}"

umask 0002

DATE=$(date "+%Y-%m-%d-%H-%M")
LOG_DIR="${OUTPUT_DIR}/errs_and_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${DATE}.log"
ERR_FILE="${LOG_DIR}/${DATE}.err"

QUEUE=preemptable
NUM_NODES=1
BSUB_GROUP=grp_preemptable
JOB_NAME="pz-mythos-1b-poc"

echo "========================================="
echo "  OpenMythos 1B PoC Training"
echo "========================================="
echo "  GPUs:          $NUM_GPUS"
echo "  Target tokens: ${TARGET_TOKENS}B"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Dataset:       $DATASET_PATH"
echo "  ClearML:       $CLEARML_PROJECT / $EXPERIMENT_NAME"
echo "  Log:           $LOG_FILE"
echo "  Err:           $ERR_FILE"
echo "========================================="

# ---------------------------------------------------------------------------
# Submit LSF job
# ---------------------------------------------------------------------------
bsub \
    -J "${JOB_NAME}" \
    -q "${QUEUE}" \
    -o "${LOG_FILE}" \
    -e "${ERR_FILE}" \
    -n "${NUM_NODES}" \
    -gpu "num=${NUM_GPUS}/task:mode=exclusive_process" \
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
    bash -c "
        source \$(conda info --base)/etc/profile.d/conda.sh && \
        conda activate openmythos && \
        cd ${REPO_DIR} && \
        if [ ${NUM_GPUS} -eq 1 ]; then \
            python training/1b_poc_fineweb.py; \
        else \
            torchrun --nproc_per_node=${NUM_GPUS} training/1b_poc_fineweb.py; \
        fi
    " 2>&1 | tee "${OUTPUT_DIR}/${DATE}_submit.log"
