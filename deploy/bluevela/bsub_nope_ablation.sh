#!/usr/bin/env bash
# Submits the 3-variant NoPE ablation to BlueVela LSF.
# Each variant runs as an independent 4-GPU single-node job on the preemptable
# queue. Variants are passed as positional args to run_nope_ablation.sh
# (no new env vars introduced).
#
# Usage:
#   bash deploy/bluevela/bsub_nope_ablation.sh [baseline|scoped|partial|all]
# Default: all three variants submitted.
#
# Required env: CLEARML_API_HOST, CLEARML_API_ACCESS_KEY, CLEARML_API_SECRET_KEY, HF_TOKEN

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
TARGET_TOKENS="${TARGET_TOKENS:-1}"   # 1 B per variant per the spec
CLEARML_PROJECT="${CLEARML_PROJECT:-granite-mythos}"
DATASET_PATH="${DATASET_PATH:-/proj/datasets/pzerfos/fineweb-edu-100B/sample/100BT}"

NUM_NODES=1
GPUS_PER_NODE=4
QUEUE=preemptable
BSUB_GROUP=grp_preemptable

SELECT="${1:-all}"
case "$SELECT" in
    baseline|scoped|partial) VARIANTS=("$SELECT") ;;
    all) VARIANTS=(baseline scoped partial) ;;
    *) echo "Usage: $0 [baseline|scoped|partial|all]"; exit 1 ;;
esac

umask 0002
DATE=$(date "+%Y-%m-%d-%H-%M")
LOG_DIR="${OUTPUT_DIR}/errs_and_logs"
mkdir -p "$LOG_DIR"

for variant in "${VARIANTS[@]}"; do
    JOB_NAME="pz-mythos-nope-${variant}"
    LOG_FILE="${LOG_DIR}/nope-${variant}-${DATE}.log"
    ERR_FILE="${LOG_DIR}/nope-${variant}-${DATE}.err"

    echo "========================================="
    echo "  Submitting NoPE ablation: ${variant}"
    echo "========================================="
    echo "  GPUs:         ${GPUS_PER_NODE} (single node)"
    echo "  Variant:      ${variant}"
    echo "  Tokens/var:   ${TARGET_TOKENS} B"
    echo "  Log:          ${LOG_FILE}"
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
        GPUS_PER_NODE="${GPUS_PER_NODE}" \
        CLEARML_API_HOST="${CLEARML_API_HOST}" \
        CLEARML_API_ACCESS_KEY="${CLEARML_API_ACCESS_KEY}" \
        CLEARML_API_SECRET_KEY="${CLEARML_API_SECRET_KEY}" \
        HF_TOKEN="${HF_TOKEN}" \
        CLEARML_PROJECT="${CLEARML_PROJECT}" \
        OUTPUT_DIR="${OUTPUT_DIR}" \
        TARGET_TOKENS="${TARGET_TOKENS}" \
        DATASET_PATH="${DATASET_PATH}" \
        bash -c "
            source \$(conda info --base)/etc/profile.d/conda.sh &&
            conda activate openmythos &&
            cd ${REPO_DIR} &&
            bash deploy/bluevela/run_nope_ablation.sh ${variant}
        "
done
