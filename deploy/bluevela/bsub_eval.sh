#!/usr/bin/env bash
# Submit OpenMythos checkpoint evaluation job to BlueVela LSF.
#
# Usage:
#   bash deploy/bluevela/bsub_eval.sh
#
# Optional environment variables:
#   CHECKPOINT  -- path to .pt checkpoint (default: latest in /proj/checkpoints/pzerfos/openmythos/checkpoints)
#   DATASET_PATH -- parquet directory for held-out eval (default: /proj/datasets/pzerfos/fineweb-edu-100B/sample/100BT)
#   DEPTH_SWEEP  -- comma-separated n_loops values (default: 1,2,4,8,12,16,24,32)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CKPT_DIR="/proj/checkpoints/pzerfos/openmythos/checkpoints"
CHECKPOINT="${CHECKPOINT:-$(ls -t "${CKPT_DIR}"/step_*.pt 2>/dev/null | head -1)}"
DATASET_PATH="${DATASET_PATH:-/proj/datasets/pzerfos/fineweb-edu-100B/sample/100BT}"
DEPTH_SWEEP="${DEPTH_SWEEP:-1,2,4,8,12,16,24,32}"

OUTPUT_DIR="/u/pzerfos/data/output/experiments"
LOG_DIR="${OUTPUT_DIR}/errs_and_logs"
mkdir -p "$LOG_DIR"

DATE=$(date "+%Y-%m-%d-%H-%M")
LOG_FILE="${LOG_DIR}/eval-${DATE}.log"
ERR_FILE="${LOG_DIR}/eval-${DATE}.err"

QUEUE=preemptable
BSUB_GROUP=grp_preemptable
JOB_NAME="pz-mythos-eval"

echo "========================================="
echo "  OpenMythos Checkpoint Evaluation"
echo "========================================="
echo "  Checkpoint:  $CHECKPOINT"
echo "  Dataset:     $DATASET_PATH"
echo "  Depth sweep: $DEPTH_SWEEP"
echo "  Log:         $LOG_FILE"
echo "  Err:         $ERR_FILE"
echo "========================================="

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found in $CKPT_DIR"
    exit 1
fi

bsub \
    -J "${JOB_NAME}" \
    -q "${QUEUE}" \
    -o "${LOG_FILE}" \
    -e "${ERR_FILE}" \
    -n 1 \
    -gpu "num=1/task:mode=exclusive_process" \
    -G "${BSUB_GROUP}" \
    blaunch \
    PYTHONUNBUFFERED=1 \
    bash -c "
        source \$(conda info --base)/etc/profile.d/conda.sh && \
        conda activate openmythos && \
        cd ${REPO_DIR} && \
        python evaluations/eval_checkpoint.py \
            --checkpoint ${CHECKPOINT} \
            --dataset-path ${DATASET_PATH} \
            --depth-sweep ${DEPTH_SWEEP} \
            --device cuda
    " 2>&1 | tee "${LOG_DIR}/eval-${DATE}_submit.log"
