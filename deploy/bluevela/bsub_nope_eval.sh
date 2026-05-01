#!/usr/bin/env bash
# Submit NoPE-ablation post-training eval jobs to BlueVela LSF.
#
# Usage:
#   VARIANT=baseline EVAL=length bash deploy/bluevela/bsub_nope_eval.sh
#   VARIANT=scoped   EVAL=depth  bash deploy/bluevela/bsub_nope_eval.sh
#
# Submits one 1-GPU job per call. Loop in the shell to cover all 6 combos.
#
# Env vars:
#   VARIANT     -- baseline | scoped | partial   (required)
#   EVAL        -- length   | depth              (required)
#   CHECKPOINT  -- override checkpoint path      (default: final step_0030517.pt for the variant)
#   HELD_OUT    -- held-out parquet shard        (default: 013_00009.parquet — not reached in the 1B run)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${VARIANT:?must set VARIANT to baseline|scoped|partial}"
: "${EVAL:?must set EVAL to length|depth}"

case "$VARIANT" in
    baseline|scoped|partial) ;;
    *) echo "ERROR: VARIANT must be baseline|scoped|partial (got: $VARIANT)"; exit 1 ;;
esac

case "$EVAL" in
    length) EVAL_SCRIPT="evaluations/eval_length_gen.py" ;;
    depth)  EVAL_SCRIPT="evaluations/eval_depth_gen.py"  ;;
    *) echo "ERROR: EVAL must be length|depth (got: $EVAL)"; exit 1 ;;
esac

CKPT_DIR="/proj/checkpoints/pzerfos/openmythos/checkpoints/nope-ablation/${VARIANT}"
CHECKPOINT="${CHECKPOINT:-${CKPT_DIR}/step_0030517.pt}"
HELD_OUT="${HELD_OUT:-/proj/datasets/pzerfos/fineweb-edu-100B/sample/100BT/013_00009.parquet}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

OUTPUT_DIR="/u/pzerfos/data/output/experiments"
LOG_DIR="${OUTPUT_DIR}/errs_and_logs"
mkdir -p "$LOG_DIR"

DATE=$(date "+%Y-%m-%d-%H-%M")
TAG="nope-eval-${EVAL}-${VARIANT}"
LOG_FILE="${LOG_DIR}/${TAG}-${DATE}.log"
ERR_FILE="${LOG_DIR}/${TAG}-${DATE}.err"

QUEUE=preemptable
BSUB_GROUP=grp_preemptable
JOB_NAME="pz-${TAG}"

echo "========================================="
echo "  NoPE Eval: $EVAL / $VARIANT"
echo "  Checkpoint: $CHECKPOINT"
echo "  Held-out:   $HELD_OUT"
echo "  Log:        $LOG_FILE"
echo "========================================="

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
        python ${EVAL_SCRIPT} \
            --checkpoint ${CHECKPOINT} \
            --variant ${VARIANT} \
            --held-out-shard ${HELD_OUT}
    " 2>&1 | tee "${LOG_DIR}/${TAG}-${DATE}_submit.log"
