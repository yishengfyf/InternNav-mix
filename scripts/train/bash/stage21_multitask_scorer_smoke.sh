#!/usr/bin/env bash
# Stage21b offline scorer smoke: audit -> balanced multi-head updates -> package.
# This never launches Habitat and never modifies Frozen S2/NextDiT parameters.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

DATA_DIR=${STAGE21_TRAIN_DATA_DIR:?Set STAGE21_TRAIN_DATA_DIR to datasets/500ep}
RETURN_ROOT=${STAGE21_TRAIN_RETURN_ROOT:-results/stage_17}
PIPELINE_TAG=${STAGE21_TRAIN_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
RETURN_NAME=${STAGE21_TRAIN_RETURN_NAME:-stage21b_multitask_scorer_smoke_return_${PIPELINE_TAG}}
DEST="${RETURN_ROOT}/${RETURN_NAME}"
EPOCHS=${STAGE21_TRAIN_EPOCHS:-2}
SMOKE_STEPS=${STAGE21_TRAIN_SMOKE_STEPS:-30}
GPU=${STAGE21_TRAIN_GPU:-0}

PIPELINE_COMPLETE=0
FAILED_STAGE=initialization

package_status() {
  local exit_status=$?
  [[ -d "${DEST}" ]] || return
  if [[ "${PIPELINE_COMPLETE}" != "1" && "${exit_status}" != "0" ]]; then
    printf '%s\n' "${FAILED_STAGE}" > "${DEST}/FAILED_STAGE.txt"
    printf '%s\n' "${exit_status}" > "${DEST}/EXIT_STATUS.txt"
    echo "STAGE21_MULTITASK_TRAIN_STATUS=failed"
  fi
  git rev-parse HEAD > "${DEST}/git_commit.txt"
  git status --short > "${DEST}/git_status_short.txt"
  find "${DEST}" -type f | sort > "${DEST}/RETURN_MANIFEST.txt"
}
trap package_status EXIT

test -d "${DATA_DIR}"
test -f "${DATA_DIR}/summary.json"
test ! -e "${DEST}"
mkdir -p "${DEST}/training"
printf '%s\n' "${DATA_DIR}" > "${DEST}/SOURCE_DATASET.txt"

FAILED_STAGE=dataset_audit
python3 scripts/train/audit_stage21_multitask_dataset.py \
  --data-dir "${DATA_DIR}" --output "${DEST}/dataset_audit.json"

FAILED_STAGE=multitask_train_smoke
CUDA_VISIBLE_DEVICES="${GPU}" \
python3 scripts/train/train_stage21_multitask_scorer.py \
  --data-dir "${DATA_DIR}" --output-dir "${DEST}/training" \
  --epochs "${EPOCHS}" --smoke-steps "${SMOKE_STEPS}" \
  --batch-size 128 --progress-batch-size 32 --hidden-dim 128 \
  --dropout 0.10 --lr 3e-4 --weight-decay 1e-4 \
  --seed 21 --device cuda \
  | tee "${DEST}/training_stdout.log"

printf '%s\n' "offline scorer smoke only; active navigation not started" > "${DEST}/PIPELINE_STOPPED_AFTER.txt"
PIPELINE_COMPLETE=1
echo "STAGE21_MULTITASK_TRAIN_STATUS=complete"
echo "PIPELINE_STOPPED_AFTER=offline_multitask_smoke"
echo "RETURN_NAME=${RETURN_NAME}"
echo "DEST=$(readlink -f "${DEST}")"
du -sh "${DEST}"
