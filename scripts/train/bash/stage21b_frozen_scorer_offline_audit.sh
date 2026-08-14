#!/usr/bin/env bash
# Stage21b frozen scorer audit.  Offline only: no Habitat, no policy changes,
# no active recovery, no episode-time parameter update.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

DATA_DIR=${STAGE21B_AUDIT_DATA_DIR:?Set STAGE21B_AUDIT_DATA_DIR to the 500ep JSONL dataset}
CHECKPOINT_DIR=${STAGE21B_AUDIT_CHECKPOINT_DIR:?Set STAGE21B_AUDIT_CHECKPOINT_DIR to pilot/seed_53}
RETURN_ROOT=${STAGE21B_AUDIT_RETURN_ROOT:-results/stage_17}
PIPELINE_TAG=${STAGE21B_AUDIT_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
GPU=${STAGE21B_AUDIT_GPU:-0}

RUNNING_NAME="stage21b_frozen_scorer_offline_audit_running_${PIPELINE_TAG}"
SUCCESS_NAME="stage21b_frozen_scorer_offline_audit_return_${PIPELINE_TAG}"
FAILURE_NAME="stage21b_frozen_scorer_offline_audit_failure_return_${PIPELINE_TAG}"
WORK_DIR="${RETURN_ROOT}/${RUNNING_NAME}"
SUCCESS_DEST="${RETURN_ROOT}/${SUCCESS_NAME}"
FAILURE_DEST="${RETURN_ROOT}/${FAILURE_NAME}"
FAILED_STAGE=initialization
PIPELINE_COMPLETE=0

mkdir -p "${RETURN_ROOT}"
test -d "${DATA_DIR}"; test -f "${DATA_DIR}/summary.json"
test -d "${CHECKPOINT_DIR}"; test -f "${CHECKPOINT_DIR}/best.pt"
test ! -e "${WORK_DIR}"; test ! -e "${SUCCESS_DEST}"; test ! -e "${FAILURE_DEST}"
mkdir -p "${WORK_DIR}/audit"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

write_metadata() {
  local root=$1
  printf '%s\n' "${DATA_DIR}" > "${root}/SOURCE_DATASET.txt"
  printf '%s\n' "${CHECKPOINT_DIR}" > "${root}/SOURCE_CHECKPOINT.txt"
  cat > "${root}/AUDIT_SCOPE.txt" <<'EOF'
offline frozen scorer audit only
Frozen S2/NextDiT: true
Habitat started: false
Episode-time parameter update: false
Active recovery: false
EOF
  git rev-parse HEAD > "${root}/git_commit.txt"
  git status --short > "${root}/git_status_short.txt"
}

package_failure() {
  local exit_status=$?
  if [[ "${PIPELINE_COMPLETE}" == "1" || "${exit_status}" == "0" ]]; then return; fi
  if [[ -d "${WORK_DIR}" ]]; then
    printf '%s\n' "${FAILED_STAGE}" > "${WORK_DIR}/FAILED_STAGE.txt"
    printf '%s\n' "${exit_status}" > "${WORK_DIR}/EXIT_STATUS.txt"
    write_metadata "${WORK_DIR}"
    find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
    mv "${WORK_DIR}" "${FAILURE_DEST}"
    echo "STAGE21B_FROZEN_AUDIT_STATUS=failed"
    echo "FAILED_STAGE=${FAILED_STAGE}"
    echo "FAILURE_DEST=$(readlink -f "${FAILURE_DEST}")"
  fi
}
trap package_failure EXIT

FAILED_STAGE=offline_scorer_audit
CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/train/audit_stage21_multitask_scorer_shadow.py \
  --data-dir "${DATA_DIR}" --checkpoint-dir "${CHECKPOINT_DIR}" \
  --output-dir "${WORK_DIR}/audit" --device cuda \
  | tee "${WORK_DIR}/audit_stdout.log"

test -f "${WORK_DIR}/audit/audit_summary.json"
test -f "${WORK_DIR}/audit/val_predictions.jsonl"
printf '%s\n' "offline frozen scorer audit completed; no shadow action applied" > "${WORK_DIR}/AUDIT_STOPPED_AFTER.txt"
printf '%s\n' "0" > "${WORK_DIR}/EXIT_STATUS.txt"
write_metadata "${WORK_DIR}"
find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
mv "${WORK_DIR}" "${SUCCESS_DEST}"
PIPELINE_COMPLETE=1

echo "STAGE21B_FROZEN_AUDIT_STATUS=complete"
echo "AUDIT_STOPPED_AFTER=offline_frozen_scorer_audit"
echo "RETURN_NAME=${SUCCESS_NAME}"
echo "DEST=$(readlink -f "${SUCCESS_DEST}")"
du -sh "${SUCCESS_DEST}"
