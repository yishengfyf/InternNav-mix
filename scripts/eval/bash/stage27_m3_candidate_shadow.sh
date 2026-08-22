#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"
MANIFEST=${STAGE27_MANIFEST:-scripts/eval/manifests/stage25_gt_detector_fresh500_smoke4.json}
CONFIG=${STAGE27_CONFIG:-scripts/eval/configs/habitat_dual_system_vlmap_stage27_m3_candidate_shadow_cfg.py}
EVENT_MANIFEST=${STAGE27_EVENT_MANIFEST:-}
RETURN_ROOT=${STAGE27_RETURN_ROOT:-results/stage_17}
PIPELINE_TAG=${STAGE27_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
RUN_NAME="compare_vlmap_stage27_m3_candidate_shadow_${PIPELINE_TAG}"
RUN_ROOT="logs/habitat/${RUN_NAME}"
WORK_DIR="${RETURN_ROOT}/stage27_m3_candidate_shadow_running_${PIPELINE_TAG}"
SUCCESS_DEST="${RETURN_ROOT}/stage27_m3_candidate_shadow_return_${PIPELINE_TAG}"
FAILURE_DEST="${RETURN_ROOT}/stage27_m3_candidate_shadow_failure_return_${PIPELINE_TAG}"
CUDA_DEVICES=${STAGE27_CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC=${STAGE27_NPROC_PER_NODE:-4}
FAILED_STAGE=initialization
PIPELINE_COMPLETE=0

package_failure() {
  local status=$?
  if [[ "${PIPELINE_COMPLETE}" == 1 || "${status}" == 0 ]]; then return; fi
  mkdir -p "${WORK_DIR}"
  if [[ -d "${RUN_ROOT}" && ! -e "${WORK_DIR}/run" ]]; then mv "${RUN_ROOT}" "${WORK_DIR}/run"; fi
  printf '%s\n' "${FAILED_STAGE}" > "${WORK_DIR}/FAILED_STAGE.txt"
  printf '%s\n' "${status}" > "${WORK_DIR}/EXIT_STATUS.txt"
  git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
  git status --short > "${WORK_DIR}/git_status_short.txt"
  mv "${WORK_DIR}" "${FAILURE_DEST}"
  find "${FAILURE_DEST}" -type f | sort > "${FAILURE_DEST}/RETURN_MANIFEST.txt"
  echo "STAGE27_STATUS=failed"
  echo "FAILED_STAGE=${FAILED_STAGE}"
  echo "DEST=$(readlink -f "${FAILURE_DEST}")"
}
trap package_failure EXIT

test -f "${MANIFEST}"
test -f "${CONFIG}"
test ! -e "${WORK_DIR}"
test ! -e "${SUCCESS_DEST}"
test ! -e "${FAILURE_DEST}"
mkdir -p "${WORK_DIR}/episode_manifests"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

FAILED_STAGE=targeted_tests
python3 -m pytest -q tests/unit_test/test_stage27_candidate_generation.py
python3 -m py_compile \
  internnav/utils/stage27_candidate_generation.py \
  scripts/eval/analyze_stage27_m3_candidate_shadow.py \
  scripts/eval/configs/habitat_dual_system_vlmap_stage27_m3_candidate_shadow_cfg.py

FAILED_STAGE=frozen_s2_candidate_shadow
CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8} \
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
STAGE21_EPISODE_IDS="${MANIFEST}" STAGE21_RUN_NAME="${RUN_NAME}" \
STAGE21_EPISODE_SEED_REPLAY_MANIFEST="${MANIFEST}" \
STAGE27_EVENT_MANIFEST="${EVENT_MANIFEST}" \
STAGE27_EVAL_PORT=${STAGE27_EVAL_PORT:-3095} \
NPROC_PER_NODE="${NPROC}" MASTER_PORT=${STAGE27_MASTER_PORT:-3096} \
bash scripts/eval/bash/stage21_torchrun_eval.sh --config "${CONFIG}"

FAILED_STAGE=stage27_audit
python3 scripts/eval/analyze_stage27_m3_candidate_shadow.py \
  --run-root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/stage27_m3_candidate_shadow_audit.json"

FAILED_STAGE=return_packaging
mv "${RUN_ROOT}" "${WORK_DIR}/run"
cp -a "${MANIFEST}" "${WORK_DIR}/episode_manifests/"
printf '%s\n' 0 > "${WORK_DIR}/EXIT_STATUS.txt"
git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
git status --short > "${WORK_DIR}/git_status_short.txt"
find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
mv "${WORK_DIR}" "${SUCCESS_DEST}"
PIPELINE_COMPLETE=1
echo "STAGE27_STATUS=complete"
echo "DEST=$(readlink -f "${SUCCESS_DEST}")"
du -sh "${SUCCESS_DEST}"
