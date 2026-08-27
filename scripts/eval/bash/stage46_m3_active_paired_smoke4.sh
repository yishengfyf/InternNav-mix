#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

CHECKPOINT=${STAGE21C_SCORER_CHECKPOINT:?Set STAGE21C_SCORER_CHECKPOINT}
MANIFEST=${STAGE46_MANIFEST:-scripts/eval/manifests/stage25_gt_detector_fresh500_smoke4.json}
ACTIVE_CONFIG=${STAGE46_ACTIVE_CONFIG:-scripts/eval/configs/habitat_dual_system_vlmap_stage46_m3_active_cfg.py}
TAG=${STAGE46_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
RETURN_ROOT=${STAGE46_RETURN_ROOT:-results/stage_17}
CONTROL_NAME="compare_vlmap_stage46_control_${TAG}"
ACTIVE_NAME="compare_vlmap_stage46_active_${TAG}"
CONTROL_ROOT="logs/habitat/${CONTROL_NAME}"
ACTIVE_ROOT="logs/habitat/${ACTIVE_NAME}"
WORK_DIR="${RETURN_ROOT}/stage46_m3_active_paired_running_${TAG}"
DEST="${RETURN_ROOT}/stage46_m3_active_paired_return_${TAG}"
FAILURE_DEST="${RETURN_ROOT}/stage46_m3_active_paired_failure_return_${TAG}"
FAILED_STAGE=initialization
COMPLETE=0

package_run() {
  local source=$1
  local dest=$2
  mkdir -p "${dest}"
  [[ -d "${source}" ]] && cp -a "${source}/." "${dest}/"
}

package_failure() {
  local status=$?
  if [[ "${COMPLETE}" == 1 || "${status}" == 0 ]]; then return; fi
  mkdir -p "${WORK_DIR}"
  package_run "${CONTROL_ROOT}" "${WORK_DIR}/control"
  package_run "${ACTIVE_ROOT}" "${WORK_DIR}/active"
  printf '%s\n' "${FAILED_STAGE}" > "${WORK_DIR}/FAILED_STAGE.txt"
  printf '%s\n' "${status}" > "${WORK_DIR}/EXIT_STATUS.txt"
  git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
  git status --short > "${WORK_DIR}/git_status_short.txt"
  mv "${WORK_DIR}" "${FAILURE_DEST}"
  find "${FAILURE_DEST}" -type f | sort > "${FAILURE_DEST}/RETURN_MANIFEST.txt"
  echo "STAGE46_STATUS=failed"
  echo "FAILED_STAGE=${FAILED_STAGE}"
  echo "DEST=$(readlink -f "${FAILURE_DEST}")"
}
trap package_failure EXIT

test -f "${CHECKPOINT}"
test -f "${MANIFEST}"
test -f "${ACTIVE_CONFIG}"
test ! -e "${WORK_DIR}"; test ! -e "${DEST}"; test ! -e "${FAILURE_DEST}"
test ! -e "${CONTROL_ROOT}"; test ! -e "${ACTIVE_ROOT}"
mkdir -p "${WORK_DIR}/episode_manifests"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

FAILED_STAGE=targeted_tests
python3 -m pytest -q \
  tests/unit_test/test_stage27_candidate_generation.py \
  tests/unit_test/test_stage41_executor_contract.py \
  tests/unit_test/test_stage46_active_recovery.py
python3 -m py_compile \
  internnav/utils/stage46_active_recovery.py \
  scripts/eval/analyze_stage46_m3_active_paired.py \
  scripts/eval/configs/habitat_dual_system_vlmap_stage46_m3_active_cfg.py \
  "${ACTIVE_CONFIG}"

FAILED_STAGE=frozen_control
CUDA_VISIBLE_DEVICES="${STAGE46_CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
STAGE21_EPISODE_IDS="${MANIFEST}" STAGE21_RUN_NAME="${CONTROL_NAME}" \
STAGE21_EPISODE_SEED_REPLAY_MANIFEST="${MANIFEST}" \
STAGE21C_SCORER_CHECKPOINT="${CHECKPOINT}" STAGE21C_SCORER_DEVICE=cpu \
STAGE27_EVENT_MANIFEST="" STAGE27_EVAL_PORT=${STAGE46_CONTROL_EVAL_PORT:-3461} \
NPROC_PER_NODE=4 MASTER_PORT=${STAGE46_CONTROL_MASTER_PORT:-3462} \
bash scripts/eval/bash/stage21_torchrun_eval.sh \
  --config scripts/eval/configs/habitat_dual_system_vlmap_stage27_m3_candidate_shadow_cfg.py

FAILED_STAGE=one_primitive_active
CUDA_VISIBLE_DEVICES="${STAGE46_CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
STAGE21_EPISODE_IDS="${MANIFEST}" STAGE21_RUN_NAME="${ACTIVE_NAME}" \
STAGE21_EPISODE_SEED_REPLAY_MANIFEST="${MANIFEST}" \
STAGE21C_SCORER_CHECKPOINT="${CHECKPOINT}" STAGE21C_SCORER_DEVICE=cpu \
STAGE27_EVENT_MANIFEST="" STAGE46_EVAL_PORT=${STAGE46_ACTIVE_EVAL_PORT:-3463} \
NPROC_PER_NODE=4 MASTER_PORT=${STAGE46_ACTIVE_MASTER_PORT:-3464} \
bash scripts/eval/bash/stage21_torchrun_eval.sh \
  --config "${ACTIVE_CONFIG}"

FAILED_STAGE=paired_audit
python3 scripts/eval/analyze_stage27_m3_candidate_shadow.py \
  --run-root "${CONTROL_ROOT}" \
  --output "${CONTROL_ROOT}/stage27_m3_candidate_shadow_audit.json"
python3 scripts/eval/analyze_stage27_m3_candidate_shadow.py \
  --run-root "${ACTIVE_ROOT}" \
  --output "${ACTIVE_ROOT}/stage27_m3_candidate_shadow_audit.json"
python3 scripts/eval/analyze_stage46_m3_active_paired.py \
  --control-root "${CONTROL_ROOT}" --active-root "${ACTIVE_ROOT}" \
  --manifest "${MANIFEST}" --output "${ACTIVE_ROOT}/stage46_m3_active_paired_audit.json" \
  --require-all

FAILED_STAGE=return_packaging
package_run "${CONTROL_ROOT}" "${WORK_DIR}/control"
package_run "${ACTIVE_ROOT}" "${WORK_DIR}/active"
cp -a "${MANIFEST}" "${WORK_DIR}/episode_manifests/"
printf '%s\n' 0 > "${WORK_DIR}/EXIT_STATUS.txt"
git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
git status --short > "${WORK_DIR}/git_status_short.txt"
find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
mv "${WORK_DIR}" "${DEST}"
COMPLETE=1
echo "STAGE46_STATUS=complete"
echo "DEST=$(readlink -f "${DEST}")"
du -sh "${DEST}"
