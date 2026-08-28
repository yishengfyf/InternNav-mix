#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"
MANIFEST=${STAGE44_MANIFEST:?Set STAGE44_MANIFEST to a train episode JSON list}
EVAL_CONFIG=${STAGE44_EVAL_CONFIG:-scripts/eval/configs/habitat_dual_system_vlmap_stage44_train_d0_candidate_shadow_cfg.py}
TAG=${STAGE44_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
RUN_NAME="compare_vlmap_stage44_train_d0_candidate_shadow_${TAG}"
RUN_ROOT="logs/habitat/${RUN_NAME}"
RETURN_ROOT=${STAGE44_RETURN_ROOT:-results/stage_17}
WORK_DIR="${RETURN_ROOT}/stage44_train_d0_candidate_shadow_running_${TAG}"
DEST="${RETURN_ROOT}/stage44_train_d0_candidate_shadow_return_${TAG}"
FAILED_STAGE=initialization
COMPLETE=0

package_failure() {
  local status=$?
  if [[ "${COMPLETE}" == 1 || "${status}" == 0 ]]; then return; fi
  mkdir -p "${WORK_DIR}"
  [[ -d "${RUN_ROOT}" ]] && mv "${RUN_ROOT}" "${WORK_DIR}/run"
  printf '%s\n' "${FAILED_STAGE}" > "${WORK_DIR}/FAILED_STAGE.txt"
  printf '%s\n' "${status}" > "${WORK_DIR}/EXIT_STATUS.txt"
  git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
  git status --short > "${WORK_DIR}/git_status_short.txt"
  mv "${WORK_DIR}" "${RETURN_ROOT}/stage44_train_d0_candidate_shadow_failure_return_${TAG}"
}
trap package_failure EXIT

test -f "${MANIFEST}"
test ! -e "${WORK_DIR}"; test ! -e "${DEST}"
mkdir -p "${WORK_DIR}/episode_manifests"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

FAILED_STAGE=targeted_tests
python3 -m pytest -q tests/unit_test/test_stage27_candidate_generation.py
python3 -m py_compile scripts/eval/configs/habitat_dual_system_vlmap_stage44_train_d0_candidate_shadow_cfg.py scripts/eval/analyze_stage27_m3_candidate_shadow.py

FAILED_STAGE=frozen_s2_train_d0_shadow
CUDA_VISIBLE_DEVICES="${STAGE44_CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
STAGE21_EPISODE_IDS="${MANIFEST}" STAGE21_RUN_NAME="${RUN_NAME}" \
STAGE21_EPISODE_SEED_REPLAY_MANIFEST="${MANIFEST}" STAGE27_EVENT_MANIFEST="" \
STAGE21_EVAL_PORT="${STAGE44_EVAL_PORT:-3441}" NPROC_PER_NODE="${STAGE44_NPROC_PER_NODE:-4}" \
MASTER_PORT="${STAGE44_MASTER_PORT:-3442}" \
bash scripts/eval/bash/stage21_torchrun_eval.sh --config "${EVAL_CONFIG}"

FAILED_STAGE=stage44_candidate_audit
python3 scripts/eval/analyze_stage27_m3_candidate_shadow.py --run-root "${RUN_ROOT}" --output "${RUN_ROOT}/stage44_candidate_shadow_audit.json"

if [[ "${STAGE44_RUN_STAGE50_ANALYSIS:-0}" == 1 ]]; then
  python3 scripts/eval/analyze_stage50_depth_short_lookahead.py \
    --run-root "${RUN_ROOT}" \
    --output "${RUN_ROOT}/stage50_depth_short_lookahead_audit.json"
fi

FAILED_STAGE=return_packaging
mv "${RUN_ROOT}" "${WORK_DIR}/run"
cp -a "${MANIFEST}" "${WORK_DIR}/episode_manifests/"
printf '%s\n' 0 > "${WORK_DIR}/EXIT_STATUS.txt"
git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
git status --short > "${WORK_DIR}/git_status_short.txt"
find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
mv "${WORK_DIR}" "${DEST}"
COMPLETE=1
echo "STAGE44_STATUS=complete"
echo "DEST=$(readlink -f "${DEST}")"
