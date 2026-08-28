#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

CHECKPOINT=${STAGE21C_SCORER_CHECKPOINT:?Set STAGE21C_SCORER_CHECKPOINT}
MANIFEST=${STAGE55_HOLDOUT_MANIFEST:?Set STAGE55_HOLDOUT_MANIFEST}
TAG=${STAGE55_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
RETURN_ROOT=${STAGE55_RETURN_ROOT:-results/stage_17}
WORK_DIR="${RETURN_ROOT}/stage55_post_turn_guard4_holdout_running_${TAG}"
DEST="${RETURN_ROOT}/stage55_post_turn_guard4_holdout_return_${TAG}"
FAILURE_DEST="${RETURN_ROOT}/stage55_post_turn_guard4_holdout_failure_return_${TAG}"
COMPLETE=0

BASELINE_CFG=scripts/eval/configs/habitat_dual_system_vlmap_stage55_route_occ_audit_cfg.py
GUARD_CFG=scripts/eval/configs/habitat_dual_system_vlmap_stage55_route_occ_guard4_cfg.py

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
  package_run "logs/habitat/compare_vlmap_stage55_holdout_baseline_${TAG}" "${WORK_DIR}/baseline"
  package_run "logs/habitat/compare_vlmap_stage55_holdout_guard4_${TAG}" "${WORK_DIR}/guard4"
  printf '%s\n' "${status}" > "${WORK_DIR}/EXIT_STATUS.txt"
  git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
  git status --short > "${WORK_DIR}/git_status_short.txt"
  find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
  mv "${WORK_DIR}" "${FAILURE_DEST}"
}
trap package_failure EXIT

test -f "${CHECKPOINT}"
test -f "${MANIFEST}"
test ! -e "${WORK_DIR}"; test ! -e "${DEST}"; test ! -e "${FAILURE_DEST}"
test ! -e "logs/habitat/compare_vlmap_stage55_holdout_baseline_${TAG}"
test ! -e "logs/habitat/compare_vlmap_stage55_holdout_guard4_${TAG}"
mkdir -p "${WORK_DIR}/episode_manifests"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

python3 -m pytest -q \
  tests/unit_test/test_stage41_executor_contract.py \
  tests/unit_test/test_stage46_active_recovery.py \
  tests/unit_test/test_stage55_occ_2p5d_audit.py \
  tests/unit_test/test_stage55_post_turn_guard_analyzer.py
python3 -m py_compile \
  internnav/utils/stage55_occ_2p5d_audit.py \
  internnav/habitat_extensions/vln/habitat_vln_evaluator.py \
  scripts/eval/analyze_stage55_post_turn_guard.py \
  "${BASELINE_CFG}" "${GUARD_CFG}"

run_arm() {
  local arm=$1
  local config=$2
  local port=$3
  local master_port=$4
  local name="compare_vlmap_stage55_holdout_${arm}_${TAG}"
  STAGE21_EPISODE_IDS="${MANIFEST}" STAGE21_RUN_NAME="${name}" \
  STAGE21_EPISODE_SEED_REPLAY_MANIFEST="${MANIFEST}" \
  STAGE21C_SCORER_CHECKPOINT="${CHECKPOINT}" STAGE21C_SCORER_DEVICE=cpu \
  STAGE27_EVENT_MANIFEST="" STAGE46_EVAL_PORT="${port}" STAGE55_EVAL_PORT="${port}" \
  NPROC_PER_NODE=4 MASTER_PORT="${master_port}" \
  CUDA_VISIBLE_DEVICES="${STAGE55_CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  bash scripts/eval/bash/stage21_torchrun_eval.sh --config "${config}"
}

run_arm baseline "${BASELINE_CFG}" 3558 3568
run_arm guard4 "${GUARD_CFG}" 3559 3569

python3 scripts/eval/analyze_stage55_post_turn_guard.py \
  --baseline-root "logs/habitat/compare_vlmap_stage55_holdout_baseline_${TAG}" \
  --guard-root "logs/habitat/compare_vlmap_stage55_holdout_guard4_${TAG}" \
  --manifest "${MANIFEST}" \
  --output "${WORK_DIR}/stage55_post_turn_guard4_holdout_audit.json" --require-all

package_run "logs/habitat/compare_vlmap_stage55_holdout_baseline_${TAG}" "${WORK_DIR}/baseline"
package_run "logs/habitat/compare_vlmap_stage55_holdout_guard4_${TAG}" "${WORK_DIR}/guard4"
cp -a "${MANIFEST}" "${WORK_DIR}/episode_manifests/"
printf '%s\n' 0 > "${WORK_DIR}/EXIT_STATUS.txt"
git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
git status --short > "${WORK_DIR}/git_status_short.txt"
find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
mv "${WORK_DIR}" "${DEST}"
COMPLETE=1
echo "STAGE55_GUARD4_HOLDOUT_STATUS=complete"
echo "DEST=$(readlink -f "${DEST}")"
du -sh "${DEST}"
