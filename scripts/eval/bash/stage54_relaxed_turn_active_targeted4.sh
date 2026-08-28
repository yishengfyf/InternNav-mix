#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

CHECKPOINT=${STAGE21C_SCORER_CHECKPOINT:?Set STAGE21C_SCORER_CHECKPOINT}
MANIFEST=${STAGE54_MANIFEST:?Set STAGE54_MANIFEST}
TAG=${STAGE54_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
RETURN_ROOT=${STAGE54_RETURN_ROOT:-results/stage_17}
WORK_DIR="${RETURN_ROOT}/stage54_relaxed_turn_active_running_${TAG}"
DEST="${RETURN_ROOT}/stage54_relaxed_turn_active_return_${TAG}"
FAILURE_DEST="${RETURN_ROOT}/stage54_relaxed_turn_active_failure_return_${TAG}"
FAILED_STAGE=initialization
COMPLETE=0

declare -A CONFIGS
CONFIGS[control]="scripts/eval/configs/habitat_dual_system_vlmap_stage27_m3_candidate_shadow_cfg.py"
CONFIGS[strict]="scripts/eval/configs/habitat_dual_system_vlmap_stage49_frozen_reorient_active_cfg.py"
CONFIGS[route_occ]="scripts/eval/configs/habitat_dual_system_vlmap_stage54_route_occ_turn_active_cfg.py"
CONFIGS[route_only]="scripts/eval/configs/habitat_dual_system_vlmap_stage54_route_only_turn_active_cfg.py"

package_run() {
  local source=$1
  local dest=$2
  mkdir -p "${dest}"
  [[ -d "${source}" ]] && cp -a "${source}/." "${dest}/"
}

package_all() {
  local root=$1
  for arm in control strict route_occ route_only; do
    package_run "logs/habitat/compare_vlmap_stage54_${arm}_${TAG}" "${root}/${arm}"
  done
}

package_failure() {
  local status=$?
  if [[ "${COMPLETE}" == 1 || "${status}" == 0 ]]; then return; fi
  mkdir -p "${WORK_DIR}"
  package_all "${WORK_DIR}"
  printf '%s\n' "${FAILED_STAGE}" > "${WORK_DIR}/FAILED_STAGE.txt"
  printf '%s\n' "${status}" > "${WORK_DIR}/EXIT_STATUS.txt"
  git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
  git status --short > "${WORK_DIR}/git_status_short.txt"
  mv "${WORK_DIR}" "${FAILURE_DEST}"
  find "${FAILURE_DEST}" -type f | sort > "${FAILURE_DEST}/RETURN_MANIFEST.txt"
  echo "STAGE54_STATUS=failed"
  echo "FAILED_STAGE=${FAILED_STAGE}"
  echo "DEST=$(readlink -f "${FAILURE_DEST}")"
}
trap package_failure EXIT

test -f "${CHECKPOINT}"
test -f "${MANIFEST}"
test ! -e "${WORK_DIR}"; test ! -e "${DEST}"; test ! -e "${FAILURE_DEST}"
for arm in control strict route_occ route_only; do
  test -f "${CONFIGS[$arm]}"
  test ! -e "logs/habitat/compare_vlmap_stage54_${arm}_${TAG}"
done
mkdir -p "${WORK_DIR}/episode_manifests"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

FAILED_STAGE=targeted_tests
python3 -m pytest -q \
  tests/unit_test/test_stage27_candidate_generation.py \
  tests/unit_test/test_stage41_executor_contract.py \
  tests/unit_test/test_stage46_active_recovery.py
python3 -m py_compile \
  internnav/utils/sparse_occ_memory.py \
  internnav/utils/stage46_active_recovery.py \
  internnav/habitat_extensions/vln/habitat_vln_evaluator.py \
  scripts/eval/analyze_stage54_relaxed_turn_active.py \
  "${CONFIGS[strict]}" "${CONFIGS[route_occ]}" "${CONFIGS[route_only]}"

run_arm() {
  local arm=$1
  local port=$2
  local master_port=$3
  local name="compare_vlmap_stage54_${arm}_${TAG}"
  STAGE21_EPISODE_IDS="${MANIFEST}" STAGE21_RUN_NAME="${name}" \
  STAGE21_EPISODE_SEED_REPLAY_MANIFEST="${MANIFEST}" \
  STAGE21C_SCORER_CHECKPOINT="${CHECKPOINT}" STAGE21C_SCORER_DEVICE=cpu \
  STAGE27_EVENT_MANIFEST="" STAGE46_EVAL_PORT="${port}" STAGE54_EVAL_PORT="${port}" \
  NPROC_PER_NODE=4 MASTER_PORT="${master_port}" \
  CUDA_VISIBLE_DEVICES="${STAGE54_CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  bash scripts/eval/bash/stage21_torchrun_eval.sh --config "${CONFIGS[$arm]}"
}

FAILED_STAGE=control
run_arm control 3540 3550
FAILED_STAGE=strict
run_arm strict 3541 3551
FAILED_STAGE=route_occ
run_arm route_occ 3542 3552
FAILED_STAGE=route_only
run_arm route_only 3543 3553

FAILED_STAGE=paired_audit
python3 scripts/eval/analyze_stage54_relaxed_turn_active.py \
  --control-root "logs/habitat/compare_vlmap_stage54_control_${TAG}" \
  --strict-root "logs/habitat/compare_vlmap_stage54_strict_${TAG}" \
  --route-occ-root "logs/habitat/compare_vlmap_stage54_route_occ_${TAG}" \
  --route-only-root "logs/habitat/compare_vlmap_stage54_route_only_${TAG}" \
  --manifest "${MANIFEST}" --output "${WORK_DIR}/stage54_relaxed_turn_active_audit.json" \
  --require-all

FAILED_STAGE=return_packaging
package_all "${WORK_DIR}"
cp -a "${MANIFEST}" "${WORK_DIR}/episode_manifests/"
printf '%s\n' 0 > "${WORK_DIR}/EXIT_STATUS.txt"
git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
git status --short > "${WORK_DIR}/git_status_short.txt"
find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
mv "${WORK_DIR}" "${DEST}"
COMPLETE=1
echo "STAGE54_STATUS=complete"
echo "DEST=$(readlink -f "${DEST}")"
du -sh "${DEST}"
