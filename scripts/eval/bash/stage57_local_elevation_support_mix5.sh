#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"
CHECKPOINT=${STAGE21C_SCORER_CHECKPOINT:?Set STAGE21C_SCORER_CHECKPOINT}
MANIFEST=${STAGE57_MANIFEST:?Set STAGE57_MANIFEST}
TAG=${STAGE57_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
RETURN_ROOT=${STAGE57_RETURN_ROOT:-results/stage_17}
RUN_NAME="compare_vlmap_stage57_local_elevation_support_${TAG}"
WORK_DIR="${RETURN_ROOT}/stage57_local_elevation_support_running_${TAG}"
DEST="${RETURN_ROOT}/stage57_local_elevation_support_return_${TAG}"
FAILURE_DEST="${RETURN_ROOT}/stage57_local_elevation_support_failure_return_${TAG}"
CONFIG=scripts/eval/configs/habitat_dual_system_vlmap_stage57_local_elevation_support_cfg.py
COMPLETE=0
package_failure() {
  local status=$?
  if [[ "${COMPLETE}" == 1 || "${status}" == 0 ]]; then return; fi
  mkdir -p "${WORK_DIR}"
  [[ -d "logs/habitat/${RUN_NAME}" ]] && cp -a "logs/habitat/${RUN_NAME}/." "${WORK_DIR}/run/"
  printf '%s\n' "${status}" > "${WORK_DIR}/EXIT_STATUS.txt"
  git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
  find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
  mv "${WORK_DIR}" "${FAILURE_DEST}"
}
trap package_failure EXIT
test -f "${CHECKPOINT}"
test -f "${MANIFEST}"
test ! -e "${WORK_DIR}"
test ! -e "${DEST}"
test ! -e "${FAILURE_DEST}"
test ! -e "logs/habitat/${RUN_NAME}"
mkdir -p "${WORK_DIR}/episode_manifests"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1
python3 -m pytest -q tests/unit_test/test_stage55_occ_2p5d_audit.py tests/unit_test/test_stage56_floor_frame_consensus.py tests/unit_test/test_stage57_local_elevation_support.py
python3 -m py_compile internnav/utils/stage57_local_elevation_support.py internnav/habitat_extensions/vln/habitat_vln_evaluator.py scripts/eval/analyze_stage57_local_elevation_support.py "${CONFIG}"
STAGE21_EPISODE_IDS="${MANIFEST}" STAGE21_RUN_NAME="${RUN_NAME}" STAGE21_EPISODE_SEED_REPLAY_MANIFEST="${MANIFEST}" STAGE21C_SCORER_CHECKPOINT="${CHECKPOINT}" STAGE21C_SCORER_DEVICE=cpu STAGE23A_EVAL_PORT=3560 NPROC_PER_NODE=4 MASTER_PORT=3571 CUDA_VISIBLE_DEVICES="${STAGE57_CUDA_VISIBLE_DEVICES:-0,1,2,3}" bash scripts/eval/bash/stage21_torchrun_eval.sh --config "${CONFIG}"
python3 scripts/eval/analyze_stage57_local_elevation_support.py --run-root "logs/habitat/${RUN_NAME}" --manifest "${MANIFEST}" --output "${WORK_DIR}/stage57_local_elevation_support_audit.json" --require-all
cp -a "logs/habitat/${RUN_NAME}/." "${WORK_DIR}/run/"
cp -a "${MANIFEST}" "${WORK_DIR}/episode_manifests/"
printf '%s\n' 0 > "${WORK_DIR}/EXIT_STATUS.txt"
git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
git status --short > "${WORK_DIR}/git_status_short.txt"
find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
mv "${WORK_DIR}" "${DEST}"
COMPLETE=1
echo "STAGE57_STATUS=complete"
echo "DEST=$(readlink -f "${DEST}")"
du -sh "${DEST}"
