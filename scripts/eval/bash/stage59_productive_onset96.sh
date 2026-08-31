#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"
CHECKPOINT=${STAGE21C_SCORER_CHECKPOINT:?Set STAGE21C_SCORER_CHECKPOINT}
MANIFEST=${STAGE59_MANIFEST:?Set STAGE59_MANIFEST}
TAG=${STAGE59_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${STAGE59_RUN_ROOT:-/data/usr_data/yifeifeng/internnav/stage_results/runs}
RETURN_ROOT=${STAGE59_RETURN_ROOT:-/data/usr_data/yifeifeng/internnav/stage_results}
RUN_NAME="compare_vlmap_stage59_productive_onset_${TAG}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
WORK_DIR="${RETURN_ROOT}/stage59_productive_onset_running_${TAG}"
DEST="${RETURN_ROOT}/stage59_productive_onset_return_${TAG}"
FAILURE_DEST="${RETURN_ROOT}/stage59_productive_onset_failure_return_${TAG}"
CONFIG=${STAGE59_CONFIG:-scripts/eval/configs/habitat_dual_system_vlmap_stage59_productive_onset_cfg.py}
COMPLETE=0

package_failure() {
  local status=$?
  if [[ "${COMPLETE}" == 1 || "${status}" == 0 ]]; then return; fi
  mkdir -p "${WORK_DIR}"
  printf '%s\n' "${status}" > "${WORK_DIR}/EXIT_STATUS.txt"
  git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
  find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
  mv "${WORK_DIR}" "${FAILURE_DEST}"
}
trap package_failure EXIT

test -f "${CHECKPOINT}"
test -f "${MANIFEST}"
test ! -e "${RUN_DIR}"
test ! -e "${WORK_DIR}"
test ! -e "${DEST}"
test ! -e "${FAILURE_DEST}"
mkdir -p "${RUN_ROOT}" "${WORK_DIR}/episode_manifests" "${WORK_DIR}/run"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

python3 -m pytest -q \
  tests/unit_test/test_stage58_geometry_contract.py \
  tests/unit_test/test_stage58_support_policy.py \
  tests/unit_test/test_stage59_productive_onset.py
python3 -m py_compile \
  internnav/utils/stage59_productive_onset.py \
  internnav/habitat_extensions/vln/habitat_vln_evaluator.py \
  scripts/eval/analyze_stage59_productive_onset.py \
  "${CONFIG}"

STAGE21_EPISODE_IDS="${MANIFEST}" \
STAGE21_RUN_NAME="${RUN_NAME}" \
STAGE21_EPISODE_SEED_REPLAY_MANIFEST="${MANIFEST}" \
STAGE21C_SCORER_CHECKPOINT="${CHECKPOINT}" \
STAGE21C_SCORER_DEVICE=cpu \
STAGE59_RUN_ROOT="${RUN_ROOT}" \
STAGE59_EVAL_PORT=3590 \
NPROC_PER_NODE=4 \
MASTER_PORT=3591 \
CUDA_VISIBLE_DEVICES="${STAGE59_CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
bash scripts/eval/bash/stage21_torchrun_eval.sh --config "${CONFIG}"

python3 scripts/eval/analyze_stage59_productive_onset.py \
  --run-root "${RUN_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${WORK_DIR}/stage59_productive_onset_audit.json" \
  --require-all

cp -a "${MANIFEST}" "${WORK_DIR}/episode_manifests/"
[[ -f "${RUN_DIR}/result.json" ]] && cp -a "${RUN_DIR}/result.json" "${WORK_DIR}/run/"
[[ -f "${RUN_DIR}/progress.json" ]] && cp -a "${RUN_DIR}/progress.json" "${WORK_DIR}/run/"
printf '%s\n' 0 > "${WORK_DIR}/EXIT_STATUS.txt"
git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
git status --short > "${WORK_DIR}/git_status_short.txt"
find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
mv "${WORK_DIR}" "${DEST}"
COMPLETE=1
echo "STAGE59_STATUS=complete"
echo "RUN_DIR=${RUN_DIR}"
echo "DEST=${DEST}"
du -sh "${DEST}"
