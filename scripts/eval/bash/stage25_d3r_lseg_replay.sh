#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

RUN_ROOT=${STAGE25_D3R_RUN_ROOT:?Set STAGE25_D3R_RUN_ROOT}
CANDIDATES=${STAGE25_D3R_CANDIDATES:?Set STAGE25_D3R_CANDIDATES}
CHECKPOINT=${STAGE25_D3R_LSEG_CHECKPOINT:?Set STAGE25_D3R_LSEG_CHECKPOINT}
VLMAPS_REPO=${STAGE25_D3R_VLMAPS_REPO:-/home/yifeifeng/workspace/vlmaps}
TAG=${STAGE25_D3R_TAG:-$(date +%Y%m%d_%H%M%S)}
RETURN_ROOT=${STAGE21_RETURN_ROOT:-results/stage_17}
WORK_DIR="${RETURN_ROOT}/stage25_d3r_lseg_running_${TAG}"
SUCCESS_DIR="${RETURN_ROOT}/stage25_d3r_lseg_return_${TAG}"
FAILURE_DIR="${RETURN_ROOT}/stage25_d3r_lseg_failure_return_${TAG}"
CUDA_DEVICES=${STAGE25_D3R_CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC=${STAGE25_D3R_NPROC_PER_NODE:-4}
WINDOW_STEPS=${STAGE25_D3R_WINDOW_STEPS:-24}
MAX_FRAMES=${STAGE25_D3R_MAX_FRAMES:-}
FAILED_STAGE=initialization
COMPLETE=0

package_failure() {
  local status=$?
  if [[ "${COMPLETE}" == "1" || "${status}" == "0" ]]; then return; fi
  printf '%s\n' "${FAILED_STAGE}" > "${WORK_DIR}/FAILED_STAGE.txt"
  printf '%s\n' "${status}" > "${WORK_DIR}/EXIT_STATUS.txt"
  mv "${WORK_DIR}" "${FAILURE_DIR}"
  echo "STAGE25_D3R_STATUS=failed"
  echo "DEST=$(readlink -f "${FAILURE_DIR}")"
}
trap package_failure EXIT

test -d "${RUN_ROOT}"
test -f "${CANDIDATES}"
test -d "${VLMAPS_REPO}"
test -f "${CHECKPOINT}"
test ! -e "${WORK_DIR}"
test ! -e "${SUCCESS_DIR}"
test ! -e "${FAILURE_DIR}"
mkdir -p "${WORK_DIR}"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

FAILED_STAGE=targeted_tests
python3 -m pytest -q tests/unit_test/test_stage25_semantic_confirmation.py

FAILED_STAGE=d3r_replay
CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8} \
REPLAY_ARGS=(
  --run-root "${RUN_ROOT}" --candidates "${CANDIDATES}"
  --output-root "${WORK_DIR}/d3r_audit" --vlmaps-repo "${VLMAPS_REPO}"
  --checkpoint "${CHECKPOINT}" --device distributed --window-steps "${WINDOW_STEPS}"
)
if [[ -n "${MAX_FRAMES}" ]]; then
  REPLAY_ARGS+=(--max-frames "${MAX_FRAMES}")
fi
CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8} \
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun --standalone --nproc_per_node="${NPROC}" \
  scripts/eval/replay_stage25_d3r_lseg.py \
  "${REPLAY_ARGS[@]}"

FAILED_STAGE=d3r_analysis
python3 scripts/eval/analyze_stage25_d3r_lseg.py \
  --root "${WORK_DIR}/d3r_audit" --output "${WORK_DIR}/stage25_d3r_report.json"

FAILED_STAGE=return_packaging
printf '%s\n' "0" > "${WORK_DIR}/EXIT_STATUS.txt"
git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
mv "${WORK_DIR}" "${SUCCESS_DIR}"
COMPLETE=1
echo "STAGE25_D3R_STATUS=complete"
echo "DEST=$(readlink -f "${SUCCESS_DIR}")"
du -sh "${SUCCESS_DIR}"
