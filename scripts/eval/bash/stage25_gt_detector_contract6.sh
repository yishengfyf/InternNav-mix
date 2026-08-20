#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"
CHECKPOINT=${STAGE21C_SCORER_CHECKPOINT:?Set STAGE21C_SCORER_CHECKPOINT to seed_53/best.pt}
MANIFEST=${STAGE25_MANIFEST:-scripts/eval/manifests/stage25_gt_contract6_episode_seed_replay.json}
CONFIG=${STAGE25_CONFIG:-scripts/eval/configs/habitat_dual_system_vlmap_stage25_gt_detector_cfg.py}
RETURN_ROOT=${STAGE21_RETURN_ROOT:-results/stage_17}
PIPELINE_TAG=${STAGE21_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
CUDA_DEVICES=${STAGE21_CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC=${STAGE25_NPROC_PER_NODE:-4}
RUN_NAME="compare_vlmap_stage25_gt_detector_${PIPELINE_TAG}"
RUN_ROOT="logs/habitat/${RUN_NAME}"
WORK_DIR="${RETURN_ROOT}/stage25_gt_detector_running_${PIPELINE_TAG}"
SUCCESS_DEST="${RETURN_ROOT}/stage25_gt_detector_return_${PIPELINE_TAG}"
FAILURE_DEST="${RETURN_ROOT}/stage25_gt_detector_failure_return_${PIPELINE_TAG}"
FAILED_STAGE=initialization
PIPELINE_COMPLETE=0

package_failure() {
  local status=$?
  if [[ "${PIPELINE_COMPLETE}" == 1 || "${status}" == 0 ]]; then return; fi
  mkdir -p "${WORK_DIR}"
  [[ -d "${RUN_ROOT}" ]] && cp -a "${RUN_ROOT}/." "${WORK_DIR}/"
  printf '%s\n' "${FAILED_STAGE}" > "${WORK_DIR}/FAILED_STAGE.txt"
  printf '%s\n' "${status}" > "${WORK_DIR}/EXIT_STATUS.txt"
  git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
  git status --short > "${WORK_DIR}/git_status_short.txt"
  mv "${WORK_DIR}" "${FAILURE_DEST}"
  find "${FAILURE_DEST}" -type f | sort > "${FAILURE_DEST}/RETURN_MANIFEST.txt"
  echo "STAGE25_STATUS=failed"
  echo "FAILED_STAGE=${FAILED_STAGE}"
  echo "DEST=$(readlink -f "${FAILURE_DEST}")"
}
trap package_failure EXIT

test -f "${CHECKPOINT}"
test -f "${MANIFEST}"
test -d "${STAGE24D_VLMAPS_REPO:-/home/yifeifeng/workspace/vlmaps}"
test -f "${STAGE24D_LSEG_CHECKPOINT:-results/stage_17/stage24d_lseg_safe_checkpoint_20260820/demo_e200_state_dict.pt}"
test ! -e "${WORK_DIR}"
test ! -e "${SUCCESS_DEST}"
test ! -e "${FAILURE_DEST}"
test ! -e "${RUN_ROOT}"
mkdir -p "${WORK_DIR}/episode_manifests"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

FAILED_STAGE=targeted_tests
python3 -m pytest -q \
  tests/unit_test/test_replay_ledger_lossless.py \
  tests/unit_test/test_stage25_gt_detector.py \
  tests/unit_test/test_lseg_online_shadow.py

FAILED_STAGE=frozen_s2_gt_contract_evaluation
CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8} \
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
STAGE21_EPISODE_IDS="${MANIFEST}" STAGE21_RUN_NAME="${RUN_NAME}" \
STAGE21_EPISODE_SEED_REPLAY_MANIFEST="${MANIFEST}" \
STAGE21C_SCORER_CHECKPOINT="${CHECKPOINT}" STAGE21C_SCORER_DEVICE=cpu \
STAGE25_EVAL_PORT=${STAGE25_EVAL_PORT:-2795} \
NPROC_PER_NODE="${NPROC}" MASTER_PORT=${STAGE25_MASTER_PORT:-2796} \
bash scripts/eval/bash/stage21_torchrun_eval.sh --config "${CONFIG}"

FAILED_STAGE=contract_and_detector_audit
python3 scripts/eval/analyze_stage24a_replay_ledger.py \
  --run-root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/stage25_replay_ledger_integrity.json"
python3 scripts/eval/analyze_stage25_gt_detector.py \
  --run-root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/stage25_detector_audit" --require-all

FAILED_STAGE=return_packaging
cp -a "${RUN_ROOT}/." "${WORK_DIR}/"
cp -a "${MANIFEST}" "${WORK_DIR}/episode_manifests/"
printf '%s\n' 0 > "${WORK_DIR}/EXIT_STATUS.txt"
printf '%s\n' "${CHECKPOINT}" > "${WORK_DIR}/SOURCE_CHECKPOINT.txt"
sha256sum "${MANIFEST}" > "${WORK_DIR}/MANIFEST_SHA256.txt"
git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
git status --short > "${WORK_DIR}/git_status_short.txt"
find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
mv "${WORK_DIR}" "${SUCCESS_DEST}"
PIPELINE_COMPLETE=1
echo "STAGE25_STATUS=complete"
echo "DEST=$(readlink -f "${SUCCESS_DEST}")"
du -sh "${SUCCESS_DEST}"
