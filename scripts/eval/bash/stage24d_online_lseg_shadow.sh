#!/usr/bin/env bash
# Stage24D same-process online LSeg shadow on fixed replay episodes.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

CHECKPOINT=${STAGE21C_SCORER_CHECKPOINT:?Set STAGE21C_SCORER_CHECKPOINT to seed_53/best.pt}
MANIFEST=${STAGE24D_MANIFEST:-scripts/eval/manifests/stage23a_sensor_smoke1_episode_seed_replay.json}
CONFIG=${STAGE24D_CONFIG:-scripts/eval/configs/habitat_dual_system_vlmap_stage24d_online_lseg_shadow_cfg.py}
BASELINE_ROOT=${STAGE24D_BASELINE_ROOT:?Set STAGE24D_BASELINE_ROOT to the matching Stage24A return directory}
RETURN_ROOT=${STAGE21_RETURN_ROOT:-results/stage_17}
PIPELINE_TAG=${STAGE21_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
CUDA_DEVICES=${STAGE21_CUDA_VISIBLE_DEVICES:-0}
NPROC=${STAGE24D_NPROC_PER_NODE:-1}
RUN_NAME="compare_vlmap_stage24d_online_lseg_shadow_${PIPELINE_TAG}"
RUN_ROOT="logs/habitat/${RUN_NAME}"
WORK_DIR="${RETURN_ROOT}/stage24d_online_lseg_shadow_running_${PIPELINE_TAG}"
SUCCESS_DEST="${RETURN_ROOT}/stage24d_online_lseg_shadow_return_${PIPELINE_TAG}"
FAILURE_DEST="${RETURN_ROOT}/stage24d_online_lseg_shadow_failure_return_${PIPELINE_TAG}"
FAILED_STAGE=initialization
PIPELINE_COMPLETE=0

package_failure() {
  local status=$?
  if [[ "${PIPELINE_COMPLETE}" == "1" || "${status}" == "0" ]]; then return; fi
  mkdir -p "${WORK_DIR}"
  [[ -d "${RUN_ROOT}" ]] && cp -a "${RUN_ROOT}/." "${WORK_DIR}/"
  printf '%s\n' "${FAILED_STAGE}" > "${WORK_DIR}/FAILED_STAGE.txt"
  printf '%s\n' "${status}" > "${WORK_DIR}/EXIT_STATUS.txt"
  git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
  git status --short > "${WORK_DIR}/git_status_short.txt"
  mv "${WORK_DIR}" "${FAILURE_DEST}"
  find "${FAILURE_DEST}" -type f | sort > "${FAILURE_DEST}/RETURN_MANIFEST.txt"
  echo "STAGE24D_ONLINE_LSEG_STATUS=failed"
  echo "FAILED_STAGE=${FAILED_STAGE}"
  echo "DEST=$(readlink -f "${FAILURE_DEST}")"
}
trap package_failure EXIT

test -f "${CHECKPOINT}"
test -f "${MANIFEST}"
test -d "${BASELINE_ROOT}"
test -d "${STAGE24D_VLMAPS_REPO:-/home/yifeifeng/workspace/vlmaps}"
test -f "${STAGE24D_LSEG_CHECKPOINT:-/home/yifeifeng/workspace/vlmaps/vlmaps/lseg/checkpoints/demo_e200.ckpt}"
test ! -e "${WORK_DIR}"
test ! -e "${SUCCESS_DEST}"
test ! -e "${FAILURE_DEST}"
test ! -e "${RUN_ROOT}"
mkdir -p "${WORK_DIR}/episode_manifests"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

FAILED_STAGE=online_lseg_shadow_evaluation
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
STAGE21_EPISODE_IDS="${MANIFEST}" STAGE21_RUN_NAME="${RUN_NAME}" \
STAGE21_EPISODE_SEED_REPLAY_MANIFEST="${MANIFEST}" \
STAGE21C_SCORER_CHECKPOINT="${CHECKPOINT}" STAGE21C_SCORER_DEVICE=cpu \
STAGE24D_EVAL_PORT=${STAGE24D_EVAL_PORT:-2695} \
NPROC_PER_NODE="${NPROC}" MASTER_PORT=${STAGE24D_MASTER_PORT:-2696} \
bash scripts/eval/bash/stage21_torchrun_eval.sh --config "${CONFIG}"

FAILED_STAGE=online_lseg_shadow_audit
python3 scripts/eval/analyze_stage24d_online_lseg_shadow.py \
  --run-root "${RUN_ROOT}" --baseline-root "${BASELINE_ROOT}" \
  --output "${RUN_ROOT}/stage24d_online_lseg_shadow_audit.json" --require-all

FAILED_STAGE=return_packaging
cp -a "${RUN_ROOT}/." "${WORK_DIR}/"
cp -a "${MANIFEST}" "${WORK_DIR}/episode_manifests/"
printf '%s\n' "0" > "${WORK_DIR}/EXIT_STATUS.txt"
printf '%s\n' "${CHECKPOINT}" > "${WORK_DIR}/SOURCE_CHECKPOINT.txt"
printf '%s\n' "${BASELINE_ROOT}" > "${WORK_DIR}/BASELINE_ROOT.txt"
sha256sum "${MANIFEST}" > "${WORK_DIR}/MANIFEST_SHA256.txt"
git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
git status --short > "${WORK_DIR}/git_status_short.txt"
find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
mv "${WORK_DIR}" "${SUCCESS_DEST}"
PIPELINE_COMPLETE=1
echo "STAGE24D_ONLINE_LSEG_STATUS=complete"
echo "DEST=$(readlink -f "${SUCCESS_DEST}")"
du -sh "${SUCCESS_DEST}"
