#!/usr/bin/env bash
# Stage23A Layer1/2 smoke: complete sensor pose and observed-volume OCC audit.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

CHECKPOINT=${STAGE21C_SCORER_CHECKPOINT:?Set STAGE21C_SCORER_CHECKPOINT to seed_53/best.pt}
MANIFEST=${STAGE23A_SENSOR_MANIFEST:-scripts/eval/manifests/stage23a_sensor_smoke1_episode_seed_replay.json}
CONFIG=${STAGE23A_SENSOR_CONFIG:-scripts/eval/configs/habitat_dual_system_vlmap_stage23a_sensor_pose_occ_cfg.py}
RETURN_ROOT=${STAGE21_RETURN_ROOT:-results/stage_17}
PIPELINE_TAG=${STAGE21_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
CUDA_DEVICES=${STAGE21_CUDA_VISIBLE_DEVICES:-0,1,2}
NPROC=${STAGE23A_NPROC_PER_NODE:-3}
RUN_NAME="compare_vlmap_stage23a_sensor_pose_smoke1_${PIPELINE_TAG}"
RUN_ROOT="logs/habitat/${RUN_NAME}"
WORK_DIR="${RETURN_ROOT}/stage23a_sensor_pose_smoke1_running_${PIPELINE_TAG}"
SUCCESS_DEST="${RETURN_ROOT}/stage23a_sensor_pose_smoke1_return_${PIPELINE_TAG}"
FAILURE_DEST="${RETURN_ROOT}/stage23a_sensor_pose_smoke1_failure_return_${PIPELINE_TAG}"
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
  echo "STAGE23A_SENSOR_POSE_STATUS=failed"
  echo "FAILED_STAGE=${FAILED_STAGE}"
  echo "DEST=$(readlink -f "${FAILURE_DEST}")"
}
trap package_failure EXIT

test -f "${CHECKPOINT}"
test -f "${MANIFEST}"
test ! -e "${WORK_DIR}"
test ! -e "${SUCCESS_DEST}"
test ! -e "${FAILURE_DEST}"
test ! -e "${RUN_ROOT}"
mkdir -p "${WORK_DIR}/episode_manifests"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

FAILED_STAGE=sensor_pose_shadow_evaluation
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
STAGE21_EPISODE_IDS="${MANIFEST}" STAGE21_RUN_NAME="${RUN_NAME}" \
STAGE21_EPISODE_SEED_REPLAY_MANIFEST="${MANIFEST}" \
STAGE21C_SCORER_CHECKPOINT="${CHECKPOINT}" STAGE21C_SCORER_DEVICE=cpu \
STAGE23A_EVAL_PORT=${STAGE23A_EVAL_PORT:-2573} \
NPROC_PER_NODE="${NPROC}" MASTER_PORT=${STAGE23A_MASTER_PORT:-2574} \
bash scripts/eval/bash/stage21_torchrun_eval.sh \
  --config "${CONFIG}"

FAILED_STAGE=sensor_pose_automatic_audit
python3 scripts/eval/analyze_stage23a_sensor_pose_occ.py \
  --run-root "${RUN_ROOT}" --manifest "${MANIFEST}" \
  --output "${RUN_ROOT}/stage23a_sensor_pose_occ_audit.json" --require-all \
  ${STAGE23A_ANALYZER_EXTRA_ARGS:-}

FAILED_STAGE=return_packaging
cp -a "${RUN_ROOT}/." "${WORK_DIR}/"
cp -a "${MANIFEST}" "${WORK_DIR}/episode_manifests/"
printf '%s\n' "0" > "${WORK_DIR}/EXIT_STATUS.txt"
printf '%s\n' "${CHECKPOINT}" > "${WORK_DIR}/SOURCE_CHECKPOINT.txt"
sha256sum "${MANIFEST}" > "${WORK_DIR}/MANIFEST_SHA256.txt"
git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
git status --short > "${WORK_DIR}/git_status_short.txt"
cat > "${WORK_DIR}/EXPERIMENT_SCOPE.txt" <<'EOF'
Stage23A sensor-pose smoke uses Habitat's complete depth-sensor 6DoF pose as
an audit-only reference. It compares current GPS+compass/pitch mapping with
the independent observed-volume branch. GT never enters navigation or action.
EOF
find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
mv "${WORK_DIR}" "${SUCCESS_DEST}"
PIPELINE_COMPLETE=1

echo "STAGE23A_SENSOR_POSE_STATUS=complete"
echo "DEST=$(readlink -f "${SUCCESS_DEST}")"
du -sh "${SUCCESS_DEST}"
