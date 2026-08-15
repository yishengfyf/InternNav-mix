#!/usr/bin/env bash
# Replay the six known strict loop events and audit map->visible-free-pixel bridging.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

CHECKPOINT=${STAGE21C_SCORER_CHECKPOINT:?Set STAGE21C_SCORER_CHECKPOINT to seed_53/best.pt}
MANIFEST=${STAGE21C_PROJECTION_MANIFEST:-scripts/eval/manifests/stage21c_strict_active_5_episode_ids.json}
REFERENCE_ROOT=${STAGE21C_REPLAY_REFERENCE_ROOT:-results/stage_17/stage21c_multitask_scorer_shadow500_return_20260814_134102}
RETURN_ROOT=${STAGE21_RETURN_ROOT:-results/stage_17}
PIPELINE_TAG=${STAGE21_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
CUDA_DEVICES=${STAGE21_CUDA_VISIBLE_DEVICES:-0,1,2,3}
RUN_NAME="compare_vlmap_stage21c_projection_bridge_shadow5_${PIPELINE_TAG}"
RUN_ROOT="logs/habitat/${RUN_NAME}"
RUNNING_NAME="stage21c_projection_bridge_shadow5_running_${PIPELINE_TAG}"
SUCCESS_NAME="stage21c_projection_bridge_shadow5_return_${PIPELINE_TAG}"
FAILURE_NAME="stage21c_projection_bridge_shadow5_failure_return_${PIPELINE_TAG}"
WORK_DIR="${RETURN_ROOT}/${RUNNING_NAME}"
SUCCESS_DEST="${RETURN_ROOT}/${SUCCESS_NAME}"
FAILURE_DEST="${RETURN_ROOT}/${FAILURE_NAME}"
FAILED_STAGE=initialization
PIPELINE_COMPLETE=0

package_failure() {
  local exit_status=$?
  if [[ "${PIPELINE_COMPLETE}" == "1" || "${exit_status}" == "0" ]]; then return; fi
  if [[ -d "${WORK_DIR}" ]]; then
    if [[ -d "${RUN_ROOT}" ]]; then cp -a "${RUN_ROOT}/." "${WORK_DIR}/"; fi
    printf '%s\n' "${FAILED_STAGE}" > "${WORK_DIR}/FAILED_STAGE.txt"
    printf '%s\n' "${exit_status}" > "${WORK_DIR}/EXIT_STATUS.txt"
    git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
    git status --short > "${WORK_DIR}/git_status_short.txt"
    mv "${WORK_DIR}" "${FAILURE_DEST}"
    find "${FAILURE_DEST}" -type f | sort > "${FAILURE_DEST}/RETURN_MANIFEST.txt"
    echo "STAGE21C_PROJECTION_BRIDGE_STATUS=failed"
    echo "FAILED_STAGE=${FAILED_STAGE}"
    echo "DEST=$(readlink -f "${FAILURE_DEST}")"
  fi
}
trap package_failure EXIT

mkdir -p "${RETURN_ROOT}"
test -f "${CHECKPOINT}"
test -f "${MANIFEST}"
test -f "${REFERENCE_ROOT}/progress.json"
test ! -e "${WORK_DIR}"; test ! -e "${SUCCESS_DEST}"; test ! -e "${FAILURE_DEST}"
test ! -e "${RUN_ROOT}"
mkdir -p "${WORK_DIR}/episode_manifests"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

FAILED_STAGE=projection_bridge_evaluation
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
STAGE21_EPISODE_IDS="${MANIFEST}" STAGE21_RUN_NAME="${RUN_NAME}" \
STAGE21_EPISODE_SEED_REPLAY_MANIFEST="${MANIFEST}" \
STAGE21C_SCORER_CHECKPOINT="${CHECKPOINT}" STAGE21C_SCORER_DEVICE=cpu \
STAGE21C_PROJECTION_MAX_ANGLE_DEG=${STAGE21C_PROJECTION_MAX_ANGLE_DEG:-30.0} \
STAGE21C_PROJECTION_EVAL_PORT=${STAGE21C_PROJECTION_EVAL_PORT:-2491} \
NPROC_PER_NODE=4 MASTER_PORT=${STAGE21C_PROJECTION_MASTER_PORT:-2492} \
bash scripts/eval/bash/stage21_torchrun_eval.sh \
  --config scripts/eval/configs/habitat_dual_system_vlmap_stage21c_projection_bridge_shadow_cfg.py

FAILED_STAGE=projection_bridge_audit
python3 scripts/eval/analyze_stage21c_projection_bridge_shadow.py \
  --run-root "${RUN_ROOT}" --expected-episodes 5 \
  --seed-manifest "${MANIFEST}" --reference-root "${REFERENCE_ROOT}" \
  --output "${RUN_ROOT}/stage21c_projection_bridge_shadow_audit.json" \
  --require-all

FAILED_STAGE=return_packaging
cp -a "${RUN_ROOT}/." "${WORK_DIR}/"
cp -a "${MANIFEST}" "${WORK_DIR}/episode_manifests/"
printf '%s\n' "0" > "${WORK_DIR}/EXIT_STATUS.txt"
printf '%s\n' "${CHECKPOINT}" > "${WORK_DIR}/SOURCE_CHECKPOINT.txt"
printf '%s\n' "${REFERENCE_ROOT}" > "${WORK_DIR}/SOURCE_REFERENCE_ROOT.txt"
sha256sum "${MANIFEST}" > "${WORK_DIR}/MANIFEST_SHA256.txt"
git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
git status --short > "${WORK_DIR}/git_status_short.txt"
cat > "${WORK_DIR}/EXPERIMENT_SCOPE.txt" <<'EOF'
Stage21c five-episode explicit-seed projection-bridge shadow.
Frozen S2/NextDiT navigation and frozen Stage21 scorer.
The exact map candidate, old fixed directional pixel, and visible free local proxies are audited.
No bridge proposal changes S2 output or enters the action queue.
EOF
find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
mv "${WORK_DIR}" "${SUCCESS_DEST}"
PIPELINE_COMPLETE=1

echo "STAGE21C_PROJECTION_BRIDGE_STATUS=complete"
echo "RETURN_NAME=${SUCCESS_NAME}"
echo "DEST=$(readlink -f "${SUCCESS_DEST}")"
du -sh "${SUCCESS_DEST}"
