#!/usr/bin/env bash
# Stage23A.1: reference-path flat control with surface-only diagnostics.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

CHECKPOINT=${STAGE21C_SCORER_CHECKPOINT:?Set STAGE21C_SCORER_CHECKPOINT to seed_53/best.pt}
MANIFEST=${STAGE23A_FLAT_MANIFEST:-scripts/eval/manifests/stage23a_flat_control3_episode_seed_replay.json}
REFERENCE_ROOT=${STAGE23A_REFERENCE_ROOT:-results/stage_17/stage21c_path_reobserve_active_paired500_overnight_failure_return_paired500_20260817_015559/partial_runs/control}
RETURN_ROOT=${STAGE21_RETURN_ROOT:-results/stage_17}
PIPELINE_TAG=${STAGE21_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
CUDA_DEVICES=${STAGE21_CUDA_VISIBLE_DEVICES:-0,1,2}
RUN_NAME="compare_vlmap_stage23a_flat_control3_${PIPELINE_TAG}"
RUN_ROOT="logs/habitat/${RUN_NAME}"
WORK_DIR="${RETURN_ROOT}/stage23a_flat_control3_running_${PIPELINE_TAG}"
SUCCESS_DEST="${RETURN_ROOT}/stage23a_flat_control3_return_${PIPELINE_TAG}"
FAILURE_DEST="${RETURN_ROOT}/stage23a_flat_control3_failure_return_${PIPELINE_TAG}"
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
  echo "STAGE23A_FLAT_CONTROL_STATUS=failed"
  echo "FAILED_STAGE=${FAILED_STAGE}"
  echo "DEST=$(readlink -f "${FAILURE_DEST}")"
}
trap package_failure EXIT

test -f "${CHECKPOINT}"
test -f "${MANIFEST}"
test -f "${REFERENCE_ROOT}/progress.json"
test ! -e "${WORK_DIR}"
test ! -e "${SUCCESS_DEST}"
test ! -e "${FAILURE_DEST}"
test ! -e "${RUN_ROOT}"
mkdir -p "${WORK_DIR}/episode_manifests"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

FAILED_STAGE=flat_control_shadow_evaluation
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
STAGE21_EPISODE_IDS="${MANIFEST}" STAGE21_RUN_NAME="${RUN_NAME}" \
STAGE21_EPISODE_SEED_REPLAY_MANIFEST="${MANIFEST}" \
STAGE21C_SCORER_CHECKPOINT="${CHECKPOINT}" STAGE21C_SCORER_DEVICE=cpu \
STAGE23A_EVAL_PORT=${STAGE23A_EVAL_PORT:-2573} \
NPROC_PER_NODE=3 MASTER_PORT=${STAGE23A_MASTER_PORT:-2574} \
bash scripts/eval/bash/stage21_torchrun_eval.sh \
  --config scripts/eval/configs/habitat_dual_system_vlmap_stage23a_pose_occ_audit_cfg.py

FAILED_STAGE=flat_control_automatic_audit
python3 scripts/eval/analyze_stage23a_pose_occ_audit.py \
  --run-root "${RUN_ROOT}" --manifest "${MANIFEST}" \
  --reference-root "${REFERENCE_ROOT}" \
  --output "${RUN_ROOT}/stage23a_flat_control_audit.json" \
  --min-flat-episodes 2 --min-height-change-episodes 0 --require-all

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
Stage23A.1 replays three short train episodes whose R2R reference paths have
zero vertical range. It measures whether Frozen S2's executed RGB-D trajectory
also stays flat, and emits current/oracle-height surface-only and evidence views.
No GT pose enters navigation, candidate generation, triage, prompts, or actions.
EOF
find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
mv "${WORK_DIR}" "${SUCCESS_DEST}"
PIPELINE_COMPLETE=1

echo "STAGE23A_FLAT_CONTROL_STATUS=complete"
echo "DEST=$(readlink -f "${SUCCESS_DEST}")"
du -sh "${SUCCESS_DEST}"
