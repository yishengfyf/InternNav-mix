#!/usr/bin/env bash
# Stage22B: exact Stage22A replay with pitch-aware OCC depth projection.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

CHECKPOINT=${STAGE21C_SCORER_CHECKPOINT:?Set STAGE21C_SCORER_CHECKPOINT to seed_53/best.pt}
MANIFEST=${STAGE22_ROUTE_MANIFEST:-scripts/eval/manifests/stage22a_paired500_strict11_episode_seed_replay.json}
NAV_REFERENCE_ROOT=${STAGE22_ROUTE_REFERENCE_ROOT:-results/stage_17/stage21c_path_reobserve_active_paired500_overnight_failure_return_paired500_20260817_015559/partial_runs/control}
STAGE22A_ROOT=${STAGE22A_ROOT:?Set STAGE22A_ROOT to the returned Stage22A run root}
RETURN_ROOT=${STAGE21_RETURN_ROOT:-results/stage_17}
PIPELINE_TAG=${STAGE21_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
CUDA_DEVICES=${STAGE21_CUDA_VISIBLE_DEVICES:-0,1,2,3}
EXPECTED_EPISODES=11
RUN_NAME="compare_vlmap_stage22b_pitch_aware_occ_shadow11_${PIPELINE_TAG}"
RUN_ROOT="logs/habitat/${RUN_NAME}"
WORK_DIR="${RETURN_ROOT}/stage22b_pitch_aware_occ_shadow11_running_${PIPELINE_TAG}"
SUCCESS_DEST="${RETURN_ROOT}/stage22b_pitch_aware_occ_shadow11_return_${PIPELINE_TAG}"
FAILURE_DEST="${RETURN_ROOT}/stage22b_pitch_aware_occ_shadow11_failure_return_${PIPELINE_TAG}"
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
  echo "STAGE22B_PITCH_AWARE_OCC_STATUS=failed"
  echo "FAILED_STAGE=${FAILED_STAGE}"
  echo "DEST=$(readlink -f "${FAILURE_DEST}")"
}
trap package_failure EXIT

test -f "${CHECKPOINT}"
test -f "${MANIFEST}"
test -f "${NAV_REFERENCE_ROOT}/progress.json"
test -f "${STAGE22A_ROOT}/stage22a_executed_route_occ_shadow_audit.json"
test ! -e "${WORK_DIR}"
test ! -e "${SUCCESS_DEST}"
test ! -e "${FAILURE_DEST}"
test ! -e "${RUN_ROOT}"
mkdir -p "${WORK_DIR}/episode_manifests"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

FAILED_STAGE=route_occ_shadow_evaluation
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
STAGE21_EPISODE_IDS="${MANIFEST}" STAGE21_RUN_NAME="${RUN_NAME}" \
STAGE21_EPISODE_SEED_REPLAY_MANIFEST="${MANIFEST}" \
STAGE21C_SCORER_CHECKPOINT="${CHECKPOINT}" STAGE21C_SCORER_DEVICE=cpu \
STAGE22_PITCH_EVAL_PORT=${STAGE22_PITCH_EVAL_PORT:-2563} \
NPROC_PER_NODE=4 MASTER_PORT=${STAGE22_PITCH_MASTER_PORT:-2564} \
bash scripts/eval/bash/stage21_torchrun_eval.sh \
  --config scripts/eval/configs/habitat_dual_system_vlmap_stage22b_pitch_aware_occ_shadow_cfg.py

FAILED_STAGE=route_occ_automatic_audit
python3 scripts/eval/analyze_stage22b_pitch_aware_occ.py \
  --run-root "${RUN_ROOT}" --expected-episodes "${EXPECTED_EPISODES}" \
  --seed-manifest "${MANIFEST}" --navigation-reference-root "${NAV_REFERENCE_ROOT}" \
  --stage22a-root "${STAGE22A_ROOT}" \
  --output "${RUN_ROOT}/stage22b_pitch_aware_occ_shadow_audit.json" --require-all

FAILED_STAGE=return_packaging
cp -a "${RUN_ROOT}/." "${WORK_DIR}/"
cp -a "${MANIFEST}" "${WORK_DIR}/episode_manifests/"
printf '%s\n' "0" > "${WORK_DIR}/EXIT_STATUS.txt"
printf '%s\n' "${CHECKPOINT}" > "${WORK_DIR}/SOURCE_CHECKPOINT.txt"
printf '%s\n' "${NAV_REFERENCE_ROOT}" > "${WORK_DIR}/SOURCE_NAVIGATION_REFERENCE_ROOT.txt"
printf '%s\n' "${STAGE22A_ROOT}" > "${WORK_DIR}/SOURCE_STAGE22A_ROOT.txt"
sha256sum "${MANIFEST}" > "${WORK_DIR}/MANIFEST_SHA256.txt"
git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
git status --short > "${WORK_DIR}/git_status_short.txt"
cat > "${WORK_DIR}/EXPERIMENT_SCOPE.txt" <<'EOF'
Stage22B exact replay of Stage22A strict11 with only camera-pitch-aware OCC
depth back-projection enabled. Frozen navigation, strict detector, candidates,
ranking, triage, S2 outputs and actions remain unchanged and all route audits
remain shadow-only.
EOF
find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
mv "${WORK_DIR}" "${SUCCESS_DEST}"
PIPELINE_COMPLETE=1

echo "STAGE22B_PITCH_AWARE_OCC_STATUS=complete"
echo "RETURN_NAME=$(basename "${SUCCESS_DEST}")"
echo "DEST=$(readlink -f "${SUCCESS_DEST}")"
du -sh "${SUCCESS_DEST}"
