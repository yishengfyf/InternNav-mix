#!/usr/bin/env bash
# Stage22C: audit the exact Stage22A routes on pitch-aware OCC.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

CHECKPOINT=${STAGE21C_SCORER_CHECKPOINT:?Set STAGE21C_SCORER_CHECKPOINT to seed_53/best.pt}
EPISODE_MANIFEST=${STAGE22_ROUTE_MANIFEST:-scripts/eval/manifests/stage22a_paired500_strict11_episode_seed_replay.json}
FIXED_ROUTE_MANIFEST=${STAGE22_FIXED_ROUTE_MANIFEST:-scripts/eval/manifests/stage22c_stage22a_fixed_route12.json}
NAV_REFERENCE_ROOT=${STAGE22_ROUTE_REFERENCE_ROOT:-results/stage_17/stage21c_path_reobserve_active_paired500_overnight_failure_return_paired500_20260817_015559/partial_runs/control}
STAGE22A_ROOT=${STAGE22A_ROOT:?Set STAGE22A_ROOT to the returned Stage22A run root}
RETURN_ROOT=${STAGE21_RETURN_ROOT:-results/stage_17}
PIPELINE_TAG=${STAGE21_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
CUDA_DEVICES=${STAGE21_CUDA_VISIBLE_DEVICES:-0,1,2,3}
EXPECTED_EPISODES=11
FIXED_ROUTE_LABEL=${STAGE22_FIXED_ROUTE_LABEL:-stage22c_fixed_route_pitch_occ_shadow11}
FIXED_ROUTE_CONFIG=${STAGE22_FIXED_ROUTE_CONFIG:-scripts/eval/configs/habitat_dual_system_vlmap_stage22c_fixed_route_pitch_occ_shadow_cfg.py}
FIXED_ROUTE_AUDIT_NAME=${STAGE22_FIXED_ROUTE_AUDIT_NAME:-stage22c_fixed_route_pitch_occ_shadow_audit.json}
RUN_NAME="compare_vlmap_${FIXED_ROUTE_LABEL}_${PIPELINE_TAG}"
RUN_ROOT="logs/habitat/${RUN_NAME}"
WORK_DIR="${RETURN_ROOT}/${FIXED_ROUTE_LABEL}_running_${PIPELINE_TAG}"
SUCCESS_DEST="${RETURN_ROOT}/${FIXED_ROUTE_LABEL}_return_${PIPELINE_TAG}"
FAILURE_DEST="${RETURN_ROOT}/${FIXED_ROUTE_LABEL}_failure_return_${PIPELINE_TAG}"
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
  echo "STAGE22C_FIXED_ROUTE_PITCH_OCC_STATUS=failed"
  echo "FAILED_STAGE=${FAILED_STAGE}"
  echo "DEST=$(readlink -f "${FAILURE_DEST}")"
}
trap package_failure EXIT

test -f "${CHECKPOINT}"
test -f "${EPISODE_MANIFEST}"
test -f "${FIXED_ROUTE_MANIFEST}"
test -f "${NAV_REFERENCE_ROOT}/progress.json"
test -f "${STAGE22A_ROOT}/stage22a_executed_route_occ_shadow_audit.json"
test ! -e "${WORK_DIR}"
test ! -e "${SUCCESS_DEST}"
test ! -e "${FAILURE_DEST}"
test ! -e "${RUN_ROOT}"
mkdir -p "${WORK_DIR}/episode_manifests"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

FAILED_STAGE=fixed_route_pitch_occ_shadow_evaluation
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
STAGE21_EPISODE_IDS="${EPISODE_MANIFEST}" STAGE21_RUN_NAME="${RUN_NAME}" \
STAGE21_EPISODE_SEED_REPLAY_MANIFEST="${EPISODE_MANIFEST}" \
STAGE21C_SCORER_CHECKPOINT="${CHECKPOINT}" STAGE21C_SCORER_DEVICE=cpu \
STAGE22_FIXED_ROUTE_EVAL_PORT=${STAGE22_FIXED_ROUTE_EVAL_PORT:-2565} \
NPROC_PER_NODE=4 MASTER_PORT=${STAGE22_FIXED_ROUTE_MASTER_PORT:-2566} \
bash scripts/eval/bash/stage21_torchrun_eval.sh \
  --config "${FIXED_ROUTE_CONFIG}"

FAILED_STAGE=fixed_route_pitch_occ_automatic_audit
python3 scripts/eval/analyze_stage22c_fixed_route_pitch_occ.py \
  --run-root "${RUN_ROOT}" --expected-episodes "${EXPECTED_EPISODES}" \
  --seed-manifest "${EPISODE_MANIFEST}" \
  --navigation-reference-root "${NAV_REFERENCE_ROOT}" \
  --stage22a-root "${STAGE22A_ROOT}" \
  --fixed-route-manifest "${FIXED_ROUTE_MANIFEST}" \
  --output "${RUN_ROOT}/${FIXED_ROUTE_AUDIT_NAME}" \
  ${STAGE22_REQUIRE_EVIDENCE:+--require-evidence} \
  ${STAGE22_REQUIRE_HEIGHT_EVIDENCE:+--require-height-evidence} \
  --require-all

FAILED_STAGE=return_packaging
cp -a "${RUN_ROOT}/." "${WORK_DIR}/"
cp -a "${EPISODE_MANIFEST}" "${WORK_DIR}/episode_manifests/"
cp -a "${FIXED_ROUTE_MANIFEST}" "${WORK_DIR}/episode_manifests/"
printf '%s\n' "0" > "${WORK_DIR}/EXIT_STATUS.txt"
printf '%s\n' "${CHECKPOINT}" > "${WORK_DIR}/SOURCE_CHECKPOINT.txt"
printf '%s\n' "${NAV_REFERENCE_ROOT}" > "${WORK_DIR}/SOURCE_NAVIGATION_REFERENCE_ROOT.txt"
printf '%s\n' "${STAGE22A_ROOT}" > "${WORK_DIR}/SOURCE_STAGE22A_ROOT.txt"
sha256sum "${EPISODE_MANIFEST}" "${FIXED_ROUTE_MANIFEST}" > "${WORK_DIR}/MANIFEST_SHA256.txt"
git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
git status --short > "${WORK_DIR}/git_status_short.txt"
cat > "${WORK_DIR}/EXPERIMENT_SCOPE.txt" <<EOF
${FIXED_ROUTE_LABEL} replays the same 11 Frozen episodes and audits the exact
12 Stage22A trigger/anchor/source routes on pitch-aware OCC. Dynamic candidate
and triage changes are logged separately. No route cell, output, or action is
modified. Evidence audit: ${STAGE22_REQUIRE_EVIDENCE:-disabled}.
Height-aligned evidence audit: ${STAGE22_REQUIRE_HEIGHT_EVIDENCE:-disabled}.
EOF
find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
mv "${WORK_DIR}" "${SUCCESS_DEST}"
PIPELINE_COMPLETE=1

echo "STAGE22C_FIXED_ROUTE_PITCH_OCC_STATUS=complete"
echo "RETURN_NAME=$(basename "${SUCCESS_DEST}")"
echo "DEST=$(readlink -f "${SUCCESS_DEST}")"
du -sh "${SUCCESS_DEST}"
