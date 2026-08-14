#!/usr/bin/env bash
# Stage21d targeted 10ep data-chain smoke -> independent balanced 40ep shadow.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

EPISODES_FILE=${STAGE21_EPISODES_FILE:-data/vln_ce/raw_data/r2r/train/train.json.gz}
CHECKPOINT=${STAGE21C_SCORER_CHECKPOINT:?Set STAGE21C_SCORER_CHECKPOINT to seed_53/best.pt}
TARGETED_MANIFEST=${STAGE21D_TARGETED_MANIFEST:-scripts/eval/manifests/stage21d_context_targeted_10_episode_ids.json}
RETURN_ROOT=${STAGE21_RETURN_ROOT:-results/stage_17}
PIPELINE_TAG=${STAGE21_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
CUDA_DEVICES=${STAGE21_CUDA_VISIBLE_DEVICES:-0,1,2,3}

RUNNING_NAME="stage21d_recovery_context_shadow_10to40_running_${PIPELINE_TAG}"
SUCCESS_NAME="stage21d_recovery_context_shadow_10to40_return_${PIPELINE_TAG}"
FAILURE_NAME="stage21d_recovery_context_shadow_10to40_failure_return_${PIPELINE_TAG}"
WORK_DIR="${RETURN_ROOT}/${RUNNING_NAME}"
SUCCESS_DEST="${RETURN_ROOT}/${SUCCESS_NAME}"
FAILURE_DEST="${RETURN_ROOT}/${FAILURE_NAME}"
TARGETED_NAME="compare_vlmap_stage21d_recovery_context_targeted10_${PIPELINE_TAG}"
SHADOW40_NAME="compare_vlmap_stage21d_recovery_context_balanced40_${PIPELINE_TAG}"
TARGETED_ROOT="logs/habitat/${TARGETED_NAME}"
SHADOW40_ROOT="logs/habitat/${SHADOW40_NAME}"
MANIFEST40="${WORK_DIR}/episode_manifests/train_balanced_seed37_40_episode_ids.json"
FAILED_STAGE=initialization
PIPELINE_COMPLETE=0

write_metadata() {
  local root=$1
  git rev-parse HEAD > "${root}/git_commit.txt"
  git status --short > "${root}/git_status_short.txt"
  printf '%s\n' "${CHECKPOINT}" > "${root}/SOURCE_CHECKPOINT.txt"
  cat > "${root}/EXPERIMENT_SCOPE.txt" <<'EOF'
Stage21d recovery-conditioned S2 re-query shadow A/B.
Variants per natural S2 query: text_only and text_images.
Current frame is present only once in the base S2 input; event images are first-repeat/safe-anchor frames.
Counterfactual outputs never enter an action queue. Frozen S2/NextDiT/scorer parameters and navigation actions remain unchanged.
EOF
}

package_run() {
  local source=$1
  local dest=$2
  mkdir -p "${dest}"
  if [[ -d "${source}" ]]; then cp -a "${source}/." "${dest}/"; fi
}

package_failure() {
  local exit_status=$?
  if [[ "${PIPELINE_COMPLETE}" == "1" || "${exit_status}" == "0" ]]; then return; fi
  if [[ -d "${WORK_DIR}" ]]; then
    package_run "${TARGETED_ROOT}" "${WORK_DIR}/targeted10"
    package_run "${SHADOW40_ROOT}" "${WORK_DIR}/balanced40_partial"
    printf '%s\n' "${FAILED_STAGE}" > "${WORK_DIR}/FAILED_STAGE.txt"
    printf '%s\n' "${exit_status}" > "${WORK_DIR}/EXIT_STATUS.txt"
    write_metadata "${WORK_DIR}"
    mv "${WORK_DIR}" "${FAILURE_DEST}"
    find "${FAILURE_DEST}" -type f | sort > "${FAILURE_DEST}/RETURN_MANIFEST.txt"
    echo "STAGE21D_CONTEXT_SHADOW_STATUS=failed"
    echo "FAILED_STAGE=${FAILED_STAGE}"
    echo "DEST=$(readlink -f "${FAILURE_DEST}")"
  fi
}
trap package_failure EXIT

mkdir -p "${RETURN_ROOT}"
test -f "${EPISODES_FILE}"; test -f "${CHECKPOINT}"; test -f "${TARGETED_MANIFEST}"
test ! -e "${WORK_DIR}"; test ! -e "${SUCCESS_DEST}"; test ! -e "${FAILURE_DEST}"
test ! -e "${TARGETED_ROOT}"; test ! -e "${SHADOW40_ROOT}"
mkdir -p "${WORK_DIR}/episode_manifests"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

FAILED_STAGE=manifest_40ep
python3 scripts/eval/select_balanced_r2r_episodes.py \
  --episodes-file "${EPISODES_FILE}" --max-episodes 40 --seed 37 \
  --shuffle-within-scene --output "${MANIFEST40}" \
  --summary-output "${WORK_DIR}/episode_manifests/train_balanced_seed37_40_summary.json"
cp -a "${TARGETED_MANIFEST}" "${WORK_DIR}/episode_manifests/"

run_shadow() {
  local run_name=$1
  local manifest=$2
  local count=$3
  local eval_port=$4
  local master_port=$5
  local run_root="logs/habitat/${run_name}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
  STAGE21_EPISODE_IDS="${manifest}" STAGE21_RUN_NAME="${run_name}" \
  STAGE21C_SCORER_CHECKPOINT="${CHECKPOINT}" STAGE21C_SCORER_DEVICE=cpu \
  STAGE21D_RECOVERY_CONTEXT_VARIANTS=text_only,text_images \
  STAGE21D_RECOVERY_CONTEXT_MAX_IMAGES=2 STAGE21D_RECOVERY_CONTEXT_TTL_QUERIES=2 \
  STAGE21D_EVAL_PORT="${eval_port}" NPROC_PER_NODE=4 MASTER_PORT="${master_port}" \
  bash scripts/eval/bash/stage21_torchrun_eval.sh \
    --config scripts/eval/configs/habitat_dual_system_vlmap_stage21d_recovery_context_shadow_cfg.py
  python3 scripts/eval/analyze_stage21d_recovery_context_shadow.py \
    --run-root "${run_root}" --expected-episodes "${count}" \
    --output "${run_root}/stage21d_recovery_context_shadow_audit.json" --require-all
}

FAILED_STAGE=targeted10_evaluation_and_audit
run_shadow "${TARGETED_NAME}" "${TARGETED_MANIFEST}" 10 \
  ${STAGE21D_TARGETED_EVAL_PORT:-2485} ${STAGE21D_TARGETED_MASTER_PORT:-2486}

FAILED_STAGE=balanced40_evaluation_and_audit
run_shadow "${SHADOW40_NAME}" "${MANIFEST40}" 40 \
  ${STAGE21D_40_EVAL_PORT:-2487} ${STAGE21D_40_MASTER_PORT:-2488}

FAILED_STAGE=return_packaging
package_run "${TARGETED_ROOT}" "${WORK_DIR}/targeted10"
package_run "${SHADOW40_ROOT}" "${WORK_DIR}/balanced40_full"
cp -a "${SHADOW40_ROOT}/progress.json" "${SHADOW40_ROOT}/result.json" "${WORK_DIR}/"
cp -a "${SHADOW40_ROOT}/vlmap_safety_debug" "${WORK_DIR}/"
cp -a "${SHADOW40_ROOT}/stage21d_recovery_context_shadow_audit.json" "${WORK_DIR}/"
printf '%s\n' "0" > "${WORK_DIR}/EXIT_STATUS.txt"
write_metadata "${WORK_DIR}"
sha256sum "${WORK_DIR}/episode_manifests/"*.json > "${WORK_DIR}/MANIFEST_SHA256.txt"
find "${WORK_DIR}/vlmap_safety_debug" -type f \
  \( -name 's2_recovery_context_events.jsonl' -o -path '*/s2_action_loop_snapshots/*.jpg' \) \
  | sort > "${WORK_DIR}/VISUAL_REVIEW_FILES.txt"
mv "${WORK_DIR}" "${SUCCESS_DEST}"
find "${SUCCESS_DEST}" -type f | sort > "${SUCCESS_DEST}/RETURN_MANIFEST.txt"
PIPELINE_COMPLETE=1

echo "STAGE21D_CONTEXT_SHADOW_STATUS=complete"
echo "RETURN_NAME=${SUCCESS_NAME}"
echo "DEST=$(readlink -f "${SUCCESS_DEST}")"
du -sh "${SUCCESS_DEST}"
