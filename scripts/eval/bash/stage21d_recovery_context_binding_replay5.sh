#!/usr/bin/env bash
# Replay the five Stage21d context episodes after fixing prompt/image binding.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

CHECKPOINT=${STAGE21C_SCORER_CHECKPOINT:?Set STAGE21C_SCORER_CHECKPOINT to seed_53/best.pt}
MANIFEST=${STAGE21D_BINDING_REPLAY_MANIFEST:-scripts/eval/manifests/stage21d_context_binding_replay_5_episode_ids.json}
RETURN_ROOT=${STAGE21_RETURN_ROOT:-results/stage_17}
PIPELINE_TAG=${STAGE21_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
CUDA_DEVICES=${STAGE21_CUDA_VISIBLE_DEVICES:-0,1,2,3}
RUN_NAME="compare_vlmap_stage21d_context_binding_replay5_${PIPELINE_TAG}"
RUN_ROOT="logs/habitat/${RUN_NAME}"
RUNNING_NAME="stage21d_context_binding_replay5_running_${PIPELINE_TAG}"
SUCCESS_NAME="stage21d_context_binding_replay5_return_${PIPELINE_TAG}"
FAILURE_NAME="stage21d_context_binding_replay5_failure_return_${PIPELINE_TAG}"
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
    echo "STAGE21D_BINDING_REPLAY_STATUS=failed"
    echo "FAILED_STAGE=${FAILED_STAGE}"
    echo "DEST=$(readlink -f "${FAILURE_DEST}")"
  fi
}
trap package_failure EXIT

mkdir -p "${RETURN_ROOT}"
test -f "${CHECKPOINT}"; test -f "${MANIFEST}"
test ! -e "${WORK_DIR}"; test ! -e "${SUCCESS_DEST}"; test ! -e "${FAILURE_DEST}"
test ! -e "${RUN_ROOT}"
mkdir -p "${WORK_DIR}/episode_manifests"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

FAILED_STAGE=replay5_evaluation
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
STAGE21_EPISODE_IDS="${MANIFEST}" STAGE21_RUN_NAME="${RUN_NAME}" \
STAGE21_EPISODE_SEED_REPLAY_MANIFEST="${MANIFEST}" \
STAGE21C_SCORER_CHECKPOINT="${CHECKPOINT}" STAGE21C_SCORER_DEVICE=cpu \
STAGE21D_RECOVERY_CONTEXT_VARIANTS=text_only,text_images \
STAGE21D_RECOVERY_CONTEXT_MAX_IMAGES=2 STAGE21D_RECOVERY_CONTEXT_TTL_QUERIES=2 \
STAGE21D_RECOVERY_CONTEXT_SAVE_IMAGES=1 \
STAGE21D_EVAL_PORT=${STAGE21D_BINDING_EVAL_PORT:-2489} \
NPROC_PER_NODE=4 MASTER_PORT=${STAGE21D_BINDING_MASTER_PORT:-2490} \
bash scripts/eval/bash/stage21_torchrun_eval.sh \
  --config scripts/eval/configs/habitat_dual_system_vlmap_stage21d_recovery_context_shadow_cfg.py

FAILED_STAGE=replay5_audit
python3 scripts/eval/analyze_stage21d_recovery_context_shadow.py \
  --run-root "${RUN_ROOT}" --expected-episodes 5 \
  --seed-manifest "${MANIFEST}" \
  --output "${RUN_ROOT}/stage21d_recovery_context_shadow_audit.json" --require-all

FAILED_STAGE=return_packaging
cp -a "${RUN_ROOT}/." "${WORK_DIR}/"
cp -a "${MANIFEST}" "${WORK_DIR}/episode_manifests/"
printf '%s\n' "0" > "${WORK_DIR}/EXIT_STATUS.txt"
printf '%s\n' "${CHECKPOINT}" > "${WORK_DIR}/SOURCE_CHECKPOINT.txt"
sha256sum "${MANIFEST}" > "${WORK_DIR}/MANIFEST_SHA256.txt"
git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
git status --short > "${WORK_DIR}/git_status_short.txt"
cat > "${WORK_DIR}/EXPERIMENT_SCOPE.txt" <<'EOF'
Stage21d five-episode explicit-seed binding replay.
Frozen S2/NextDiT navigation; text-only and corrected text-images are shadow counterfactuals.
No context output enters the action queue. No active recovery is enabled.
EOF
find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
mv "${WORK_DIR}" "${SUCCESS_DEST}"
PIPELINE_COMPLETE=1

echo "STAGE21D_BINDING_REPLAY_STATUS=complete"
echo "RETURN_NAME=${SUCCESS_NAME}"
echo "DEST=$(readlink -f "${SUCCESS_DEST}")"
du -sh "${SUCCESS_DEST}"
