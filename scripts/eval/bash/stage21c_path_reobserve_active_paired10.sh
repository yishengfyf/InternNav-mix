#!/usr/bin/env bash
# Paired10 bounded path reorient/reobserve active evaluation.
# Includes the five known strict episodes plus five new explicit-seed episodes.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

CHECKPOINT=${STAGE21C_SCORER_CHECKPOINT:?Set STAGE21C_SCORER_CHECKPOINT to seed_53/best.pt}
MANIFEST=${STAGE21C_PATH_ACTIVE_MANIFEST:-scripts/eval/manifests/stage21c_path_reobserve_active_paired10.json}
REFERENCE_MANIFEST=${STAGE21C_REFERENCE_MANIFEST:-scripts/eval/manifests/stage21c_strict_active_5_episode_ids.json}
REFERENCE_ROOT=${STAGE21C_REPLAY_REFERENCE_ROOT:-results/stage_17/stage21c_multitask_scorer_shadow500_return_20260814_134102}
RETURN_ROOT=${STAGE21_RETURN_ROOT:-results/stage_17}
PIPELINE_TAG=${STAGE21_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
CUDA_DEVICES=${STAGE21_CUDA_VISIBLE_DEVICES:-0,1,2,3}
EXPECTED_EPISODES=10

RUNNING_NAME="stage21c_path_reobserve_active_paired10_running_${PIPELINE_TAG}"
SUCCESS_NAME="stage21c_path_reobserve_active_paired10_return_${PIPELINE_TAG}"
FAILURE_NAME="stage21c_path_reobserve_active_paired10_failure_return_${PIPELINE_TAG}"
WORK_DIR="${RETURN_ROOT}/${RUNNING_NAME}"
SUCCESS_DEST="${RETURN_ROOT}/${SUCCESS_NAME}"
FAILURE_DEST="${RETURN_ROOT}/${FAILURE_NAME}"
CONTROL_NAME="compare_vlmap_stage21c_path_reobserve_control_paired10_${PIPELINE_TAG}"
ACTIVE_NAME="compare_vlmap_stage21c_path_reobserve_active_paired10_${PIPELINE_TAG}"
CONTROL_ROOT="logs/habitat/${CONTROL_NAME}"
ACTIVE_ROOT="logs/habitat/${ACTIVE_NAME}"
FAILED_STAGE=initialization
PIPELINE_COMPLETE=0

write_metadata() {
  local root=$1
  git rev-parse HEAD > "${root}/git_commit.txt"
  git status --short > "${root}/git_status_short.txt"
  printf '%s\n' "${CHECKPOINT}" > "${root}/SOURCE_CHECKPOINT.txt"
  printf '%s\n' "${MANIFEST}" > "${root}/SOURCE_MANIFEST.txt"
  printf '%s\n' "${REFERENCE_ROOT}" > "${root}/SOURCE_REFERENCE_ROOT.txt"
  sha256sum "${MANIFEST}" > "${root}/MANIFEST_SHA256.txt"
  cat > "${root}/EXPERIMENT_SCOPE.txt" <<'EOF'
Paired10 Stage21c-r4 evaluation. The first five rows replay the known strict
episodes and the last five rows are new explicit-seed episodes. Control is
Frozen S2/NextDiT. Active allows only one strict bounded known-free-path
reorient/reobserve intervention per episode; it never uses blind forward,
Stage21d context, or relaxed occupied/unknown/path-corridor gates. The prior
500ep reference root is partial: missing new episodes are allowed by the
analyzer, while present known-episode references remain audited.
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
    package_run "${CONTROL_ROOT}" "${WORK_DIR}/control"
    package_run "${ACTIVE_ROOT}" "${WORK_DIR}/active"
    printf '%s\n' "${FAILED_STAGE}" > "${WORK_DIR}/FAILED_STAGE.txt"
    printf '%s\n' "${exit_status}" > "${WORK_DIR}/EXIT_STATUS.txt"
    write_metadata "${WORK_DIR}"
    mv "${WORK_DIR}" "${FAILURE_DEST}"
    find "${FAILURE_DEST}" -type f | sort > "${FAILURE_DEST}/RETURN_MANIFEST.txt"
    echo "STAGE21C_PATH_REOBSERVE_PAIRED10_STATUS=failed"
    echo "FAILED_STAGE=${FAILED_STAGE}"
    echo "DEST=$(readlink -f "${FAILURE_DEST}")"
  fi
}
trap package_failure EXIT

mkdir -p "${RETURN_ROOT}"
test -f "${CHECKPOINT}"
test -f "${MANIFEST}"
test -f "${REFERENCE_MANIFEST}"
test -f "${REFERENCE_ROOT}/progress.json"
test ! -e "${WORK_DIR}"; test ! -e "${SUCCESS_DEST}"; test ! -e "${FAILURE_DEST}"
test ! -e "${CONTROL_ROOT}"; test ! -e "${ACTIVE_ROOT}"
mkdir -p "${WORK_DIR}"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

FAILED_STAGE=control_evaluation
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
STAGE21_EPISODE_IDS="${MANIFEST}" STAGE21_RUN_NAME="${CONTROL_NAME}" \
STAGE21_EPISODE_SEED_REPLAY_MANIFEST="${MANIFEST}" \
STAGE21C_SCORER_CHECKPOINT="${CHECKPOINT}" STAGE21C_SCORER_DEVICE=cpu \
STAGE21_EVAL_PORT=${STAGE21C_PATH_CONTROL_EVAL_PORT:-2521} \
NPROC_PER_NODE=4 MASTER_PORT=${STAGE21C_PATH_CONTROL_MASTER_PORT:-2522} \
bash scripts/eval/bash/stage21_torchrun_eval.sh \
  --config scripts/eval/configs/habitat_dual_system_vlmap_stage21c_multitask_scorer_shadow_cfg.py

FAILED_STAGE=active_evaluation
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
STAGE21_EPISODE_IDS="${MANIFEST}" STAGE21_RUN_NAME="${ACTIVE_NAME}" \
STAGE21_EPISODE_SEED_REPLAY_MANIFEST="${MANIFEST}" \
STAGE21C_SCORER_CHECKPOINT="${CHECKPOINT}" STAGE21C_SCORER_DEVICE=cpu \
STAGE21C_PATH_ACTIVE_EVAL_PORT=${STAGE21C_PATH_ACTIVE_EVAL_PORT:-2523} \
NPROC_PER_NODE=4 MASTER_PORT=${STAGE21C_PATH_ACTIVE_MASTER_PORT:-2524} \
bash scripts/eval/bash/stage21_torchrun_eval.sh \
  --config scripts/eval/configs/habitat_dual_system_vlmap_stage21c_path_reobserve_active_cfg.py

FAILED_STAGE=paired_audit
python3 scripts/eval/analyze_stage21c_path_reobserve_paired.py \
  --control-root "${CONTROL_ROOT}" --active-root "${ACTIVE_ROOT}" \
  --expected-episodes "${EXPECTED_EPISODES}" \
  --seed-manifest "${MANIFEST}" --reference-root "${REFERENCE_ROOT}" \
  --reference-manifest "${REFERENCE_MANIFEST}" \
  --allow-reference-missing \
  --output "${ACTIVE_ROOT}/stage21c_path_reobserve_active_paired_audit.json" \
  --require-all

FAILED_STAGE=return_packaging
package_run "${CONTROL_ROOT}" "${WORK_DIR}/control"
package_run "${ACTIVE_ROOT}" "${WORK_DIR}/active_full"
cp -a "${ACTIVE_ROOT}/progress.json" "${ACTIVE_ROOT}/result.json" "${WORK_DIR}/"
cp -a "${ACTIVE_ROOT}/vlmap_safety_debug" "${WORK_DIR}/"
cp -a "${ACTIVE_ROOT}/stage21c_path_reobserve_active_paired_audit.json" "${WORK_DIR}/"
mkdir -p "${WORK_DIR}/episode_manifests"
cp -a "${MANIFEST}" "${WORK_DIR}/episode_manifests/"
printf '%s\n' "0" > "${WORK_DIR}/EXIT_STATUS.txt"
write_metadata "${WORK_DIR}"
mv "${WORK_DIR}" "${SUCCESS_DEST}"
find "${SUCCESS_DEST}" -type f | sort > "${SUCCESS_DEST}/RETURN_MANIFEST.txt"
PIPELINE_COMPLETE=1

echo "STAGE21C_PATH_REOBSERVE_PAIRED10_STATUS=complete"
echo "RETURN_NAME=${SUCCESS_NAME}"
echo "DEST=$(readlink -f "${SUCCESS_DEST}")"
du -sh "${SUCCESS_DEST}"
