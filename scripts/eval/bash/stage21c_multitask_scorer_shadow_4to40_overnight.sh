#!/usr/bin/env bash
# Stage21c: 4ep implementation smoke -> 40ep frozen scorer shadow.
# Four GPUs shard Habitat episodes; the scorer is inference-only in each worker.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

EPISODES_FILE=${STAGE21C_EPISODES_FILE:-data/vln_ce/raw_data/r2r/train/train.json.gz}
CHECKPOINT=${STAGE21C_SCORER_CHECKPOINT:?Set STAGE21C_SCORER_CHECKPOINT to seed_53/best.pt}
RETURN_ROOT=${STAGE21C_RETURN_ROOT:-results/stage_17}
MANIFEST_DIR=${STAGE21C_MANIFEST_DIR:-data/stage21}
PIPELINE_TAG=${STAGE21C_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
CUDA_VISIBLE_DEVICES_VALUE=${STAGE21C_CUDA_VISIBLE_DEVICES:-0,1,2,3}
MIN_VALID_RATE=${STAGE21C_MIN_VALID_RATE:-0.95}
MAX_P95_LATENCY_MS=${STAGE21C_MAX_P95_LATENCY_MS:-250}

RUNNING_NAME="stage21c_multitask_scorer_shadow_4to40_running_${PIPELINE_TAG}"
SUCCESS_NAME="stage21c_multitask_scorer_shadow_4to40_return_${PIPELINE_TAG}"
FAILURE_NAME="stage21c_multitask_scorer_shadow_4to40_failure_return_${PIPELINE_TAG}"
WORK_DIR="${RETURN_ROOT}/${RUNNING_NAME}"
SUCCESS_DEST="${RETURN_ROOT}/${SUCCESS_NAME}"
FAILURE_DEST="${RETURN_ROOT}/${FAILURE_NAME}"
DATASET_DIR="${WORK_DIR}/episode_manifests"
FAILED_STAGE=initialization
PIPELINE_COMPLETE=0
RUN_ROOT_4=""
RUN_ROOT_40=""

mkdir -p "${RETURN_ROOT}" "${MANIFEST_DIR}"
test -f "${EPISODES_FILE}"; test -f "${CHECKPOINT}"
test ! -e "${WORK_DIR}"; test ! -e "${SUCCESS_DEST}"; test ! -e "${FAILURE_DEST}"
mkdir -p "${WORK_DIR}/smoke_runs/4ep" "${WORK_DIR}/runs/40ep" "${WORK_DIR}/audits" "${DATASET_DIR}"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

write_metadata() {
  local root=$1
  printf '%s\n' "${CHECKPOINT}" > "${root}/SOURCE_CHECKPOINT.txt"
  printf '%s\n' "${EPISODES_FILE}" > "${root}/SOURCE_EPISODES_FILE.txt"
  cat > "${root}/SHADOW_SCOPE.txt" <<'EOF'
frozen scorer shadow only
Frozen S2/NextDiT: true
Habitat episode action source: original S2 only
Episode-time parameter update: false
Active recovery: false
Four GPUs are episode workers; scorer has no gradient/DDP synchronization
EOF
  git rev-parse HEAD > "${root}/git_commit.txt"
  git status --short > "${root}/git_status_short.txt"
}

package_failure() {
  local exit_status=$?
  if [[ "${PIPELINE_COMPLETE}" == "1" || "${exit_status}" == "0" ]]; then return; fi
  if [[ -d "${WORK_DIR}" ]]; then
    printf '%s\n' "${FAILED_STAGE}" > "${WORK_DIR}/FAILED_STAGE.txt"
    printf '%s\n' "${exit_status}" > "${WORK_DIR}/EXIT_STATUS.txt"
    write_metadata "${WORK_DIR}"
    find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
    mv "${WORK_DIR}" "${FAILURE_DEST}"
    echo "STAGE21C_SHADOW_STATUS=failed"
    echo "FAILED_STAGE=${FAILED_STAGE}"
    echo "FAILURE_DEST=$(readlink -f "${FAILURE_DEST}")"
  fi
}
trap package_failure EXIT

for count in 4 40; do
  FAILED_STAGE="manifest_${count}ep"
  python3 scripts/eval/select_balanced_r2r_episodes.py \
    --episodes-file "${EPISODES_FILE}" \
    --output "${DATASET_DIR}/train_balanced_${count}_episode_ids.json" \
    --summary-output "${DATASET_DIR}/train_balanced_${count}_episode_ids_summary.json" \
    --max-episodes "${count}" --seed 21 --shuffle-within-scene
done

run_shadow() {
  local label=$1
  local count=$2
  local eval_port=$3
  local master_port=$4
  local run_name="compare_vlmap_stage21c_multitask_scorer_shadow_${label}_${PIPELINE_TAG}"
  local manifest="${DATASET_DIR}/train_balanced_${count}_episode_ids.json"
  local run_root="logs/habitat/${run_name}"
  local config="scripts/eval/configs/habitat_dual_system_vlmap_stage21c_multitask_scorer_shadow_cfg.py"
  test ! -e "${run_root}"
  FAILED_STAGE="${label}_evaluation"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
  STAGE21_EPISODE_IDS="${manifest}" STAGE21_RUN_NAME="${run_name}" \
  STAGE21C_SCORER_CHECKPOINT="${CHECKPOINT}" STAGE21C_SCORER_DEVICE="cpu" \
  STAGE21_EVAL_PORT="${eval_port}" NPROC_PER_NODE=4 MASTER_PORT="${master_port}" \
  bash scripts/eval/bash/stage21_torchrun_eval.sh --config "${config}"
  FAILED_STAGE="${label}_audit"
  python3 scripts/eval/analyze_stage21c_multitask_scorer_shadow.py \
    --run-root "${run_root}" --expected-episodes "${count}" \
    --min-valid-rate "${MIN_VALID_RATE}" --max-p95-latency-ms "${MAX_P95_LATENCY_MS}" \
    --output "${run_root}/stage21c_multitask_scorer_shadow_audit.json" --require-all
  if [[ "${count}" == "4" ]]; then RUN_ROOT_4="${run_root}"; else RUN_ROOT_40="${run_root}"; fi
}

run_shadow 4ep 4 ${STAGE21C_4EP_EVAL_PORT:-2471} ${STAGE21C_4EP_MASTER_PORT:-2472}
run_shadow 40ep 40 ${STAGE21C_40EP_EVAL_PORT:-2473} ${STAGE21C_40EP_MASTER_PORT:-2474}

FAILED_STAGE=return_packaging
mkdir -p "${WORK_DIR}/smoke_runs/4ep" "${WORK_DIR}/runs/40ep" "${WORK_DIR}/checkpoint"
cp -a "${RUN_ROOT_4}/progress.json" "${RUN_ROOT_4}/result.json" \
  "${RUN_ROOT_4}/stage21c_multitask_scorer_shadow_audit.json" "${WORK_DIR}/smoke_runs/4ep/"
cp -a "${RUN_ROOT_4}/vlmap_safety_debug" "${WORK_DIR}/smoke_runs/4ep/"
cp -a "${RUN_ROOT_40}/progress.json" "${RUN_ROOT_40}/result.json" \
  "${RUN_ROOT_40}/stage21c_multitask_scorer_shadow_audit.json" "${WORK_DIR}/runs/40ep/"
cp -a "${RUN_ROOT_40}/vlmap_safety_debug" "${WORK_DIR}/runs/40ep/"
cp -a "${CHECKPOINT}" "${CHECKPOINT%/*}/feature_schema.json" "${CHECKPOINT%/*}/normalizer.json" \
  "${CHECKPOINT%/*}/training_config.json" "${WORK_DIR}/checkpoint/"
printf '%s\n' "4ep and 40ep frozen scorer shadow passed; no action was changed and no active recovery ran" \
  > "${WORK_DIR}/SHADOW_STOPPED_AFTER.txt"
printf '%s\n' "0" > "${WORK_DIR}/EXIT_STATUS.txt"
write_metadata "${WORK_DIR}"
find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
mv "${WORK_DIR}" "${SUCCESS_DEST}"
PIPELINE_COMPLETE=1

echo "STAGE21C_SHADOW_STATUS=complete"
echo "SHADOW_STOPPED_AFTER=frozen_40ep_shadow"
echo "RETURN_NAME=${SUCCESS_NAME}"
echo "DEST=$(readlink -f "${SUCCESS_DEST}")"
du -sh "${SUCCESS_DEST}"
