#!/usr/bin/env bash
# Stage21c: 500ep frozen multi-task scorer shadow with ranking, triage and snapshot audit.
# Four GPUs shard Habitat episodes. The scorer is inference-only and never changes an action.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

EPISODES_FILE=${STAGE21C_EPISODES_FILE:-data/vln_ce/raw_data/r2r/train/train.json.gz}
CHECKPOINT=${STAGE21C_SCORER_CHECKPOINT:?Set STAGE21C_SCORER_CHECKPOINT to seed_53/best.pt}
RETURN_ROOT=${STAGE21C_RETURN_ROOT:-results/stage_17}
PIPELINE_TAG=${STAGE21C_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
CUDA_VISIBLE_DEVICES_VALUE=${STAGE21C_CUDA_VISIBLE_DEVICES:-0,1,2,3}
MIN_VALID_RATE=${STAGE21C_MIN_VALID_RATE:-0.95}
MAX_P95_LATENCY_MS=${STAGE21C_MAX_P95_LATENCY_MS:-250}
MAX_EVENTS_PER_EPISODE=${STAGE21C_MAX_EVENTS_PER_EPISODE:-64}
LOOP_SNAPSHOTS_PER_EPISODE=${STAGE21C_LOOP_SNAPSHOTS_PER_EPISODE:-3}
MIN_LOOP_EVENTS=${STAGE21C_MIN_LOOP_EVENTS:-1}
MAX_SUCCESS_TRIGGER_RATE=${STAGE21C_MAX_SUCCESS_TRIGGER_RATE:-0.20}
MAX_STRICT_TIER_RATE=${STAGE21C_MAX_STRICT_TIER_RATE:-0.75}
STRICT_RATE_MIN_EVENTS=${STAGE21C_STRICT_RATE_MIN_EVENTS:-4}

RUNNING_NAME="stage21c_multitask_scorer_shadow500_running_${PIPELINE_TAG}"
SUCCESS_NAME="stage21c_multitask_scorer_shadow500_return_${PIPELINE_TAG}"
FAILURE_NAME="stage21c_multitask_scorer_shadow500_failure_return_${PIPELINE_TAG}"
WORK_DIR="${RETURN_ROOT}/${RUNNING_NAME}"
SUCCESS_DEST="${RETURN_ROOT}/${SUCCESS_NAME}"
FAILURE_DEST="${RETURN_ROOT}/${FAILURE_NAME}"
DATASET_DIR="${WORK_DIR}/episode_manifests"
RUN_NAME="compare_vlmap_stage21c_multitask_scorer_shadow_500ep_${PIPELINE_TAG}"
RUN_ROOT="logs/habitat/${RUN_NAME}"
MANIFEST="${DATASET_DIR}/train_balanced_500_episode_ids.json"
AUDIT_PATH="${RUN_ROOT}/stage21c_multitask_scorer_shadow_audit.json"
SUMMARY_PATH="${RUN_ROOT}/STAGE21C_KEY_METRICS.md"
FAILED_STAGE=initialization
PIPELINE_COMPLETE=0

write_metadata() {
  local root=$1
  printf '%s\n' "${CHECKPOINT}" > "${root}/SOURCE_CHECKPOINT.txt"
  printf '%s\n' "${EPISODES_FILE}" > "${root}/SOURCE_EPISODES_FILE.txt"
  printf '%s\n' "${RUN_NAME}" > "${root}/RUN_NAME.txt"
  cat > "${root}/SHADOW_SCOPE.txt" <<'EOF'
500ep frozen scorer shadow only
Frozen S2/NextDiT: true
Habitat episode action source: original S2 only
Episode-time parameter update: false
Active recovery: false
Four GPUs are episode workers; scorer has no gradient/DDP synchronization
S2 loop and stuck RGB snapshots: enabled
EOF
  git rev-parse HEAD > "${root}/git_commit.txt"
  git status --short > "${root}/git_status_short.txt"
}

write_manifest() {
  local root=$1
  find "${root}" -type f | sed "s#^${root}/##" | sort > "${root}/RETURN_MANIFEST.txt"
}

package_failure() {
  local exit_status=$?
  if [[ "${PIPELINE_COMPLETE}" == "1" || "${exit_status}" == "0" ]]; then
    return
  fi
  if [[ -d "${WORK_DIR}" ]]; then
    mkdir -p "${WORK_DIR}/partial_run"
    if [[ -d "${RUN_ROOT}" ]]; then
      cp -a "${RUN_ROOT}/." "${WORK_DIR}/partial_run/"
    fi
    printf '%s\n' "${FAILED_STAGE}" > "${WORK_DIR}/FAILED_STAGE.txt"
    printf '%s\n' "${exit_status}" > "${WORK_DIR}/EXIT_STATUS.txt"
    write_metadata "${WORK_DIR}"
    mv "${WORK_DIR}" "${FAILURE_DEST}"
    write_manifest "${FAILURE_DEST}"
    echo "STAGE21C_SHADOW500_STATUS=failed"
    echo "FAILED_STAGE=${FAILED_STAGE}"
    echo "FAILURE_DEST=$(readlink -f "${FAILURE_DEST}")"
  fi
}
trap package_failure EXIT

mkdir -p "${RETURN_ROOT}"
test -f "${EPISODES_FILE}"
test -f "${CHECKPOINT}"
test ! -e "${WORK_DIR}"
test ! -e "${SUCCESS_DEST}"
test ! -e "${FAILURE_DEST}"
test ! -e "${RUN_ROOT}"
mkdir -p "${WORK_DIR}/checkpoint" "${DATASET_DIR}"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

FAILED_STAGE=manifest_500ep
python3 scripts/eval/select_balanced_r2r_episodes.py \
  --episodes-file "${EPISODES_FILE}" \
  --output "${MANIFEST}" \
  --summary-output "${DATASET_DIR}/train_balanced_500_episode_ids_summary.json" \
  --max-episodes 500 --seed 21 --shuffle-within-scene

FAILED_STAGE=500ep_evaluation
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
STAGE21_EPISODE_IDS="${MANIFEST}" \
STAGE21_RUN_NAME="${RUN_NAME}" \
STAGE21C_SCORER_CHECKPOINT="${CHECKPOINT}" \
STAGE21C_SCORER_DEVICE="cpu" \
STAGE21C_MAX_EVENTS_PER_EPISODE="${MAX_EVENTS_PER_EPISODE}" \
STAGE21C_LOOP_SNAPSHOTS_PER_EPISODE="${LOOP_SNAPSHOTS_PER_EPISODE}" \
STAGE21_EVAL_PORT="${STAGE21C_500EP_EVAL_PORT:-2475}" \
NPROC_PER_NODE=4 \
MASTER_PORT="${STAGE21C_500EP_MASTER_PORT:-2476}" \
bash scripts/eval/bash/stage21_torchrun_eval.sh \
  --config scripts/eval/configs/habitat_dual_system_vlmap_stage21c_multitask_scorer_shadow_cfg.py

FAILED_STAGE=500ep_audit
python3 scripts/eval/analyze_stage21c_multitask_scorer_shadow.py \
  --run-root "${RUN_ROOT}" \
  --expected-episodes 500 \
  --min-valid-rate "${MIN_VALID_RATE}" \
  --max-p95-latency-ms "${MAX_P95_LATENCY_MS}" \
  --min-loop-events "${MIN_LOOP_EVENTS}" \
  --max-success-trigger-rate "${MAX_SUCCESS_TRIGGER_RATE}" \
  --max-strict-tier-rate "${MAX_STRICT_TIER_RATE}" \
  --strict-rate-min-events "${STRICT_RATE_MIN_EVENTS}" \
  --output "${AUDIT_PATH}" \
  --summary-output "${SUMMARY_PATH}" \
  --require-all

FAILED_STAGE=return_packaging
cp -a "${RUN_ROOT}/progress.json" "${RUN_ROOT}/result.json" \
  "${AUDIT_PATH}" "${SUMMARY_PATH}" "${WORK_DIR}/"
cp -a "${RUN_ROOT}/vlmap_safety_debug" "${WORK_DIR}/"
cp -a "${CHECKPOINT}" "${CHECKPOINT%/*}/feature_schema.json" \
  "${CHECKPOINT%/*}/normalizer.json" "${CHECKPOINT%/*}/training_config.json" \
  "${WORK_DIR}/checkpoint/"
find "${WORK_DIR}/vlmap_safety_debug" -type f \
  \( -path '*/s2_action_loop_snapshots/*.jpg' -o -path '*/stuck_snapshots/*.jpg' \
     -o -path '*/stuck_snapshots/*.json' -o -name 's2_action_loop_events.jsonl' \) \
  | sed "s#^${WORK_DIR}/##" | sort > "${WORK_DIR}/VISUAL_REVIEW_FILES.txt"
printf '%s\n' "500ep frozen scorer shadow audit passed; no action was changed and no active recovery ran" \
  > "${WORK_DIR}/SHADOW_STOPPED_AFTER.txt"
printf '%s\n' "0" > "${WORK_DIR}/EXIT_STATUS.txt"
write_metadata "${WORK_DIR}"
mv "${WORK_DIR}" "${SUCCESS_DEST}"
write_manifest "${SUCCESS_DEST}"
PIPELINE_COMPLETE=1

echo "STAGE21C_SHADOW500_STATUS=complete"
echo "SHADOW_STOPPED_AFTER=frozen_500ep_shadow"
echo "ACTIVE_RECOVERY_RAN=0"
echo "RETURN_NAME=${SUCCESS_NAME}"
echo "DEST=$(readlink -f "${SUCCESS_DEST}")"
du -sh "${SUCCESS_DEST}"
