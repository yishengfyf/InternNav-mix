#!/usr/bin/env bash
# Stage21 gated overnight chain: reuse validated 3ep -> 40ep audit -> 500ep audit.
# The frozen policy remains shadow-only; this script never starts adapter training.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

EPISODES_FILE=${STAGE21_TRAIN_EPISODES_FILE:-data/vln_ce/raw_data/r2r/train/train.json.gz}
FAILURE_MANIFEST=${STAGE21_FAILURE_MANIFEST:-data/stage21/stage21a_r3_failure_snapshot_3_episode_ids.json}
REUSE_3EP_RUN_ROOT=${STAGE21_REUSE_3EP_RUN_ROOT:?Set STAGE21_REUSE_3EP_RUN_ROOT to the validated 3ep run}
PIPELINE_TAG=${STAGE21_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
MANIFEST_DIR=${STAGE21_MANIFEST_DIR:-data/stage21}
DATASET_ROOT=${STAGE21_DATASET_ROOT:-data/stage21/datasets}
RETURN_ROOT=${STAGE21_RETURN_ROOT:-results/stage_17}
PIPELINE_LOG=${STAGE21_PIPELINE_LOG:-}

MIN_LOOP_EVENTS_40=${STAGE21_MIN_LOOP_EVENTS_40:-1}
MIN_CANDIDATE_COVERAGE=${STAGE21_MIN_CANDIDATE_COVERAGE:-0.95}
MAX_SUCCESS_TRIGGER_RATE=${STAGE21_MAX_SUCCESS_TRIGGER_RATE:-0.20}
MAX_STRICT_TIER_RATE=${STAGE21_MAX_STRICT_TIER_RATE:-0.75}
STRICT_RATE_MIN_EVENTS=${STAGE21_STRICT_RATE_MIN_EVENTS:-4}
MIN_RECOVERY_FEATURE_COVERAGE=${STAGE21_MIN_RECOVERY_FEATURE_COVERAGE:-0.95}

PIPELINE_COMPLETE=0
FAILED_STAGE=initialization
CURRENT_RUN_ROOT=""
CURRENT_DATASET_DIR=""
RUN_ROOT_3="${REUSE_3EP_RUN_ROOT}"
RUN_ROOT_40=""
RUN_ROOT_500=""
DATASET_DIR_40=""
DATASET_DIR_500=""

package_failure() {
  local exit_status=$?
  if [[ "${PIPELINE_COMPLETE}" == "1" || "${exit_status}" == "0" ]]; then return; fi
  local failure_dest="${RETURN_ROOT}/stage21a_s2_loop_40to500_failure_return_${PIPELINE_TAG}"
  mkdir -p "${failure_dest}/partial_runs" "${failure_dest}/partial_datasets" \
    "${failure_dest}/episode_manifests"
  [[ -d "${RUN_ROOT_3}" ]] && cp -a "${RUN_ROOT_3}" "${failure_dest}/partial_runs/3ep"
  [[ -n "${RUN_ROOT_40}" && -d "${RUN_ROOT_40}" ]] && cp -a "${RUN_ROOT_40}" "${failure_dest}/partial_runs/40ep"
  [[ -n "${RUN_ROOT_500}" && -d "${RUN_ROOT_500}" ]] && cp -a "${RUN_ROOT_500}" "${failure_dest}/partial_runs/500ep"
  [[ -n "${CURRENT_RUN_ROOT}" && "${CURRENT_RUN_ROOT}" != "${RUN_ROOT_40}" && \
    "${CURRENT_RUN_ROOT}" != "${RUN_ROOT_500}" && -d "${CURRENT_RUN_ROOT}" ]] && \
    cp -a "${CURRENT_RUN_ROOT}" "${failure_dest}/partial_runs/current"
  [[ -n "${DATASET_DIR_40}" && -d "${DATASET_DIR_40}" ]] && \
    cp -a "${DATASET_DIR_40}" "${failure_dest}/partial_datasets/40ep"
  [[ -n "${DATASET_DIR_500}" && -d "${DATASET_DIR_500}" ]] && \
    cp -a "${DATASET_DIR_500}" "${failure_dest}/partial_datasets/500ep"
  [[ -n "${CURRENT_DATASET_DIR}" && "${CURRENT_DATASET_DIR}" != "${DATASET_DIR_40}" && \
    "${CURRENT_DATASET_DIR}" != "${DATASET_DIR_500}" && -d "${CURRENT_DATASET_DIR}" ]] && \
    cp -a "${CURRENT_DATASET_DIR}" "${failure_dest}/partial_datasets/current"
  find "${MANIFEST_DIR}" -maxdepth 1 -type f \
    \( -name 'train_balanced_40_episode_ids*.json' -o -name 'train_balanced_500_episode_ids*.json' \) \
    -exec cp -a {} "${failure_dest}/episode_manifests/" \;
  [[ -f "${FAILURE_MANIFEST}" ]] && cp -a "${FAILURE_MANIFEST}" "${failure_dest}/episode_manifests/"
  [[ -n "${PIPELINE_LOG}" && -f "${PIPELINE_LOG}" ]] && cp -a "${PIPELINE_LOG}" "${failure_dest}/pipeline.log"
  printf '%s\n' "${FAILED_STAGE}" > "${failure_dest}/FAILED_STAGE.txt"
  printf '%s\n' "${exit_status}" > "${failure_dest}/EXIT_STATUS.txt"
  git rev-parse HEAD > "${failure_dest}/git_commit.txt"
  git status --short > "${failure_dest}/git_status_short.txt"
  find "${failure_dest}" -type f | sort > "${failure_dest}/RETURN_MANIFEST.txt"
  echo "STAGE21_OVERNIGHT_STATUS=failed"
  echo "FAILED_STAGE=${FAILED_STAGE}"
  echo "FAILURE_DEST=$(readlink -f "${failure_dest}")"
}
trap package_failure EXIT

test -f "${EPISODES_FILE}"
test -f "${FAILURE_MANIFEST}"
test -d "${RUN_ROOT_3}"
test -f "${RUN_ROOT_3}/progress.json"
mkdir -p "${MANIFEST_DIR}" "${DATASET_ROOT}" "${RETURN_ROOT}"

for episode_count in 40 500; do
  python3 scripts/eval/select_balanced_r2r_episodes.py \
    --episodes-file "${EPISODES_FILE}" \
    --output "${MANIFEST_DIR}/train_balanced_${episode_count}_episode_ids.json" \
    --summary-output "${MANIFEST_DIR}/train_balanced_${episode_count}_episode_ids_summary.json" \
    --max-episodes "${episode_count}" --seed 21 --shuffle-within-scene
done

FAILED_STAGE=3ep_regression_audit
python3 scripts/eval/analyze_stage21_s2_action_loops.py \
  --run-root "${RUN_ROOT_3}" --expected-episodes 3 \
  --output "${RUN_ROOT_3}/s2_action_loop_audit.json" \
  --forbid-episode 5q7pvUzZiYa/9357 \
  --require-episode SN83YJsR3w2/5982 --max-first-step SN83YJsR3w2/5982=55 \
  --require-episode V2XKFyX4ASd/775 --max-first-step V2XKFyX4ASd/775=70 \
  --min-candidate-coverage 1.0 --min-loop-events 2 --require-all

run_shadow_and_audit() {
  local label=$1
  local episode_count=$2
  local eval_port=$3
  local master_port=$4
  local minimum_loop_events=$5
  local run_name="compare_vlmap_stage21a_s2_loop_shadow_${label}_${PIPELINE_TAG}"
  local manifest="${MANIFEST_DIR}/train_balanced_${episode_count}_episode_ids.json"
  local run_root="logs/habitat/${run_name}"
  local dataset_dir="${DATASET_ROOT}/${run_name}"

  CURRENT_RUN_ROOT="${run_root}"
  CURRENT_DATASET_DIR="${dataset_dir}"
  test ! -e "${run_root}"
  test ! -e "${dataset_dir}"

  FAILED_STAGE="${label}_evaluation"
  CUDA_VISIBLE_DEVICES=${STAGE21_CUDA_VISIBLE_DEVICES:-0,1,2,3} \
  STAGE21_EPISODE_IDS="${manifest}" STAGE21_RUN_NAME="${run_name}" \
  STAGE21_EVAL_PORT="${eval_port}" NPROC_PER_NODE=4 MASTER_PORT="${master_port}" \
  bash scripts/eval/bash/stage21_torchrun_eval.sh

  FAILED_STAGE="${label}_loop_audit"
  python3 scripts/eval/analyze_stage21_s2_action_loops.py \
    --run-root "${run_root}" --expected-episodes "${episode_count}" \
    --output "${run_root}/s2_action_loop_audit.json" \
    --min-loop-events "${minimum_loop_events}" \
    --min-candidate-coverage "${MIN_CANDIDATE_COVERAGE}" \
    --max-success-trigger-rate "${MAX_SUCCESS_TRIGGER_RATE}" \
    --max-strict-tier-rate "${MAX_STRICT_TIER_RATE}" \
    --strict-rate-min-events "${STRICT_RATE_MIN_EVENTS}" --require-all

  FAILED_STAGE="${label}_dataset_build"
  python3 scripts/eval/build_stage21_candidate_recoverability_dataset.py \
    --run-root "${run_root}" --episodes-file "${EPISODES_FILE}" \
    --output-dir "${dataset_dir}" --reference-frame episodic_gps \
    --reference-coordinate-mode x_neg_y --gps-coordinate-mode x_neg_y \
    --quaternion-order xyzw --split-key scene --split-seed 21

  FAILED_STAGE="${label}_dataset_audit"
  python3 - "${run_root}" "${dataset_dir}" "${episode_count}" \
    "${MIN_RECOVERY_FEATURE_COVERAGE}" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
dataset_dir = Path(sys.argv[2])
expected_episodes = int(sys.argv[3])
minimum_feature_coverage = float(sys.argv[4])
progress = [
    json.loads(line)
    for line in (run_root / "progress.json").read_text().splitlines()
    if line.strip()
]
loop_audit = json.loads((run_root / "s2_action_loop_audit.json").read_text())
summary = json.loads((dataset_dir / "summary.json").read_text())
candidate_rows = summary["candidate_recoverability_rows"]

assert len(progress) == expected_episodes, (len(progress), expected_episodes)
assert (run_root / "result.json").is_file(), run_root / "result.json"
assert loop_audit["passed"] is True, loop_audit
assert summary["run_dir_count"] == 4, summary["run_dir_count"]
assert summary["event_schema_version"] == "stage21a_r3_v3", summary
assert summary["counts"]["duplicate_rows_removed"] == 0, summary["counts"]
assert summary["counts"]["label_rows"] > 0, summary["counts"]
assert summary["reference_join_rate"] >= 0.98, summary["reference_join_rate"]
assert summary["active_safety_check"]["passed"] is True, summary["active_safety_check"]
assert summary["split_audit"]["scene_overlap_count"] == 0, summary["split_audit"]
assert candidate_rows["status"] == "ok", candidate_rows
assert candidate_rows["gt_leakage_scan"]["passed"] is True, candidate_rows
assert candidate_rows["active_gate_safe_used_as_recovery_target"] is False, candidate_rows
assert candidate_rows["cycle_feature_audit"]["passed"] is True, candidate_rows
assert summary["task_rows"]["progress"] > 0, summary["task_rows"]
assert summary["task_rows"]["recovery_proxy"] > 0, summary["task_rows"]
assert summary["task_rows"]["safety"] > 0, summary["task_rows"]
coverage = candidate_rows["recovery_feature_coverage"]["rates"]
for field in (
    "recovery_feature_schema_version",
    "anchor_visible_free_ratio",
    "anchor_branch_count",
    "anchor_executable_exit_count",
    "anchor_connected_component_count",
    "anchor_branch_depth_mean",
    "anchor_short_cycle_risk",
    "current_to_anchor_free_ratio_gain",
    "current_to_anchor_branch_gain",
):
    assert coverage[field] >= minimum_feature_coverage, (field, coverage[field])
print(json.dumps({
    "episodes": len(progress),
    "loop_audit": loop_audit,
    "task_rows": summary["task_rows"],
    "recovery_feature_coverage": coverage,
}, indent=2))
PY

  LAST_RUN_NAME="${run_name}"
  LAST_RUN_ROOT="${run_root}"
  LAST_DATASET_DIR="${dataset_dir}"
}

run_shadow_and_audit 40ep 40 ${STAGE21_40_EVAL_PORT:-2461} \
  ${STAGE21_40_MASTER_PORT:-2462} "${MIN_LOOP_EVENTS_40}"
RUN_ROOT_40="${LAST_RUN_ROOT}"
DATASET_DIR_40="${LAST_DATASET_DIR}"

run_shadow_and_audit 500ep 500 ${STAGE21_500_EVAL_PORT:-2463} \
  ${STAGE21_500_MASTER_PORT:-2464} 1
RUN_NAME_500="${LAST_RUN_NAME}"
RUN_ROOT_500="${LAST_RUN_ROOT}"
DATASET_DIR_500="${LAST_DATASET_DIR}"

FAILED_STAGE=return_packaging
RETURN_NAME="stage21a_s2_loop_shadow500_return_${PIPELINE_TAG}"
DEST="${RETURN_ROOT}/${RETURN_NAME}"
test ! -e "${DEST}"
mkdir -p "${DEST}/smoke_runs" "${DEST}/datasets" "${DEST}/episode_manifests"
cp -a "${RUN_ROOT_500}/progress.json" "${RUN_ROOT_500}/result.json" \
  "${RUN_ROOT_500}/s2_action_loop_audit.json" "${DEST}/"
cp -a "${RUN_ROOT_500}/vlmap_safety_debug" "${DEST}/"
cp -a "${RUN_ROOT_3}" "${DEST}/smoke_runs/3ep"
cp -a "${RUN_ROOT_40}" "${DEST}/smoke_runs/40ep"
cp -a "${DATASET_DIR_40}" "${DEST}/datasets/40ep"
cp -a "${DATASET_DIR_500}" "${DEST}/datasets/500ep"
cp -a "${FAILURE_MANIFEST}" "${DEST}/episode_manifests/"
cp -a "${MANIFEST_DIR}"/train_balanced_{40,500}_episode_ids*.json "${DEST}/episode_manifests/"
[[ -n "${PIPELINE_LOG}" && -f "${PIPELINE_LOG}" ]] && cp -a "${PIPELINE_LOG}" "${DEST}/pipeline.log"
git rev-parse HEAD > "${DEST}/git_commit.txt"
git status --short > "${DEST}/git_status_short.txt"
printf '%s\n' "${RUN_NAME_500}" > "${DEST}/RUN_NAME.txt"
printf '%s\n' "500ep_shadow_audit; training_not_started" > "${DEST}/PIPELINE_STOPPED_AFTER.txt"
find "${DEST}" -type f | sort > "${DEST}/RETURN_MANIFEST.txt"

PIPELINE_COMPLETE=1
echo "STAGE21_OVERNIGHT_STATUS=complete"
echo "PIPELINE_STOPPED_AFTER=500ep_shadow_audit"
echo "TRAINING_STARTED=0"
echo "RETURN_NAME=${RETURN_NAME}"
echo "DEST=$(readlink -f "${DEST}")"
du -sh "${DEST}"
