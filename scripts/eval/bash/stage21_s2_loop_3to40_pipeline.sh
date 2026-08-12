#!/usr/bin/env bash
# Stage21 bounded shadow chain: deterministic 3ep loop replay -> audit -> balanced 40ep -> audit.
# It deliberately stops after 40ep for manual false-positive review and never starts 500ep/training.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

EPISODES_FILE=${STAGE21_TRAIN_EPISODES_FILE:-data/vln_ce/raw_data/r2r/train/train.json.gz}
FAILURE_MANIFEST=${STAGE21_FAILURE_MANIFEST:-data/stage21/stage21a_r3_failure_snapshot_3_episode_ids.json}
PIPELINE_TAG=${STAGE21_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
MANIFEST_DIR=${STAGE21_MANIFEST_DIR:-data/stage21}
DATASET_ROOT=${STAGE21_DATASET_ROOT:-data/stage21/datasets}
RETURN_ROOT=${STAGE21_RETURN_ROOT:-results/stage_17}
PIPELINE_LOG=${STAGE21_PIPELINE_LOG:-}

PIPELINE_COMPLETE=0
RUN_ROOT_3=""
RUN_ROOT_40=""
DATASET_DIR_40=""
CURRENT_RUN_ROOT=""

package_failure() {
  local exit_status=$?
  if [[ "${PIPELINE_COMPLETE}" == "1" || "${exit_status}" == "0" ]]; then return; fi
  local failure_dest="${RETURN_ROOT}/stage21a_s2_loop_3to40_failure_return_${PIPELINE_TAG}"
  mkdir -p "${failure_dest}/partial_runs" "${failure_dest}/episode_manifests"
  [[ -n "${RUN_ROOT_3}" && -d "${RUN_ROOT_3}" ]] && cp -a "${RUN_ROOT_3}" "${failure_dest}/partial_runs/3ep"
  [[ -n "${RUN_ROOT_40}" && -d "${RUN_ROOT_40}" ]] && cp -a "${RUN_ROOT_40}" "${failure_dest}/partial_runs/40ep"
  [[ -n "${CURRENT_RUN_ROOT}" && -d "${CURRENT_RUN_ROOT}" ]] && cp -a "${CURRENT_RUN_ROOT}" "${failure_dest}/partial_runs/current"
  [[ -n "${DATASET_DIR_40}" && -d "${DATASET_DIR_40}" ]] && cp -a "${DATASET_DIR_40}" "${failure_dest}/dataset40"
  [[ -f "${FAILURE_MANIFEST}" ]] && cp -a "${FAILURE_MANIFEST}" "${failure_dest}/episode_manifests/"
  [[ -f "${MANIFEST_DIR}/train_balanced_40_episode_ids.json" ]] && cp -a "${MANIFEST_DIR}"/train_balanced_40_episode_ids*.json "${failure_dest}/episode_manifests/"
  [[ -n "${PIPELINE_LOG}" && -f "${PIPELINE_LOG}" ]] && cp -a "${PIPELINE_LOG}" "${failure_dest}/pipeline.log"
  git rev-parse HEAD > "${failure_dest}/git_commit.txt"
  git status --short > "${failure_dest}/git_status_short.txt"
  printf '%s\n' "${exit_status}" > "${failure_dest}/EXIT_STATUS.txt"
  find "${failure_dest}" -type f | sort > "${failure_dest}/RETURN_MANIFEST.txt"
  echo "STAGE21_S2_LOOP_PIPELINE_STATUS=failed"
  echo "FAILURE_DEST=$(readlink -f "${failure_dest}")"
}
trap package_failure EXIT

test -f "${EPISODES_FILE}"
test -f "${FAILURE_MANIFEST}"
mkdir -p "${MANIFEST_DIR}" "${DATASET_ROOT}" "${RETURN_ROOT}"

python3 scripts/eval/select_balanced_r2r_episodes.py \
  --episodes-file "${EPISODES_FILE}" \
  --output "${MANIFEST_DIR}/train_balanced_40_episode_ids.json" \
  --summary-output "${MANIFEST_DIR}/train_balanced_40_episode_ids_summary.json" \
  --max-episodes 40 --seed 21 --shuffle-within-scene

RUN_NAME_3="compare_vlmap_stage21a_s2_loop_failure3_${PIPELINE_TAG}"
RUN_ROOT_3="logs/habitat/${RUN_NAME_3}"
CURRENT_RUN_ROOT="${RUN_ROOT_3}"
test ! -e "${RUN_ROOT_3}"
CUDA_VISIBLE_DEVICES=${STAGE21_SMOKE_CUDA_VISIBLE_DEVICES:-0} \
STAGE21_EPISODE_IDS="${FAILURE_MANIFEST}" STAGE21_RUN_NAME="${RUN_NAME_3}" \
STAGE21_EVAL_PORT=${STAGE21_SMOKE_EVAL_PORT:-2451} NPROC_PER_NODE=1 \
MASTER_PORT=${STAGE21_SMOKE_MASTER_PORT:-2452} \
bash scripts/eval/bash/stage21_torchrun_eval.sh \
  --config scripts/eval/configs/habitat_dual_system_vlmap_stage21a_r3_failure_snapshot_probe_cfg.py

python3 scripts/eval/analyze_stage21_s2_action_loops.py \
  --run-root "${RUN_ROOT_3}" --expected-episodes 3 \
  --output "${RUN_ROOT_3}/s2_action_loop_audit.json" \
  --forbid-episode 5q7pvUzZiYa/9357 \
  --require-episode SN83YJsR3w2/5982 --max-first-step SN83YJsR3w2/5982=55 \
  --require-episode V2XKFyX4ASd/775 --max-first-step V2XKFyX4ASd/775=70 \
  --min-candidate-coverage 1.0 \
  --require-all

RUN_NAME_40="compare_vlmap_stage21a_s2_loop_shadow40_${PIPELINE_TAG}"
RUN_ROOT_40="logs/habitat/${RUN_NAME_40}"
DATASET_DIR_40="${DATASET_ROOT}/${RUN_NAME_40}"
CURRENT_RUN_ROOT="${RUN_ROOT_40}"
test ! -e "${RUN_ROOT_40}"
test ! -e "${DATASET_DIR_40}"
CUDA_VISIBLE_DEVICES=${STAGE21_40_CUDA_VISIBLE_DEVICES:-0,1,2,3} \
STAGE21_EPISODE_IDS="${MANIFEST_DIR}/train_balanced_40_episode_ids.json" \
STAGE21_RUN_NAME="${RUN_NAME_40}" STAGE21_EVAL_PORT=${STAGE21_40_EVAL_PORT:-2453} \
NPROC_PER_NODE=4 MASTER_PORT=${STAGE21_40_MASTER_PORT:-2454} \
bash scripts/eval/bash/stage21_torchrun_eval.sh

python3 scripts/eval/analyze_stage21_s2_action_loops.py \
  --run-root "${RUN_ROOT_40}" --expected-episodes 40 \
  --output "${RUN_ROOT_40}/s2_action_loop_audit.json" --require-all

python3 scripts/eval/build_stage21_candidate_recoverability_dataset.py \
  --run-root "${RUN_ROOT_40}" --episodes-file "${EPISODES_FILE}" \
  --output-dir "${DATASET_DIR_40}" --reference-frame episodic_gps \
  --reference-coordinate-mode x_neg_y --gps-coordinate-mode x_neg_y \
  --quaternion-order xyzw --split-key scene --split-seed 21

python3 - "${RUN_ROOT_40}" "${DATASET_DIR_40}" <<'PY'
import json, sys
from pathlib import Path
run_root, dataset_dir = map(Path, sys.argv[1:])
progress = [json.loads(x) for x in (run_root / "progress.json").read_text().splitlines() if x.strip()]
summary = json.loads((dataset_dir / "summary.json").read_text())
loop_audit = json.loads((run_root / "s2_action_loop_audit.json").read_text())
assert len(progress) == 40, len(progress)
assert loop_audit["passed"] is True, loop_audit
assert loop_audit["gt_leakage_scan"]["passed"] is True, loop_audit
assert loop_audit["shadow_safety"]["passed"] is True, loop_audit
assert summary["reference_join_rate"] >= 0.98, summary["reference_join_rate"]
assert summary["active_safety_check"]["passed"] is True, summary["active_safety_check"]
assert summary["candidate_recoverability_rows"]["gt_leakage_scan"]["passed"] is True
assert summary["split_audit"]["scene_overlap_count"] == 0
assert summary["task_rows"]["recovery_proxy"] > 0, summary["task_rows"]
print(json.dumps({"episodes": 40, "loop_audit": loop_audit, "task_rows": summary["task_rows"]}, indent=2))
PY

RETURN_NAME="stage21a_s2_loop_3to40_return_${PIPELINE_TAG}"
DEST="${RETURN_ROOT}/${RETURN_NAME}"
test ! -e "${DEST}"
mkdir -p "${DEST}/smoke_runs" "${DEST}/datasets" "${DEST}/episode_manifests"
cp -a "${RUN_ROOT_40}/progress.json" "${RUN_ROOT_40}/result.json" \
  "${RUN_ROOT_40}/s2_action_loop_audit.json" "${DEST}/"
cp -a "${RUN_ROOT_40}/vlmap_safety_debug" "${DEST}/"
cp -a "${RUN_ROOT_3}" "${DEST}/smoke_runs/failure3"
cp -a "${DATASET_DIR_40}" "${DEST}/datasets/40ep"
cp -a "${FAILURE_MANIFEST}" "${DEST}/episode_manifests/"
cp -a "${MANIFEST_DIR}"/train_balanced_40_episode_ids*.json "${DEST}/episode_manifests/"
[[ -n "${PIPELINE_LOG}" && -f "${PIPELINE_LOG}" ]] && cp -a "${PIPELINE_LOG}" "${DEST}/pipeline.log"
git rev-parse HEAD > "${DEST}/git_commit.txt"
git status --short > "${DEST}/git_status_short.txt"
printf '%s\n' "${RUN_NAME_40}" > "${DEST}/RUN_NAME.txt"
printf '%s\n' "40" > "${DEST}/MAX_EPISODES.txt"
find "${DEST}" -type f | sort > "${DEST}/RETURN_MANIFEST.txt"
PIPELINE_COMPLETE=1
echo "STAGE21_S2_LOOP_PIPELINE_STATUS=complete"
echo "PIPELINE_STOPPED_AFTER=40ep_manual_review_gate"
echo "RETURN_NAME=${RETURN_NAME}"
echo "DEST=$(readlink -f "${DEST}")"
du -sh "${DEST}"
