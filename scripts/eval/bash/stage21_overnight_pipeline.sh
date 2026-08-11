#!/bin/bash
# Gated Stage21a overnight pipeline:
# 4ep/1GPU smoke -> 40ep/4GPU smoke -> 500ep/4GPU Run A0 -> package.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

EPISODES_FILE=${STAGE21_TRAIN_EPISODES_FILE:-data/vln_ce/raw_data/r2r/train/train.json.gz}
PIPELINE_TAG=${STAGE21_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
MANIFEST_DIR=${STAGE21_MANIFEST_DIR:-data/stage21}
DATASET_ROOT=${STAGE21_DATASET_ROOT:-data/stage21/datasets}
RETURN_ROOT=${STAGE21_RETURN_ROOT:-results/stage_17}
PIPELINE_LOG=${STAGE21_PIPELINE_LOG:-}

PIPELINE_COMPLETE=0
CURRENT_RUN_ROOT=""
CURRENT_DATASET_DIR=""
RUN_ROOT_4=""
RUN_ROOT_40=""
DATASET_DIR_4=""
DATASET_DIR_40=""

package_failure() {
  local exit_status=$?
  if [[ "${PIPELINE_COMPLETE}" == "1" || "${exit_status}" == "0" ]]; then
    return
  fi
  local failure_name="stage21a_train_recovery_shadow_failure_return_${PIPELINE_TAG}"
  local failure_dest="${RETURN_ROOT}/${failure_name}"
  mkdir -p "${failure_dest}/partial_runs" "${failure_dest}/partial_datasets" \
    "${failure_dest}/episode_manifests"
  if [[ -n "${RUN_ROOT_4}" && -d "${RUN_ROOT_4}" ]]; then
    cp -a "${RUN_ROOT_4}" "${failure_dest}/partial_runs/4ep"
  fi
  if [[ -n "${RUN_ROOT_40}" && -d "${RUN_ROOT_40}" ]]; then
    cp -a "${RUN_ROOT_40}" "${failure_dest}/partial_runs/40ep"
  fi
  if [[ -n "${CURRENT_RUN_ROOT}" && -d "${CURRENT_RUN_ROOT}" ]]; then
    cp -a "${CURRENT_RUN_ROOT}" "${failure_dest}/partial_runs/current"
  fi
  if [[ -n "${DATASET_DIR_4}" && -d "${DATASET_DIR_4}" ]]; then
    cp -a "${DATASET_DIR_4}" "${failure_dest}/partial_datasets/4ep"
  fi
  if [[ -n "${DATASET_DIR_40}" && -d "${DATASET_DIR_40}" ]]; then
    cp -a "${DATASET_DIR_40}" "${failure_dest}/partial_datasets/40ep"
  fi
  if [[ -n "${CURRENT_DATASET_DIR}" && -d "${CURRENT_DATASET_DIR}" ]]; then
    cp -a "${CURRENT_DATASET_DIR}" "${failure_dest}/partial_datasets/current"
  fi
  if [[ -d "${MANIFEST_DIR}" ]]; then
    find "${MANIFEST_DIR}" -maxdepth 1 -type f -name 'train_balanced_*_episode_ids*.json' \
      -exec cp -a {} "${failure_dest}/episode_manifests/" \;
  fi
  if [[ -n "${PIPELINE_LOG}" && -f "${PIPELINE_LOG}" ]]; then
    cp -a "${PIPELINE_LOG}" "${failure_dest}/overnight_pipeline.log"
  fi
  git rev-parse HEAD > "${failure_dest}/git_commit.txt"
  git status --short > "${failure_dest}/git_status_short.txt"
  printf '%s\n' "${exit_status}" > "${failure_dest}/EXIT_STATUS.txt"
  find "${failure_dest}" -type f | sort > "${failure_dest}/RETURN_MANIFEST.txt"
  echo "STAGE21_PIPELINE_STATUS=failed"
  echo "FAILURE_DEST=$(readlink -f "${failure_dest}")"
}

trap package_failure EXIT

test -f "${EPISODES_FILE}"
mkdir -p "${MANIFEST_DIR}" "${DATASET_ROOT}" "${RETURN_ROOT}"

for episode_count in 4 40 500; do
  python scripts/eval/select_balanced_r2r_episodes.py \
    --episodes-file "${EPISODES_FILE}" \
    --output "${MANIFEST_DIR}/train_balanced_${episode_count}_episode_ids.json" \
    --summary-output "${MANIFEST_DIR}/train_balanced_${episode_count}_episode_ids_summary.json" \
    --max-episodes "${episode_count}" \
    --seed 21 \
    --shuffle-within-scene
done

run_and_audit() {
  local label=$1
  local episode_count=$2
  local nproc=$3
  local eval_port=$4
  local master_port=$5
  local expected_run_dirs=$6
  local run_name="compare_vlmap_stage21a_train_recovery_shadow_${label}_${PIPELINE_TAG}"
  local manifest="${MANIFEST_DIR}/train_balanced_${episode_count}_episode_ids.json"
  local run_root="logs/habitat/${run_name}"
  local dataset_dir="${DATASET_ROOT}/${run_name}"

  CURRENT_RUN_ROOT=${run_root}
  CURRENT_DATASET_DIR=${dataset_dir}

  test ! -e "${run_root}"
  test ! -e "${dataset_dir}"

  STAGE21_EPISODE_IDS="${manifest}" \
  STAGE21_RUN_NAME="${run_name}" \
  STAGE21_EVAL_PORT="${eval_port}" \
  NPROC_PER_NODE="${nproc}" \
  MASTER_PORT="${master_port}" \
  bash scripts/eval/bash/stage21_torchrun_eval.sh

  python scripts/eval/build_stage21_candidate_recoverability_dataset.py \
    --run-root "${run_root}" \
    --episodes-file "${EPISODES_FILE}" \
    --output-dir "${dataset_dir}" \
    --reference-frame episodic_gps \
    --reference-coordinate-mode x_neg_y \
    --gps-coordinate-mode x_neg_y \
    --quaternion-order xyzw \
    --split-key scene \
    --split-seed 21

  python - "${run_root}" "${dataset_dir}" "${episode_count}" "${expected_run_dirs}" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
dataset_dir = Path(sys.argv[2])
expected_episodes = int(sys.argv[3])
expected_run_dirs = int(sys.argv[4])

assert (run_root / "result.json").is_file(), run_root / "result.json"
progress_path = run_root / "progress.json"
assert progress_path.is_file(), progress_path
progress_rows = [json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
assert len(progress_rows) == expected_episodes, (len(progress_rows), expected_episodes)

summary = json.loads((dataset_dir / "summary.json").read_text())
assert summary["run_dir_count"] == expected_run_dirs, summary["run_dir_count"]
assert summary["counts"]["duplicate_rows_removed"] == 0, summary["counts"]
assert summary["counts"]["label_rows"] > 0, summary["counts"]
assert summary["reference_join_rate"] >= 0.98, summary["reference_join_rate"]
assert summary["active_safety_check"]["passed"] is True, summary["active_safety_check"]
assert summary["split_audit"]["scene_overlap_count"] == 0, summary["split_audit"]
candidate_rows = summary["candidate_recoverability_rows"]
assert candidate_rows["status"] == "ok", candidate_rows
assert candidate_rows["counts"]["candidate_rows"] > 0, candidate_rows
print(json.dumps({
    "run_root": str(run_root),
    "dataset_dir": str(dataset_dir),
    "episodes": len(progress_rows),
    "run_dir_count": summary["run_dir_count"],
    "reference_join_rate": summary["reference_join_rate"],
    "candidate_rows": candidate_rows["counts"]["candidate_rows"],
    "positive_vs_negative_pairs": candidate_rows["counts"]["positive_vs_negative_pairs"],
    "active_applied": summary["active_safety_check"]["applied_count"],
}, indent=2))
PY

  LAST_RUN_NAME=${run_name}
  LAST_RUN_ROOT=${run_root}
  LAST_DATASET_DIR=${dataset_dir}
}

run_and_audit 4ep_smoke 4 1 2421 2422 1
RUN_ROOT_4=${LAST_RUN_ROOT}
DATASET_DIR_4=${LAST_DATASET_DIR}

run_and_audit 40ep_smoke 40 4 2423 2424 4
RUN_ROOT_40=${LAST_RUN_ROOT}
DATASET_DIR_40=${LAST_DATASET_DIR}

# This expensive run is reached only when both smoke audits exit successfully.
run_and_audit a0_500ep 500 4 2425 2426 4
RUN_NAME_500=${LAST_RUN_NAME}
RUN_ROOT_500=${LAST_RUN_ROOT}
DATASET_DIR_500=${LAST_DATASET_DIR}

RETURN_NAME="stage21a_train_recovery_shadow_a0_500ep_return_${PIPELINE_TAG}"
DEST="${RETURN_ROOT}/${RETURN_NAME}"
test ! -e "${DEST}"
mkdir -p "${DEST}/smoke_runs" "${DEST}/datasets" "${DEST}/episode_manifests"

# Keep the main 500ep result at the package root to follow the project return contract.
cp -a "${RUN_ROOT_500}/progress.json" "${RUN_ROOT_500}/result.json" "${DEST}/"
cp -a "${RUN_ROOT_500}/vlmap_safety_debug" "${DEST}/"
cp -a "${RUN_ROOT_4}" "${DEST}/smoke_runs/4ep"
cp -a "${RUN_ROOT_40}" "${DEST}/smoke_runs/40ep"
cp -a "${DATASET_DIR_4}" "${DEST}/datasets/4ep"
cp -a "${DATASET_DIR_40}" "${DEST}/datasets/40ep"
cp -a "${DATASET_DIR_500}" "${DEST}/datasets/500ep"
cp -a "${MANIFEST_DIR}"/train_balanced_{4,40,500}_episode_ids*.json "${DEST}/episode_manifests/"
if [[ -n "${PIPELINE_LOG}" && -f "${PIPELINE_LOG}" ]]; then
  cp -a "${PIPELINE_LOG}" "${DEST}/overnight_pipeline.log"
fi

git rev-parse HEAD > "${DEST}/git_commit.txt"
git status --short > "${DEST}/git_status_short.txt"
printf '%s\n' "${RUN_NAME_500}" > "${DEST}/RUN_NAME.txt"
find "${DEST}" -type f | sort > "${DEST}/RETURN_MANIFEST.txt"

PIPELINE_COMPLETE=1
echo "STAGE21_PIPELINE_STATUS=complete"
echo "RETURN_NAME=${RETURN_NAME}"
echo "DEST=$(readlink -f "${DEST}")"
du -sh "${DEST}"
