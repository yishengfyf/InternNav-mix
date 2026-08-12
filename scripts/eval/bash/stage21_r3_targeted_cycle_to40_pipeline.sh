#!/usr/bin/env bash
# Stage21a-r3 bounded daytime pipeline:
# targeted 8ep cycle smoke -> audit -> balanced 40ep/4GPU shadow -> audit -> package.
# This script never starts the 500ep stage.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

EPISODES_FILE=${STAGE21_TRAIN_EPISODES_FILE:-data/vln_ce/raw_data/r2r/train/train.json.gz}
TARGETED_MANIFEST=${STAGE21_R3_TARGETED_MANIFEST:-data/stage21/stage21a_r3_targeted_8_episode_ids.json}
PIPELINE_TAG=${STAGE21_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
MANIFEST_DIR=${STAGE21_MANIFEST_DIR:-data/stage21}
DATASET_ROOT=${STAGE21_DATASET_ROOT:-data/stage21/datasets}
RETURN_ROOT=${STAGE21_RETURN_ROOT:-results/stage_17}
PIPELINE_LOG=${STAGE21_PIPELINE_LOG:-}

PIPELINE_COMPLETE=0
CURRENT_RUN_ROOT=""
CURRENT_DATASET_DIR=""
RUN_ROOT_SMOKE=""
RUN_ROOT_40=""
DATASET_DIR_SMOKE=""
DATASET_DIR_40=""

package_failure() {
  local exit_status=$?
  if [[ "${PIPELINE_COMPLETE}" == "1" || "${exit_status}" == "0" ]]; then
    return
  fi
  local failure_name="stage21a_r3_targeted_cycle_to40_failure_return_${PIPELINE_TAG}"
  local failure_dest="${RETURN_ROOT}/${failure_name}"
  mkdir -p "${failure_dest}/partial_runs" "${failure_dest}/partial_datasets" \
    "${failure_dest}/episode_manifests"
  [[ -n "${RUN_ROOT_SMOKE}" && -d "${RUN_ROOT_SMOKE}" ]] && cp -a "${RUN_ROOT_SMOKE}" "${failure_dest}/partial_runs/smoke8ep"
  [[ -n "${RUN_ROOT_40}" && -d "${RUN_ROOT_40}" ]] && cp -a "${RUN_ROOT_40}" "${failure_dest}/partial_runs/40ep"
  [[ -n "${CURRENT_RUN_ROOT}" && -d "${CURRENT_RUN_ROOT}" ]] && cp -a "${CURRENT_RUN_ROOT}" "${failure_dest}/partial_runs/current"
  [[ -n "${DATASET_DIR_SMOKE}" && -d "${DATASET_DIR_SMOKE}" ]] && cp -a "${DATASET_DIR_SMOKE}" "${failure_dest}/partial_datasets/smoke8ep"
  [[ -n "${DATASET_DIR_40}" && -d "${DATASET_DIR_40}" ]] && cp -a "${DATASET_DIR_40}" "${failure_dest}/partial_datasets/40ep"
  [[ -n "${CURRENT_DATASET_DIR}" && -d "${CURRENT_DATASET_DIR}" ]] && cp -a "${CURRENT_DATASET_DIR}" "${failure_dest}/partial_datasets/current"
  [[ -f "${TARGETED_MANIFEST}" ]] && cp -a "${TARGETED_MANIFEST}" "${failure_dest}/episode_manifests/"
  [[ -f "${MANIFEST_DIR}/train_balanced_40_episode_ids.json" ]] && cp -a "${MANIFEST_DIR}"/train_balanced_40_episode_ids*.json "${failure_dest}/episode_manifests/"
  [[ -n "${PIPELINE_LOG}" && -f "${PIPELINE_LOG}" ]] && cp -a "${PIPELINE_LOG}" "${failure_dest}/pipeline.log"
  git rev-parse HEAD > "${failure_dest}/git_commit.txt"
  git status --short > "${failure_dest}/git_status_short.txt"
  printf '%s\n' "${exit_status}" > "${failure_dest}/EXIT_STATUS.txt"
  find "${failure_dest}" -type f | sort > "${failure_dest}/RETURN_MANIFEST.txt"
  echo "STAGE21_R3_PIPELINE_STATUS=failed"
  echo "FAILURE_DEST=$(readlink -f "${failure_dest}")"
}
trap package_failure EXIT

test -f "${EPISODES_FILE}"
test -f "${TARGETED_MANIFEST}"
mkdir -p "${MANIFEST_DIR}" "${DATASET_ROOT}" "${RETURN_ROOT}"

python3 - "${TARGETED_MANIFEST}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
rows = json.loads(path.read_text(encoding="utf-8"))
assert isinstance(rows, list) and 4 <= len(rows) <= 8, (path, len(rows))
assert all(isinstance(row, dict) and row.get("scene_id") is not None and row.get("episode_id") is not None for row in rows)
assert len({(str(row["scene_id"]), str(row["episode_id"])) for row in rows}) == len(rows)
print(json.dumps({"targeted_manifest": str(path), "episodes": len(rows)}))
PY

python3 scripts/eval/select_balanced_r2r_episodes.py \
  --episodes-file "${EPISODES_FILE}" \
  --output "${MANIFEST_DIR}/train_balanced_40_episode_ids.json" \
  --summary-output "${MANIFEST_DIR}/train_balanced_40_episode_ids_summary.json" \
  --max-episodes 40 --seed 21 --shuffle-within-scene

run_and_build() {
  local label=$1 manifest=$2 nproc=$3 eval_port=$4 master_port=$5 expected_dirs=$6
  local run_name="compare_vlmap_stage21a_r3_${label}_${PIPELINE_TAG}"
  local run_root="logs/habitat/${run_name}"
  local dataset_dir="${DATASET_ROOT}/${run_name}"
  CURRENT_RUN_ROOT=${run_root}; CURRENT_DATASET_DIR=${dataset_dir}
  test ! -e "${run_root}"; test ! -e "${dataset_dir}"
  STAGE21_EPISODE_IDS="${manifest}" STAGE21_RUN_NAME="${run_name}" \
  STAGE21_EVAL_PORT="${eval_port}" NPROC_PER_NODE="${nproc}" MASTER_PORT="${master_port}" \
  bash scripts/eval/bash/stage21_torchrun_eval.sh
  python3 scripts/eval/build_stage21_candidate_recoverability_dataset.py \
    --run-root "${run_root}" --episodes-file "${EPISODES_FILE}" --output-dir "${dataset_dir}" \
    --reference-frame episodic_gps --reference-coordinate-mode x_neg_y --gps-coordinate-mode x_neg_y \
    --quaternion-order xyzw --split-key scene --split-seed 21
  python3 - "${run_root}" "${dataset_dir}" "${label}" "${expected_dirs}" <<'PY'
import json, sys
from pathlib import Path
run_root = Path(sys.argv[1])
dataset_dir = Path(sys.argv[2])
label = sys.argv[3]
expected_dirs = int(sys.argv[4])
summary = json.loads((dataset_dir / "summary.json").read_text())
progress = [x for x in (run_root / "progress.json").read_text().splitlines() if x.strip()]
assert (run_root / "result.json").is_file()
assert summary["run_dir_count"] == expected_dirs
assert summary["event_schema_version"] == "stage21a_r3_v3"
assert summary["reference_join_rate"] >= 0.98
assert summary["split_audit"]["scene_overlap_count"] == 0
assert summary["candidate_recoverability_rows"]["gt_leakage_scan"]["passed"] is True
assert summary["active_safety_check"]["passed"] is True
assert summary["candidate_recoverability_rows"]["active_gate_safe_used_as_recovery_target"] is False
audit = summary["candidate_recoverability_rows"]
if label == "cycle_smoke8ep":
    assert summary["task_rows"]["recovery_proxy"] > 0, summary["task_rows"]
    assert audit["recovery_feature_coverage"]["row_count"] > 0, audit
    assert audit["cycle_feature_audit"]["passed"] is True, audit["cycle_feature_audit"]
else:
    assert summary["task_rows"]["recovery_proxy"] > 0, summary["task_rows"]
    coverage = audit["recovery_feature_coverage"]["rates"]
    for field in ("anchor_executable_exit_count", "anchor_connected_component_count", "anchor_branch_depth_mean", "anchor_short_cycle_risk"):
        assert coverage[field] >= 0.95, (field, coverage)
    assert audit["safety_proxy_audit"]["exact_one_rate"] < 0.99
    assert audit["cycle_feature_audit"]["passed"] is True, audit["cycle_feature_audit"]
print(json.dumps({"label": label, "episodes": len(progress), "task_rows": summary["task_rows"], "recovery_feature_coverage": audit["recovery_feature_coverage"], "cycle_feature_audit": audit["cycle_feature_audit"], "safety_proxy_audit": audit["safety_proxy_audit"]}, indent=2))
PY
  LAST_RUN_NAME=${run_name}; LAST_RUN_ROOT=${run_root}; LAST_DATASET_DIR=${dataset_dir}
}

run_and_build cycle_smoke8ep "${TARGETED_MANIFEST}" 1 2431 2432 1
RUN_ROOT_SMOKE=${LAST_RUN_ROOT}; DATASET_DIR_SMOKE=${LAST_DATASET_DIR}
run_and_build distribution40ep "${MANIFEST_DIR}/train_balanced_40_episode_ids.json" 4 2433 2434 4
RUN_ROOT_40=${LAST_RUN_ROOT}; DATASET_DIR_40=${LAST_DATASET_DIR}

RETURN_NAME="stage21a_r3_targeted_cycle_to40_return_${PIPELINE_TAG}"
DEST="${RETURN_ROOT}/${RETURN_NAME}"
test ! -e "${DEST}"
mkdir -p "${DEST}/smoke_runs" "${DEST}/datasets" "${DEST}/episode_manifests"
cp -a "${RUN_ROOT_40}/progress.json" "${RUN_ROOT_40}/result.json" "${DEST}/"
cp -a "${RUN_ROOT_40}/vlmap_safety_debug" "${DEST}/"
cp -a "${RUN_ROOT_SMOKE}" "${DEST}/smoke_runs/cycle8ep"
cp -a "${DATASET_DIR_SMOKE}" "${DEST}/datasets/cycle8ep"
cp -a "${DATASET_DIR_40}" "${DEST}/datasets/40ep"
cp -a "${TARGETED_MANIFEST}" "${DEST}/episode_manifests/"
cp -a "${MANIFEST_DIR}"/train_balanced_40_episode_ids*.json "${DEST}/episode_manifests/"
[[ -n "${PIPELINE_LOG}" && -f "${PIPELINE_LOG}" ]] && cp -a "${PIPELINE_LOG}" "${DEST}/pipeline.log"
git rev-parse HEAD > "${DEST}/git_commit.txt"; git status --short > "${DEST}/git_status_short.txt"
printf '%s\n' "${LAST_RUN_NAME}" > "${DEST}/RUN_NAME.txt"
printf '%s\n' "40" > "${DEST}/MAX_EPISODES.txt"
find "${DEST}" -type f | sort > "${DEST}/RETURN_MANIFEST.txt"
PIPELINE_COMPLETE=1
echo "STAGE21_R3_PIPELINE_STATUS=complete"
echo "PIPELINE_STOPPED_AFTER=40ep_audit"
echo "RETURN_NAME=${RETURN_NAME}"
echo "DEST=$(readlink -f "${DEST}")"
du -sh "${DEST}"
