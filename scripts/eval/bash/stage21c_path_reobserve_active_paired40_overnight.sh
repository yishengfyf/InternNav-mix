#!/usr/bin/env bash
# Overnight Stage21c paired-N active audit.
# Generates explicit per-episode seeds, runs Frozen control and strict-only
# bounded path reobserve active, then performs an automatic integrity audit.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

CHECKPOINT=${STAGE21C_SCORER_CHECKPOINT:?Set STAGE21C_SCORER_CHECKPOINT to seed_53/best.pt}
EXPECTED_EPISODES=${STAGE21C_EXPECTED_EPISODES:-40}
BASE_MANIFEST=${STAGE21C_BASE_EPISODE_MANIFEST:-data/stage18/train_balanced_${EXPECTED_EPISODES}_episode_ids.json}
EPISODES_FILE=${STAGE21C_TRAIN_EPISODES_FILE:-data/vln_ce/raw_data/r2r/train/train.json.gz}
REFERENCE_MANIFEST=${STAGE21C_REFERENCE_MANIFEST:-scripts/eval/manifests/stage21c_strict_active_5_episode_ids.json}
REFERENCE_ROOT=${STAGE21C_REPLAY_REFERENCE_ROOT:-results/stage_17/stage21c_multitask_scorer_shadow500_return_20260814_134102}
RETURN_ROOT=${STAGE21_RETURN_ROOT:-results/stage_17}
MANIFEST_DIR=${STAGE21C_MANIFEST_DIR:-data/stage21c}
PIPELINE_TAG=${STAGE21_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
CUDA_DEVICES=${STAGE21_CUDA_VISIBLE_DEVICES:-0,1,2,3}
[[ "${EXPECTED_EPISODES}" =~ ^[1-9][0-9]*$ ]]

MANIFEST="${MANIFEST_DIR}/stage21c_path_reobserve_active_paired${EXPECTED_EPISODES}_episode_ids_${PIPELINE_TAG}.json"
RUNNING_NAME="stage21c_path_reobserve_active_paired${EXPECTED_EPISODES}_overnight_running_${PIPELINE_TAG}"
SUCCESS_NAME="stage21c_path_reobserve_active_paired${EXPECTED_EPISODES}_overnight_return_${PIPELINE_TAG}"
FAILURE_NAME="stage21c_path_reobserve_active_paired${EXPECTED_EPISODES}_overnight_failure_return_${PIPELINE_TAG}"
WORK_DIR="${RETURN_ROOT}/${RUNNING_NAME}"
SUCCESS_DEST="${RETURN_ROOT}/${SUCCESS_NAME}"
FAILURE_DEST="${RETURN_ROOT}/${FAILURE_NAME}"
CONTROL_NAME="compare_vlmap_stage21c_path_reobserve_control_paired${EXPECTED_EPISODES}_${PIPELINE_TAG}"
ACTIVE_NAME="compare_vlmap_stage21c_path_reobserve_active_paired${EXPECTED_EPISODES}_${PIPELINE_TAG}"
CONTROL_ROOT="logs/habitat/${CONTROL_NAME}"
ACTIVE_ROOT="logs/habitat/${ACTIVE_NAME}"
FAILED_STAGE=initialization
PIPELINE_COMPLETE=0

write_metadata() {
  local root=$1
  git rev-parse HEAD > "${root}/git_commit.txt"
  git status --short > "${root}/git_status_short.txt"
  printf '%s\n' "${CHECKPOINT}" > "${root}/SOURCE_CHECKPOINT.txt"
  printf '%s\n' "${BASE_MANIFEST}" > "${root}/SOURCE_BASE_MANIFEST.txt"
  printf '%s\n' "${MANIFEST}" > "${root}/SOURCE_MANIFEST.txt"
  printf '%s\n' "${REFERENCE_ROOT}" > "${root}/SOURCE_REFERENCE_ROOT.txt"
  if [[ -f "${MANIFEST}" ]]; then
    sha256sum "${MANIFEST}" > "${root}/MANIFEST_SHA256.txt"
  fi
  cat > "${root}/EXPERIMENT_SCOPE.txt" <<EOF
Overnight Stage21c paired-${EXPECTED_EPISODES} information-collection run. The manifest is
derived from a balanced-${EXPECTED_EPISODES} episode list and receives deterministic explicit
episode_eval_seed values; the five known strict episodes retain their prior
seeds. Control is Frozen S2/NextDiT. Active permits only one strict bounded
known-free-path reorient/reobserve intervention per episode. No blind forward,
Stage21d context, or relaxed occupied/unknown/path-corridor gate is enabled.
The audit compares old reference metrics/loops only for the five known
reference episodes; all other episodes are audited by control/active pairing
and seed replay.
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
  mkdir -p "${WORK_DIR}/partial_runs" "${WORK_DIR}/episode_manifests"
  package_run "${CONTROL_ROOT}" "${WORK_DIR}/partial_runs/control"
  package_run "${ACTIVE_ROOT}" "${WORK_DIR}/partial_runs/active"
  [[ -f "${MANIFEST}" ]] && cp -a "${MANIFEST}" "${WORK_DIR}/episode_manifests/"
  printf '%s\n' "${FAILED_STAGE}" > "${WORK_DIR}/FAILED_STAGE.txt"
  printf '%s\n' "${exit_status}" > "${WORK_DIR}/EXIT_STATUS.txt"
  write_metadata "${WORK_DIR}"
  mv "${WORK_DIR}" "${FAILURE_DEST}"
  find "${FAILURE_DEST}" -type f | sort > "${FAILURE_DEST}/RETURN_MANIFEST.txt"
  echo "STAGE21C_PATH_REOBSERVE_PAIRED${EXPECTED_EPISODES}_STATUS=failed"
  echo "FAILED_STAGE=${FAILED_STAGE}"
  echo "DEST=$(readlink -f "${FAILURE_DEST}")"
}
trap package_failure EXIT

mkdir -p "${RETURN_ROOT}" "${MANIFEST_DIR}"
test -f "${CHECKPOINT}"
test -f "${REFERENCE_MANIFEST}"
test -f "${REFERENCE_ROOT}/progress.json"
test ! -e "${WORK_DIR}"; test ! -e "${SUCCESS_DEST}"; test ! -e "${FAILURE_DEST}"
test ! -e "${CONTROL_ROOT}"; test ! -e "${ACTIVE_ROOT}"

if [[ ! -f "${BASE_MANIFEST}" ]]; then
  FAILED_STAGE=base_manifest_generation
  test -f "${EPISODES_FILE}"
  mkdir -p "$(dirname "${BASE_MANIFEST}")"
  python3 scripts/eval/select_balanced_r2r_episodes.py \
    --episodes-file "${EPISODES_FILE}" --output "${BASE_MANIFEST}" \
    --summary-output "${BASE_MANIFEST%.json}_summary.json" \
    --max-episodes "${EXPECTED_EPISODES}" --seed 21 --shuffle-within-scene
fi

FAILED_STAGE=manifest_generation
python3 scripts/eval/build_stage21c_episode_seed_manifest.py \
  --input "${BASE_MANIFEST}" --output "${MANIFEST}" \
  --overrides "${REFERENCE_MANIFEST}" --include "${REFERENCE_MANIFEST}" \
  --max-episodes "${EXPECTED_EPISODES}" --base-seed 300000
python3 - "${MANIFEST}" "${EXPECTED_EPISODES}" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
expected = int(sys.argv[2])
assert len(rows) == expected, len(rows)
assert len({(r['scene_id'], int(r['episode_id'])) for r in rows}) == expected
assert len({int(r['episode_eval_seed']) for r in rows}) == expected
print(json.dumps({'episode_count': len(rows), 'unique_seed_count': len(rows), 'expected': expected}))
PY

mkdir -p "${WORK_DIR}"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

FAILED_STAGE=control_evaluation
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
STAGE21_EPISODE_IDS="${MANIFEST}" STAGE21_RUN_NAME="${CONTROL_NAME}" \
STAGE21_EPISODE_SEED_REPLAY_MANIFEST="${MANIFEST}" \
STAGE21C_SCORER_CHECKPOINT="${CHECKPOINT}" STAGE21C_SCORER_DEVICE=cpu \
STAGE21_EVAL_PORT=${STAGE21C_PATH_CONTROL_EVAL_PORT:-2541} \
NPROC_PER_NODE=4 MASTER_PORT=${STAGE21C_PATH_CONTROL_MASTER_PORT:-2542} \
bash scripts/eval/bash/stage21_torchrun_eval.sh \
  --config scripts/eval/configs/habitat_dual_system_vlmap_stage21c_multitask_scorer_shadow_cfg.py

FAILED_STAGE=active_evaluation
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
STAGE21_EPISODE_IDS="${MANIFEST}" STAGE21_RUN_NAME="${ACTIVE_NAME}" \
STAGE21_EPISODE_SEED_REPLAY_MANIFEST="${MANIFEST}" \
STAGE21C_SCORER_CHECKPOINT="${CHECKPOINT}" STAGE21C_SCORER_DEVICE=cpu \
STAGE21C_PATH_ACTIVE_EVAL_PORT=${STAGE21C_PATH_ACTIVE_EVAL_PORT:-2543} \
NPROC_PER_NODE=4 MASTER_PORT=${STAGE21C_PATH_ACTIVE_MASTER_PORT:-2544} \
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

FAILED_STAGE=diagnostic_summary
python3 scripts/eval/analyze_stage21c_path_reobserve_diagnostics.py \
  --active-root "${ACTIVE_ROOT}" \
  --audit "${ACTIVE_ROOT}/stage21c_path_reobserve_active_paired_audit.json" \
  --output "${ACTIVE_ROOT}/stage21c_path_reobserve_active_diagnostics.json"

FAILED_STAGE=automatic_audit_summary
python3 - "${ACTIVE_ROOT}/stage21c_path_reobserve_active_paired_audit.json" "${EXPECTED_EPISODES}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
expected = int(sys.argv[2])
assert d['integrity_passed'] is True, d
assert d['control_episode_count'] == expected, d
assert d['active_episode_count'] == expected, d
assert d['common_episode_count'] == expected, d
assert d['seed_replay_verified_count'] == expected, d
assert d['reference_checked_episode_count'] == 5, d
assert d['reference_metric_verified_count'] == 5, d
assert d['reference_loop_verified_count'] == 5, d
assert all(v == 0 for v in d['violations'].values()), d['violations']
print(json.dumps({
    'episodes': d['expected_episode_count'],
    'active_experiment_formed': d['active_experiment_formed'],
    'applied_intervention_count': d['applied_intervention_count'],
    'reorient_completed_event_count': d['reorient_completed_event_count'],
    'post_reobserve_state_changed_count': d['post_reobserve_state_changed_count'],
    'path_pixel_applied_event_count': d['path_pixel_applied_event_count'],
    'failed_to_success_count': d['failed_to_success_count'],
    'success_to_failed_count': d['success_to_failed_count'],
    'paired_aggregate': d['paired_aggregate'],
}, ensure_ascii=False, indent=2))
PY

FAILED_STAGE=return_packaging
package_run "${CONTROL_ROOT}" "${WORK_DIR}/control"
package_run "${ACTIVE_ROOT}" "${WORK_DIR}/active_full"
cp -a "${ACTIVE_ROOT}/progress.json" "${ACTIVE_ROOT}/result.json" \
  "${ACTIVE_ROOT}/stage21c_path_reobserve_active_paired_audit.json" "${WORK_DIR}/"
cp -a "${ACTIVE_ROOT}/vlmap_safety_debug" "${WORK_DIR}/"
mkdir -p "${WORK_DIR}/episode_manifests"
cp -a "${MANIFEST}" "${WORK_DIR}/episode_manifests/"
cp -a "${REFERENCE_MANIFEST}" "${WORK_DIR}/episode_manifests/"
printf '%s\n' "0" > "${WORK_DIR}/EXIT_STATUS.txt"
write_metadata "${WORK_DIR}"
mv "${WORK_DIR}" "${SUCCESS_DEST}"
find "${SUCCESS_DEST}" -type f | sort > "${SUCCESS_DEST}/RETURN_MANIFEST.txt"
PIPELINE_COMPLETE=1

echo "STAGE21C_PATH_REOBSERVE_PAIRED${EXPECTED_EPISODES}_STATUS=complete"
echo "RETURN_NAME=${SUCCESS_NAME}"
echo "DEST=$(readlink -f "${SUCCESS_DEST}")"
du -sh "${SUCCESS_DEST}"
