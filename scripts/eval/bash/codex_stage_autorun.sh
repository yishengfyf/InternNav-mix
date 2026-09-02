#!/usr/bin/env bash
set -euo pipefail

# Stable, no-argument entry point for approved server experiments.  The phase
# is intentionally a source-controlled constant: changing it requires a
# reviewed commit, while the SSH approval rule remains unchanged.
if [[ "$#" -ne 0 ]]; then
  echo "codex_stage_autorun.sh does not accept arguments" >&2
  exit 2
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

PHASE="stage68_native_vlmap_shadow8"
TAG="${PHASE}_$(date +%Y%m%d_%H%M%S)"

source /home/yifeifeng/miniconda3/etc/profile.d/conda.sh
conda activate habiinter

case "${PHASE}" in
  stage56_height_bin_mix5)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE56_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage23a_sensor_layer12_mix5_episode_seed_replay.json
    export STAGE56_PIPELINE_TAG="${TAG}"
    export STAGE56_RETURN_ROOT=results/stage_17
    bash scripts/eval/bash/stage56_floor_frame_consensus_mix5.sh
    result_dir="${REPO_ROOT}/results/stage_17/stage56_floor_frame_consensus_return_${TAG}"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    test -d "${result_dir}"
    ln -sfn "$(basename "${result_dir}")" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  stage57_local_elevation_support_mix5)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE57_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage23a_sensor_layer12_mix5_episode_seed_replay.json
    export STAGE57_PIPELINE_TAG="${TAG}"
    export STAGE57_RETURN_ROOT=results/stage_17
    bash scripts/eval/bash/stage57_local_elevation_support_mix5.sh
    result_dir="${REPO_ROOT}/results/stage_17/stage57_local_elevation_support_return_${TAG}"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    test -d "${result_dir}"
    ln -sfn "$(basename "${result_dir}")" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  stage57_local_elevation_support_v2_mix5)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE57_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage23a_sensor_layer12_mix5_episode_seed_replay.json
    export STAGE57_PIPELINE_TAG="${TAG}"
    export STAGE57_RETURN_ROOT=results/stage_17
    bash scripts/eval/bash/stage57_local_elevation_support_mix5.sh
    result_dir="${REPO_ROOT}/results/stage_17/stage57_local_elevation_support_return_${TAG}"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    test -d "${result_dir}"
    ln -sfn "$(basename "${result_dir}")" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  stage57_local_elevation_support_v3_mix5)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE57_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage23a_sensor_layer12_mix5_episode_seed_replay.json
    export STAGE57_PIPELINE_TAG="${TAG}"
    export STAGE57_RETURN_ROOT=results/stage_17
    bash scripts/eval/bash/stage57_local_elevation_support_mix5.sh
    result_dir="${REPO_ROOT}/results/stage_17/stage57_local_elevation_support_return_${TAG}"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    test -d "${result_dir}"
    ln -sfn "$(basename "${result_dir}")" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  stage57_local_elevation_support_v4_mix5)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE57_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage23a_sensor_layer12_mix5_episode_seed_replay.json
    export STAGE57_PIPELINE_TAG="${TAG}"
    export STAGE57_RETURN_ROOT=results/stage_17
    bash scripts/eval/bash/stage57_local_elevation_support_mix5.sh
    result_dir="${REPO_ROOT}/results/stage_17/stage57_local_elevation_support_return_${TAG}"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    test -d "${result_dir}"
    ln -sfn "$(basename "${result_dir}")" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  stage57_local_elevation_support_v5_mix5)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE57_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage23a_sensor_layer12_mix5_episode_seed_replay.json
    export STAGE57_PIPELINE_TAG="${TAG}"
    export STAGE57_RETURN_ROOT=results/stage_17
    bash scripts/eval/bash/stage57_local_elevation_support_mix5.sh
    result_dir="${REPO_ROOT}/results/stage_17/stage57_local_elevation_support_return_${TAG}"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    test -d "${result_dir}"
    ln -sfn "$(basename "${result_dir}")" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  stage57_local_elevation_support_v6_mix5)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE57_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage23a_sensor_layer12_mix5_episode_seed_replay.json
    export STAGE57_PIPELINE_TAG="${TAG}"
    export STAGE57_RETURN_ROOT=results/stage_17
    bash scripts/eval/bash/stage57_local_elevation_support_mix5.sh
    result_dir="${REPO_ROOT}/results/stage_17/stage57_local_elevation_support_return_${TAG}"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    test -d "${result_dir}"
    ln -sfn "$(basename "${result_dir}")" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  stage57_local_elevation_support_v7_mix5)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE57_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage23a_sensor_layer12_mix5_episode_seed_replay.json
    export STAGE57_PIPELINE_TAG="${TAG}"
    export STAGE57_RETURN_ROOT=results/stage_17
    bash scripts/eval/bash/stage57_local_elevation_support_mix5.sh
    result_dir="${REPO_ROOT}/results/stage_17/stage57_local_elevation_support_return_${TAG}"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    test -d "${result_dir}"
    ln -sfn "$(basename "${result_dir}")" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  stage57_local_elevation_support_v8_mix5)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE57_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage23a_sensor_layer12_mix5_episode_seed_replay.json
    export STAGE57_PIPELINE_TAG="${TAG}"
    export STAGE57_RETURN_ROOT=results/stage_17
    bash scripts/eval/bash/stage57_local_elevation_support_mix5.sh
    result_dir="${REPO_ROOT}/results/stage_17/stage57_local_elevation_support_return_${TAG}"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    test -d "${result_dir}"
    ln -sfn "$(basename "${result_dir}")" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  stage57_local_elevation_support_v9_stratified48)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE57_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage28_semantic_candidate_stratified48_episode_seed_replay.json
    export STAGE57_PIPELINE_TAG="${TAG}"
    export STAGE57_RETURN_ROOT=results/stage_17
    bash scripts/eval/bash/stage57_local_elevation_support_mix5.sh
    result_dir="${REPO_ROOT}/results/stage_17/stage57_local_elevation_support_return_${TAG}"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    test -d "${result_dir}"
    ln -sfn "$(basename "${result_dir}")" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  stage58_geometry_contract_stratified48)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE58_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage28_semantic_candidate_stratified48_episode_seed_replay.json
    export STAGE58_PIPELINE_TAG="${TAG}"
    export STAGE58_RETURN_ROOT=results/stage_17
    bash scripts/eval/bash/stage58_geometry_contract.sh
    result_dir="${REPO_ROOT}/results/stage_17/stage58_geometry_contract_return_${TAG}"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    test -d "${result_dir}"
    ln -sfn "$(basename "${result_dir}")" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  stage58_support_policy_holdout96)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE58_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage25_gt_holdout96_scene_disjoint_v2.json
    export STAGE58_PIPELINE_TAG="${TAG}"
    export STAGE58_RUN_ROOT=/data/usr_data/yifeifeng/internnav/stage_results/runs
    export STAGE58_RETURN_ROOT=/data/usr_data/yifeifeng/internnav/stage_results
    bash scripts/eval/bash/stage58_support_policy96.sh
    result_dir="/data/usr_data/yifeifeng/internnav/stage_results/stage58_support_policy_return_${TAG}"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    test -d "${result_dir}"
    ln -sfn "${result_dir}" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  stage59_productive_onset_holdout96)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE59_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage25_gt_holdout96_scene_disjoint_v2.json
    export STAGE59_PIPELINE_TAG="${TAG}"
    export STAGE59_RUN_ROOT=/data/usr_data/yifeifeng/internnav/stage_results/runs
    export STAGE59_RETURN_ROOT=/data/usr_data/yifeifeng/internnav/stage_results
    bash scripts/eval/bash/stage59_productive_onset96.sh
    result_dir="/data/usr_data/yifeifeng/internnav/stage_results/stage59_productive_onset_return_${TAG}"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    test -d "${result_dir}"
    ln -sfn "${result_dir}" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  stage59_productive_onset_fresh500)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE59_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage25_gt_detector_fresh500_episode_seed_replay.json
    export STAGE59_PIPELINE_TAG="${TAG}"
    export STAGE59_RUN_ROOT=/data/usr_data/yifeifeng/internnav/stage_results/runs
    export STAGE59_RETURN_ROOT=/data/usr_data/yifeifeng/internnav/stage_results
    bash scripts/eval/bash/stage59_productive_onset96.sh
    result_dir="/data/usr_data/yifeifeng/internnav/stage_results/stage59_productive_onset_return_${TAG}"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    test -d "${result_dir}"
    ln -sfn "${result_dir}" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  stage60_post_turn_productive_targeted36)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE59_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage60_productive_d0_targeted36_episode_seed_replay.json
    export STAGE59_CONFIG=scripts/eval/configs/habitat_dual_system_vlmap_stage60_post_turn_productive_cfg.py
    export STAGE59_PIPELINE_TAG="${TAG}"
    export STAGE59_RUN_ROOT=/data/usr_data/yifeifeng/internnav/stage_results/runs
    export STAGE59_RETURN_ROOT=/data/usr_data/yifeifeng/internnav/stage_results
    bash scripts/eval/bash/stage59_productive_onset96.sh
    result_dir="/data/usr_data/yifeifeng/internnav/stage_results/stage59_productive_onset_return_${TAG}"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    test -d "${result_dir}"
    ln -sfn "${result_dir}" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  stage63_adaptive_reobserve_targeted36)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE59_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage60_productive_d0_targeted36_episode_seed_replay.json
    export STAGE59_CONFIG=scripts/eval/configs/habitat_dual_system_vlmap_stage63_adaptive_reobserve_cfg.py
    export STAGE59_PIPELINE_TAG="${TAG}"
    export STAGE59_RUN_ROOT=/data/usr_data/yifeifeng/internnav/stage_results/runs
    export STAGE59_RETURN_ROOT=/data/usr_data/yifeifeng/internnav/stage_results
    bash scripts/eval/bash/stage59_productive_onset96.sh
    result_dir="/data/usr_data/yifeifeng/internnav/stage_results/stage59_productive_onset_return_${TAG}"
    run_dir="/data/usr_data/yifeifeng/internnav/stage_results/runs/compare_vlmap_stage59_productive_onset_${TAG}"
    python3 scripts/eval/analyze_stage63_adaptive_reobserve.py \
      --run-root "${run_dir}" \
      --output "${result_dir}/stage63_adaptive_reobserve_audit.json"
    find "${result_dir}" -type f | sort > "${result_dir}/RETURN_MANIFEST.txt"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    test -d "${result_dir}"
    ln -sfn "${result_dir}" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  stage64_recovery_subtask_support_safe12)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE59_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage64_support_safe12_episode_seed_replay.json
    export STAGE59_CONFIG=scripts/eval/configs/habitat_dual_system_vlmap_stage64_recovery_subtask_cfg.py
    export STAGE59_PIPELINE_TAG="${TAG}"
    export STAGE59_RUN_ROOT=/data/usr_data/yifeifeng/internnav/stage_results/runs
    export STAGE59_RETURN_ROOT=/data/usr_data/yifeifeng/internnav/stage_results
    python3 -m pytest -q tests/unit_test/test_stage64_recovery_subtask.py
    bash scripts/eval/bash/stage59_productive_onset96.sh
    result_dir="/data/usr_data/yifeifeng/internnav/stage_results/stage59_productive_onset_return_${TAG}"
    run_dir="/data/usr_data/yifeifeng/internnav/stage_results/runs/compare_vlmap_stage59_productive_onset_${TAG}"
    python3 scripts/eval/analyze_stage64_recovery_subtask.py \
      --run-root "${run_dir}" \
      --expected-episodes 12 \
      --output "${result_dir}/stage64_recovery_subtask_audit.json" \
      --require-all
    find "${result_dir}" -type f | sort > "${result_dir}/RETURN_MANIFEST.txt"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    test -d "${result_dir}"
    ln -sfn "${result_dir}" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  stage65_native_recovery_active4)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE59_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage65_native_recovery_active4_episode_seed_replay.json
    export STAGE59_CONFIG=scripts/eval/configs/habitat_dual_system_vlmap_stage65_native_recovery_active_cfg.py
    export STAGE59_PIPELINE_TAG="${TAG}"
    export STAGE59_RUN_ROOT=/data/usr_data/yifeifeng/internnav/stage_results/runs
    export STAGE59_RETURN_ROOT=/data/usr_data/yifeifeng/internnav/stage_results
    export STAGE59_SKIP_AUDIT_REQUIRE_ALL=1
    python3 -m pytest -q tests/unit_test/test_stage64_recovery_subtask.py
    bash scripts/eval/bash/stage59_productive_onset96.sh
    result_dir="/data/usr_data/yifeifeng/internnav/stage_results/stage59_productive_onset_return_${TAG}"
    run_dir="/data/usr_data/yifeifeng/internnav/stage_results/runs/compare_vlmap_stage59_productive_onset_${TAG}"
    python3 scripts/eval/analyze_stage59_productive_onset.py \
      --run-root "${run_dir}" \
      --manifest "${STAGE59_MANIFEST}" \
      --output "${result_dir}/stage59_productive_onset_audit.json"
    python3 scripts/eval/analyze_stage65_native_recovery.py \
      --run-root "${run_dir}" \
      --expected-episodes 4 \
      --output "${result_dir}/stage65_native_recovery_audit.json"
    find "${result_dir}" -type f | sort > "${result_dir}/RETURN_MANIFEST.txt"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    test -d "${result_dir}"
    ln -sfn "${result_dir}" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  stage66_native_visual_audit8)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE59_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage66_native_visual_audit8_episode_seed_replay.json
    export STAGE59_CONFIG=scripts/eval/configs/habitat_dual_system_vlmap_stage66_native_visual_audit_cfg.py
    export STAGE59_PIPELINE_TAG="${TAG}"
    export STAGE59_RUN_ROOT=/data/usr_data/yifeifeng/internnav/stage_results/runs
    export STAGE59_RETURN_ROOT=/data/usr_data/yifeifeng/internnav/stage_results
    export STAGE59_SKIP_AUDIT_REQUIRE_ALL=1
    python3 -m pytest -q tests/unit_test/test_stage64_recovery_subtask.py
    bash scripts/eval/bash/stage59_productive_onset96.sh
    result_dir="/data/usr_data/yifeifeng/internnav/stage_results/stage59_productive_onset_return_${TAG}"
    run_dir="/data/usr_data/yifeifeng/internnav/stage_results/runs/compare_vlmap_stage59_productive_onset_${TAG}"
    python3 scripts/eval/analyze_stage59_productive_onset.py \
      --run-root "${run_dir}" \
      --manifest "${STAGE59_MANIFEST}" \
      --output "${result_dir}/stage59_productive_onset_audit.json"
    python3 scripts/eval/analyze_stage65_native_recovery.py \
      --run-root "${run_dir}" \
      --expected-episodes 8 \
      --output "${result_dir}/stage65_native_recovery_audit.json"
    find "${result_dir}" -type f | sort > "${result_dir}/RETURN_MANIFEST.txt"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    test -d "${result_dir}"
    ln -sfn "${result_dir}" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  stage66_package_visual_audit8)
    # Package the completed Stage66 run without rerunning Habitat.  Keep the
    # videos and JSON audit ledgers, but exclude the multi-GB raw RGB/depth
    # cache; the latter remains on the server for reproducible re-analysis.
    SOURCE_TAG="stage66_native_visual_audit8_20260902_115447"
    SOURCE_RUN="/data/usr_data/yifeifeng/internnav/stage_results/runs/compare_vlmap_stage59_productive_onset_${SOURCE_TAG}"
    SOURCE_RETURN="/data/usr_data/yifeifeng/internnav/stage_results/stage59_productive_onset_return_${SOURCE_TAG}"
    DEST="/data/usr_data/yifeifeng/internnav/stage_results/stage66_native_visual_audit8_return_${SOURCE_TAG}_v2"
    test -d "${SOURCE_RUN}"
    test ! -e "${DEST}"
    mkdir -p "${DEST}/run" "${DEST}/visual/vis_0" "${DEST}/visual/vlmap_safety_debug"
    cp -a "${SOURCE_RETURN}"/stage59_productive_onset_audit.json "${DEST}/"
    cp -a "${SOURCE_RETURN}"/stage65_native_recovery_audit.json "${DEST}/"
    cp -a "${SOURCE_RETURN}"/episode_manifests "${DEST}/"
    cp -a "${SOURCE_RUN}"/result.json "${SOURCE_RUN}"/progress.json "${DEST}/run/"
    cp -a "${SOURCE_RUN}"/vis_0/. "${DEST}/visual/vis_0/"
    while IFS= read -r src; do
      rel="${src#${SOURCE_RUN}/vlmap_safety_debug/}"
      dst="${DEST}/visual/vlmap_safety_debug/${rel}"
      mkdir -p "$(dirname "${dst}")"
      cp -a "${src}" "${dst}"
    done < <(find "${SOURCE_RUN}/vlmap_safety_debug" -type f \( -name '*.jsonl' -o -name '*.json' \) ! -path '*/replay_ledger/*/rgb/*' ! -path '*/replay_ledger/*/depth/*')
    printf '%s\n' 0 > "${DEST}/EXIT_STATUS.txt"
    git rev-parse HEAD > "${DEST}/git_commit.txt"
    git status --short > "${DEST}/git_status_short.txt"
    find "${DEST}" -type f | sort > "${DEST}/RETURN_MANIFEST.txt"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    ln -sfn "${DEST}" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  stage67_native_strict_shadow8)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE59_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage66_native_visual_audit8_episode_seed_replay.json
    export STAGE59_CONFIG=scripts/eval/configs/habitat_dual_system_vlmap_stage67_native_strict_shadow_cfg.py
    export STAGE59_PIPELINE_TAG="${TAG}"
    export STAGE59_RUN_ROOT=/data/usr_data/yifeifeng/internnav/stage_results/runs
    export STAGE59_RETURN_ROOT=/data/usr_data/yifeifeng/internnav/stage_results
    export STAGE59_SKIP_AUDIT_REQUIRE_ALL=1
    python3 -m pytest -q tests/unit_test/test_stage64_recovery_subtask.py
    bash scripts/eval/bash/stage59_productive_onset96.sh
    result_dir="/data/usr_data/yifeifeng/internnav/stage_results/stage67_native_strict_shadow_return_${TAG}"
    run_dir="/data/usr_data/yifeifeng/internnav/stage_results/runs/compare_vlmap_stage59_productive_onset_${TAG}"
    python3 scripts/eval/analyze_stage59_productive_onset.py \
      --run-root "${run_dir}" \
      --manifest "${STAGE59_MANIFEST}" \
      --output "${result_dir}/stage59_productive_onset_audit.json"
    python3 scripts/eval/analyze_stage65_native_recovery.py \
      --run-root "${run_dir}" \
      --expected-episodes 8 \
      --output "${result_dir}/stage65_native_recovery_audit.json"
    find "${result_dir}" -type f | sort > "${result_dir}/RETURN_MANIFEST.txt"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    test -d "${result_dir}"
    ln -sfn "${result_dir}" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  stage68_native_vlmap_shadow8)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE59_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage66_native_visual_audit8_episode_seed_replay.json
    export STAGE59_CONFIG=scripts/eval/configs/habitat_dual_system_vlmap_stage68_native_vlmap_shadow_cfg.py
    export STAGE59_PIPELINE_TAG="${TAG}"
    export STAGE59_RUN_ROOT=/data/usr_data/yifeifeng/internnav/stage_results/runs
    export STAGE59_RETURN_ROOT=/data/usr_data/yifeifeng/internnav/stage_results
    export STAGE59_SKIP_AUDIT_REQUIRE_ALL=1
    python3 -m pytest -q tests/unit_test/test_stage64_recovery_subtask.py
    bash scripts/eval/bash/stage59_productive_onset96.sh
    result_dir="/data/usr_data/yifeifeng/internnav/stage_results/stage68_native_vlmap_shadow_return_${TAG}"
    run_dir="/data/usr_data/yifeifeng/internnav/stage_results/runs/compare_vlmap_stage59_productive_onset_${TAG}"
    python3 scripts/eval/analyze_stage59_productive_onset.py \
      --run-root "${run_dir}" \
      --manifest "${STAGE59_MANIFEST}" \
      --output "${result_dir}/stage59_productive_onset_audit.json"
    python3 scripts/eval/analyze_stage65_native_recovery.py \
      --run-root "${run_dir}" \
      --expected-episodes 8 \
      --output "${result_dir}/stage65_native_recovery_audit.json"
    find "${result_dir}" -type f | sort > "${result_dir}/RETURN_MANIFEST.txt"
    latest_link="${REPO_ROOT}/results/stage_17/codex_latest_return"
    test -d "${result_dir}"
    ln -sfn "${result_dir}" "${latest_link}"
    echo "CODEX_LATEST_RETURN=${latest_link}"
    ;;
  *)
    echo "unsupported codex experiment phase: ${PHASE}" >&2
    exit 3
    ;;
esac
