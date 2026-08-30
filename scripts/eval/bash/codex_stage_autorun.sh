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

PHASE="stage57_local_elevation_support_v4_mix5"
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
  *)
    echo "unsupported codex experiment phase: ${PHASE}" >&2
    exit 3
    ;;
esac
