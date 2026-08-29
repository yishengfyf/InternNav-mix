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

PHASE="stage56_height_bin_mix5"
TAG="${PHASE}_$(date +%Y%m%d_%H%M%S)"

source /home/yifeifeng/miniconda3/etc/profile.d/conda.sh
conda activate habiinter

case "${PHASE}" in
  stage56_height_bin_mix5)
    export STAGE21C_SCORER_CHECKPOINT=/data/usr_data/yifeifeng/internnav/stage_results/shared_checkpoints/stage21b_seed_53/best.pt
    export STAGE56_MANIFEST=/home/yifeifeng/workspace/InternNav/scripts/eval/manifests/stage23a_sensor_layer12_mix5_episode_seed_replay.json
    export STAGE56_PIPELINE_TAG="${TAG}"
    export STAGE56_RETURN_ROOT=results/stage_17
    exec bash scripts/eval/bash/stage56_floor_frame_consensus_mix5.sh
    ;;
  *)
    echo "unsupported codex experiment phase: ${PHASE}" >&2
    exit 3
    ;;
esac
