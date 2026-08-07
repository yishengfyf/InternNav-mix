#!/bin/bash
# Stage17 candidate collection eval on multiple GPUs.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 \
#   STAGE17_EPISODE_IDS=data/stage17/train_balanced_500_episode_ids.json \
#   STAGE17_BALANCED_RUN_NAME=compare_vlmap_stage17a_train_balanced500_occ_memory_target_frontier_shadow \
#   bash scripts/eval/bash/stage17_torchrun_eval.sh

set -euo pipefail

CONFIG=scripts/eval/configs/habitat_dual_system_vlmap_stage17a_train_balanced_shadow_cfg.py
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-2392}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --nproc-per-node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --master-port)
      MASTER_PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${STAGE17_EPISODE_IDS:-}" ]]; then
  echo "STAGE17_EPISODE_IDS must be set." >&2
  exit 1
fi

BALANCED_RUN_NAME="${STAGE17_BALANCED_RUN_NAME:-${STAGE20_BALANCED_RUN_NAME:-}}"
if [[ -z "${BALANCED_RUN_NAME}" ]]; then
  echo "STAGE17_BALANCED_RUN_NAME or STAGE20_BALANCED_RUN_NAME must be set." >&2
  exit 1
fi
export STAGE17_BALANCED_RUN_NAME="${BALANCED_RUN_NAME}"

torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  scripts/eval/eval.py \
  --config "${CONFIG}"
