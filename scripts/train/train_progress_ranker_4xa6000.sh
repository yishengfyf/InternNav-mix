#!/bin/bash
# Stage17 ranker: a small tabular model, intentionally inexpensive on 4x A6000.
# Run the one-GPU smoke command first; only use torchrun after labels pass audit.

set -euo pipefail

DATA_DIR=${1:?"Usage: $0 <stage17_dataset_dir> [output_dir]"}
OUTPUT_DIR=${2:-checkpoints/stage17_progress_ranker}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29517}
EXTRA_ARGS=()

# A real progress-value dataset needs no override. Keep the temporary angle
# proxy behind an explicit environment variable even in this launcher.
if [[ "${ALLOW_ANGLE_PROXY_SMOKE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--allow-angle-proxy-training)
fi

torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  scripts/train/train_progress_ranker.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 30 \
  --batch-size 128 \
  --num-workers 4 \
  --hidden-dim 128 \
  --dropout 0.10 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --seed 17 \
  "${EXTRA_ARGS[@]}"
