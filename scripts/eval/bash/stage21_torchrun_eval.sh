#!/bin/bash
# Stage21 frozen-policy train-split shadow collection on multiple GPUs.

set -euo pipefail

CONFIG=scripts/eval/configs/habitat_dual_system_vlmap_stage21a_train_recovery_shadow_cfg.py
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-2421}

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

if [[ -z "${STAGE21_EPISODE_IDS:-}" ]]; then
  echo "STAGE21_EPISODE_IDS must be set." >&2
  exit 1
fi
if [[ -z "${STAGE21_RUN_NAME:-}" ]]; then
  echo "STAGE21_RUN_NAME must be set." >&2
  exit 1
fi

TORCHRUN_BIN=(torchrun)
if ! command -v torchrun >/dev/null 2>&1; then
  # Some evaluation environments ship torch.distributed.run but omit the
  # torchrun console entrypoint. Preserve identical arguments in that case.
  PYTHON_BIN=${PYTHON_BIN:-python}
  TORCHRUN_BIN=("${PYTHON_BIN}" -m torch.distributed.run)
fi

"${TORCHRUN_BIN[@]}" \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  scripts/eval/eval.py \
  --config "${CONFIG}"
