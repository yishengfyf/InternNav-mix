#!/usr/bin/env bash
# Always attempt both independent experiments in order and preserve both packages.

set -u

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

PIPELINE_TAG=${STAGE21_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
export STAGE21_PIPELINE_TAG="${PIPELINE_TAG}"

echo "OVERNIGHT_STAGE=stage21c_paired_active"
bash scripts/eval/bash/stage21c_strict_loop_active_paired_tiny.sh
ACTIVE_STATUS=$?

echo "OVERNIGHT_STAGE=stage21d_context_shadow"
bash scripts/eval/bash/stage21d_recovery_context_shadow_10to40.sh
CONTEXT_STATUS=$?

echo "OVERNIGHT_COMPLETE=1"
echo "PIPELINE_TAG=${PIPELINE_TAG}"
echo "STAGE21C_ACTIVE_EXIT_STATUS=${ACTIVE_STATUS}"
echo "STAGE21D_CONTEXT_EXIT_STATUS=${CONTEXT_STATUS}"

if [[ "${ACTIVE_STATUS}" != "0" || "${CONTEXT_STATUS}" != "0" ]]; then
  exit 1
fi
