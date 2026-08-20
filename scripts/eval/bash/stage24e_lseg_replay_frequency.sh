#!/usr/bin/env bash
# Stage24E offline Q/Q+K/ALL LSeg frequency audit on frozen replay ledgers.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

LEDGER_ROOT=${STAGE24E_LEDGER_ROOT:?Set STAGE24E_LEDGER_ROOT to a Stage24D return directory}
CHECKPOINT=${STAGE24E_LSEG_CHECKPOINT:?Set STAGE24E_LSEG_CHECKPOINT to tensor-only weights}
VLMAPS_REPO=${STAGE24E_VLMAPS_REPO:-/home/yifeifeng/workspace/vlmaps}
TAG=${STAGE24E_TAG:-$(date +%Y%m%d_%H%M%S)}
RETURN_ROOT=${STAGE21_RETURN_ROOT:-results/stage_17}
WORK_DIR="${RETURN_ROOT}/stage24e_lseg_frequency_running_${TAG}"
SUCCESS_DIR="${RETURN_ROOT}/stage24e_lseg_frequency_return_${TAG}"
FAILURE_DIR="${RETURN_ROOT}/stage24e_lseg_frequency_failure_return_${TAG}"
CUDA_DEVICES=${STAGE24E_CUDA_VISIBLE_DEVICES:-0}
NPROC=${STAGE24E_NPROC_PER_NODE:-1}
FAILED_STAGE=initialization
COMPLETE=0

package_failure() {
  local status=$?
  if [[ "${COMPLETE}" == "1" || "${status}" == "0" ]]; then return; fi
  printf '%s\n' "${FAILED_STAGE}" > "${WORK_DIR}/FAILED_STAGE.txt"
  printf '%s\n' "${status}" > "${WORK_DIR}/EXIT_STATUS.txt"
  mv "${WORK_DIR}" "${FAILURE_DIR}"
  echo "STAGE24E_STATUS=failed"
  echo "DEST=$(readlink -f "${FAILURE_DIR}")"
}
trap package_failure EXIT

test -d "${LEDGER_ROOT}"
test -d "${VLMAPS_REPO}"
test -f "${CHECKPOINT}"
test ! -e "${WORK_DIR}"
test ! -e "${SUCCESS_DIR}"
test ! -e "${FAILURE_DIR}"
mkdir -p "${WORK_DIR}"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

FAILED_STAGE=replay_frequency_evaluation
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun \
  --standalone --nproc_per_node="${NPROC}" \
  scripts/eval/replay_stage24e_lseg_frequency.py \
  --ledger-root "${LEDGER_ROOT}" --output-root "${WORK_DIR}/frequency_audit" \
  --vlmaps-repo "${VLMAPS_REPO}" --checkpoint "${CHECKPOINT}" \
  --device distributed

FAILED_STAGE=frequency_gate_analysis
python3 scripts/eval/analyze_stage24e_lseg_frequency.py \
  --root "${WORK_DIR}/frequency_audit" \
  --output "${WORK_DIR}/stage24e_lseg_frequency_audit.json"

FAILED_STAGE=return_packaging
printf '%s\n' "0" > "${WORK_DIR}/EXIT_STATUS.txt"
printf '%s\n' "${LEDGER_ROOT}" > "${WORK_DIR}/LEDGER_ROOT.txt"
printf '%s\n' "${CHECKPOINT}" > "${WORK_DIR}/SOURCE_CHECKPOINT.txt"
git rev-parse HEAD > "${WORK_DIR}/git_commit.txt"
git status --short > "${WORK_DIR}/git_status_short.txt"
find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
mv "${WORK_DIR}" "${SUCCESS_DIR}"
COMPLETE=1
echo "STAGE24E_STATUS=complete"
echo "DEST=$(readlink -f "${SUCCESS_DIR}")"
du -sh "${SUCCESS_DIR}"
