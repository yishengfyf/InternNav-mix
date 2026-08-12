#!/usr/bin/env bash
# Stage21a-r3 bounded diagnostic: three representative failures -> snapshot audit -> package.
# This script is shadow-only and never starts 40/500ep, training, or active intervention.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

PIPELINE_TAG=${STAGE21_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
MANIFEST=${STAGE21_FAILURE_MANIFEST:-data/stage21/stage21a_r3_failure_snapshot_3_episode_ids.json}
RUN_NAME=${STAGE21_RUN_NAME:-compare_vlmap_stage21a_r3_failure_snapshot3_${PIPELINE_TAG}}
RUN_ROOT="logs/habitat/${RUN_NAME}"
RETURN_ROOT=${STAGE21_RETURN_ROOT:-results/stage_17}
RETURN_NAME="stage21a_r3_failure_snapshot3_return_${PIPELINE_TAG}"
DEST="${RETURN_ROOT}/${RETURN_NAME}"
AUDIT_PATH="${RUN_ROOT}/stuck_snapshot_audit.json"
PIPELINE_LOG=${STAGE21_PIPELINE_LOG:-logs/habitat/${RUN_NAME}_pipeline.log}
PIPELINE_COMPLETE=0

package_failure() {
  local exit_status=$?
  if [[ "${PIPELINE_COMPLETE}" == "1" || "${exit_status}" == "0" ]]; then
    return
  fi
  local failure_dest="${RETURN_ROOT}/stage21a_r3_failure_snapshot3_failure_return_${PIPELINE_TAG}"
  mkdir -p "${failure_dest}/episode_manifests"
  [[ -f "${RUN_ROOT}/progress.json" ]] && cp -a "${RUN_ROOT}/progress.json" "${failure_dest}/"
  [[ -f "${RUN_ROOT}/result.json" ]] && cp -a "${RUN_ROOT}/result.json" "${failure_dest}/"
  [[ -d "${RUN_ROOT}/vlmap_safety_debug" ]] && cp -a "${RUN_ROOT}/vlmap_safety_debug" "${failure_dest}/"
  [[ -f "${AUDIT_PATH}" ]] && cp -a "${AUDIT_PATH}" "${failure_dest}/"
  [[ -f "${PIPELINE_LOG}" ]] && cp -a "${PIPELINE_LOG}" "${failure_dest}/pipeline.log"
  [[ -f "${MANIFEST}" ]] && cp -a "${MANIFEST}" "${failure_dest}/episode_manifests/"
  git rev-parse HEAD > "${failure_dest}/git_commit.txt"
  git status --short > "${failure_dest}/git_status_short.txt"
  printf '%s\n' "${exit_status}" > "${failure_dest}/EXIT_STATUS.txt"
  find "${failure_dest}" -type f | sort > "${failure_dest}/RETURN_MANIFEST.txt"
  echo "STAGE21_FAILURE_SNAPSHOT_STATUS=failed"
  echo "FAILURE_DEST=$(readlink -f "${failure_dest}")"
}
trap package_failure EXIT

mkdir -p "$(dirname "${MANIFEST}")" "${RETURN_ROOT}"
python3 - "${MANIFEST}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
expected = [
    {"scene_id": "5q7pvUzZiYa", "episode_id": 9357},
    {"scene_id": "SN83YJsR3w2", "episode_id": 5982},
    {"scene_id": "V2XKFyX4ASd", "episode_id": 775},
]
if path.exists():
    current = json.loads(path.read_text(encoding="utf-8"))
    assert current == expected, (path, current)
else:
    path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"manifest": str(path), "episodes": expected}, ensure_ascii=False))
PY

test ! -e "${RUN_ROOT}"
test ! -e "${DEST}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
STAGE21_EPISODE_IDS="${MANIFEST}" \
STAGE21_RUN_NAME="${RUN_NAME}" \
STAGE21_EVAL_PORT=${STAGE21_EVAL_PORT:-2441} \
NPROC_PER_NODE=1 \
MASTER_PORT=${MASTER_PORT:-2442} \
bash scripts/eval/bash/stage21_torchrun_eval.sh \
  --config scripts/eval/configs/habitat_dual_system_vlmap_stage21a_r3_failure_snapshot_probe_cfg.py

python3 scripts/eval/analyze_stage21_stuck_snapshots.py \
  --run-root "${RUN_ROOT}" \
  --episode-manifest "${MANIFEST}" \
  --output "${AUDIT_PATH}" \
  --expected-seed 5q7pvUzZiYa/9357=200001 \
  --expected-seed SN83YJsR3w2/5982=100006 \
  --expected-seed V2XKFyX4ASd/775=7 \
  --require-all

mkdir -p "${DEST}/episode_manifests"
cp -a "${RUN_ROOT}/progress.json" "${RUN_ROOT}/result.json" "${AUDIT_PATH}" "${DEST}/"
cp -a "${RUN_ROOT}/vlmap_safety_debug" "${DEST}/"
[[ -f "${PIPELINE_LOG}" ]] && cp -a "${PIPELINE_LOG}" "${DEST}/pipeline.log"
cp -a "${MANIFEST}" "${DEST}/episode_manifests/"
git rev-parse HEAD > "${DEST}/git_commit.txt"
git status --short > "${DEST}/git_status_short.txt"
printf '%s\n' "${RUN_NAME}" > "${DEST}/RUN_NAME.txt"
printf '%s\n' "3" > "${DEST}/MAX_EPISODES.txt"
find "${DEST}" -type f | sort > "${DEST}/RETURN_MANIFEST.txt"

PIPELINE_COMPLETE=1
echo "STAGE21_FAILURE_SNAPSHOT_STATUS=complete"
echo "PIPELINE_STOPPED_AFTER=3ep_snapshot_audit"
echo "RETURN_NAME=${RETURN_NAME}"
echo "DEST=$(readlink -f "${DEST}")"
du -sh "${DEST}"
