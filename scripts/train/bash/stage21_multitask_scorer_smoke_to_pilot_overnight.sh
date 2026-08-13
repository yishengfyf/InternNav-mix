#!/usr/bin/env bash
# Stage21b offline chain: dataset audit -> 30-step smoke -> 3-seed pilot -> summary.
# It never launches Habitat, loads/updates S2 or NextDiT, or performs active recovery.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"

DATA_DIR=${STAGE21_TRAIN_DATA_DIR:?Set STAGE21_TRAIN_DATA_DIR to the audited 500ep dataset}
RETURN_ROOT=${STAGE21_TRAIN_RETURN_ROOT:-results/stage_17}
PIPELINE_TAG=${STAGE21_TRAIN_PIPELINE_TAG:-$(date +%Y%m%d_%H%M%S)}
GPU=${STAGE21_TRAIN_GPU:-0}
SMOKE_STEPS=${STAGE21_TRAIN_SMOKE_STEPS:-30}
SMOKE_EPOCHS=${STAGE21_TRAIN_SMOKE_EPOCHS:-2}
PILOT_EPOCHS=${STAGE21_TRAIN_PILOT_EPOCHS:-20}
PILOT_SEEDS_TEXT=${STAGE21_TRAIN_PILOT_SEEDS:-21 37 53}

RUNNING_NAME="stage21b_multitask_scorer_smoke_to_pilot_running_${PIPELINE_TAG}"
SUCCESS_NAME="stage21b_multitask_scorer_smoke_to_pilot_return_${PIPELINE_TAG}"
FAILURE_NAME="stage21b_multitask_scorer_smoke_to_pilot_failure_return_${PIPELINE_TAG}"
WORK_DIR="${RETURN_ROOT}/${RUNNING_NAME}"
SUCCESS_DEST="${RETURN_ROOT}/${SUCCESS_NAME}"
FAILURE_DEST="${RETURN_ROOT}/${FAILURE_NAME}"
FAILED_STAGE=initialization
PIPELINE_COMPLETE=0

mkdir -p "${RETURN_ROOT}"
test -d "${DATA_DIR}"
test -f "${DATA_DIR}/summary.json"
test ! -e "${WORK_DIR}"
test ! -e "${SUCCESS_DEST}"
test ! -e "${FAILURE_DEST}"
mkdir -p "${WORK_DIR}/smoke" "${WORK_DIR}/pilot"
exec > >(tee -a "${WORK_DIR}/pipeline.log") 2>&1

write_metadata() {
  local root=$1
  printf '%s\n' "${DATA_DIR}" > "${root}/SOURCE_DATASET.txt"
  cat > "${root}/TRAINING_SCOPE.txt" <<'EOF'
offline training only
Frozen S2/NextDiT: true
Habitat started: false
Episode-time parameter update: false
Active recovery: false
EOF
  git rev-parse HEAD > "${root}/git_commit.txt"
  git status --short > "${root}/git_status_short.txt"
}

package_failure() {
  local exit_status=$?
  if [[ "${PIPELINE_COMPLETE}" == "1" || "${exit_status}" == "0" ]]; then
    return
  fi
  if [[ -d "${WORK_DIR}" ]]; then
    printf '%s\n' "${FAILED_STAGE}" > "${WORK_DIR}/FAILED_STAGE.txt"
    printf '%s\n' "${exit_status}" > "${WORK_DIR}/EXIT_STATUS.txt"
    printf '%s\n' "offline pipeline stopped; no active navigation was attempted" > "${WORK_DIR}/PILOT_STOPPED_AFTER.txt"
    write_metadata "${WORK_DIR}"
    find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
    mv "${WORK_DIR}" "${FAILURE_DEST}"
    echo "STAGE21_MULTITASK_PILOT_STATUS=failed"
    echo "FAILED_STAGE=${FAILED_STAGE}"
    echo "FAILURE_DEST=$(readlink -f "${FAILURE_DEST}")"
  fi
}
trap package_failure EXIT

read -r -a PILOT_SEEDS <<< "${PILOT_SEEDS_TEXT}"
if [[ "${#PILOT_SEEDS[@]}" -eq 0 ]]; then
  echo "No pilot seeds configured" >&2
  exit 2
fi

FAILED_STAGE=dataset_audit
python3 scripts/train/audit_stage21_multitask_dataset.py \
  --data-dir "${DATA_DIR}" --output "${WORK_DIR}/dataset_audit.json"

FAILED_STAGE=smoke_train
CUDA_VISIBLE_DEVICES="${GPU}" \
python3 scripts/train/train_stage21_multitask_scorer.py \
  --data-dir "${DATA_DIR}" --output-dir "${WORK_DIR}/smoke" \
  --epochs "${SMOKE_EPOCHS}" --smoke-steps "${SMOKE_STEPS}" \
  --batch-size 128 --progress-batch-size 32 --hidden-dim 128 \
  --dropout 0.10 --lr 3e-4 --weight-decay 1e-4 \
  --seed 21 --device cuda \
  | tee "${WORK_DIR}/smoke_stdout.log"

FAILED_STAGE=smoke_artifact_audit
PYTHONPATH=scripts/train python3 -c \
  'import json,sys; from pathlib import Path; from summarize_stage21_multitask_pilot import audit_training_dir; print(json.dumps(audit_training_dir(Path(sys.argv[1]), minimum_steps=int(sys.argv[2])), indent=2, allow_nan=False))' \
  "${WORK_DIR}/smoke" "${SMOKE_STEPS}" > "${WORK_DIR}/smoke_artifact_audit.json"

for seed in "${PILOT_SEEDS[@]}"; do
  SEED_DIR="${WORK_DIR}/pilot/seed_${seed}"
  mkdir -p "${SEED_DIR}"
  FAILED_STAGE="pilot_seed_${seed}_train"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  python3 scripts/train/train_stage21_multitask_scorer.py \
    --data-dir "${DATA_DIR}" --output-dir "${SEED_DIR}" \
    --epochs "${PILOT_EPOCHS}" --smoke-steps 0 \
    --batch-size 128 --progress-batch-size 32 --hidden-dim 128 \
    --dropout 0.10 --lr 3e-4 --weight-decay 1e-4 \
    --seed "${seed}" --device cuda \
    | tee "${WORK_DIR}/pilot_seed_${seed}_stdout.log"

  FAILED_STAGE="pilot_seed_${seed}_artifact_audit"
  PYTHONPATH=scripts/train python3 -c \
    'import json,sys; from pathlib import Path; from summarize_stage21_multitask_pilot import audit_training_dir; print(json.dumps(audit_training_dir(Path(sys.argv[1]), expected_epochs=int(sys.argv[2]), minimum_steps=int(sys.argv[2])), indent=2, allow_nan=False))' \
    "${SEED_DIR}" "${PILOT_EPOCHS}" > "${WORK_DIR}/pilot_seed_${seed}_artifact_audit.json"
done

FAILED_STAGE=pilot_summary
python3 scripts/train/summarize_stage21_multitask_pilot.py \
  --smoke-dir "${WORK_DIR}/smoke" --pilot-root "${WORK_DIR}/pilot" \
  --seeds "${PILOT_SEEDS[@]}" --expected-smoke-steps "${SMOKE_STEPS}" \
  --expected-pilot-epochs "${PILOT_EPOCHS}" \
  --output "${WORK_DIR}/pilot_summary.json"

printf '%s\n' "completed offline dataset audit, smoke, and all pilot seeds; active navigation not started" \
  > "${WORK_DIR}/PILOT_STOPPED_AFTER.txt"
printf '%s\n' "0" > "${WORK_DIR}/EXIT_STATUS.txt"
write_metadata "${WORK_DIR}"
find "${WORK_DIR}" -type f | sort > "${WORK_DIR}/RETURN_MANIFEST.txt"
mv "${WORK_DIR}" "${SUCCESS_DEST}"
PIPELINE_COMPLETE=1

echo "STAGE21_MULTITASK_PILOT_STATUS=complete"
echo "PILOT_STOPPED_AFTER=offline_three_seed_pilot"
echo "RETURN_NAME=${SUCCESS_NAME}"
echo "DEST=$(readlink -f "${SUCCESS_DEST}")"
du -sh "${SUCCESS_DEST}"
