#!/usr/bin/env bash
set -uo pipefail

expected_root=/data/usr_data/yifeifeng/internnav/worktrees/dualvln-spatial-memory-v1
repo_root=$(git rev-parse --show-toplevel)
if [[ "${repo_root}" != "${expected_root}" ]] || [[ -n "$(git status --porcelain)" ]]; then
    echo "refusing non-isolated or dirty server worktree: ${repo_root}" >&2
    exit 2
fi

result_root=/data/usr_data/yifeifeng/internnav/dualvln_mainline/results/codex_latest_return
mkdir -p "${result_root}"
exec 9>"${result_root}/task_state_repair_autorun.lock"
flock -n 9 || { echo "another task-state repair is active" >&2; exit 3; }

guard_gpu() {
    for gpu in 0 1 2 3; do
        free_mib=$(nvidia-smi --id="${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
        [[ -n "${free_mib}" && "${free_mib}" -ge 24000 ]] || return 1
    done
}
guard_gpu || { echo "a GPU is below the protected 24000 MiB threshold" >&2; exit 3; }

python=/home/yifeifeng/miniconda3/envs/internvla/bin/python
commit=$(git rev-parse --short=12 HEAD)
stage_id="task_state_repair_$(date -u +%Y%m%dT%H%M%SZ)_${commit}"
stage_dir=${result_root}/${stage_id}
mkdir -p "${stage_dir}"

"${python}" -m pytest -q \
    tests/unit_test/test_evidence_memory.py \
    tests/unit_test/test_evidence_conditioning.py \
    tests/unit_test/test_internvla_evidence_forward.py \
    tests/unit_test/test_evidence_inference.py \
    tests/unit_test/test_trainable_policy.py \
    2>&1 | tee "${stage_dir}/protocol_test.log" || exit ${PIPESTATUS[0]}

overall_status=0
for variant in R0 R1 R2; do
    guard_gpu || { overall_status=1; break; }
    case_dir=${stage_dir}/${variant}
    mkdir -p "${case_dir}/train" "${case_dir}/probe"
    extra_args=()
    if [[ "${variant}" == R1 || "${variant}" == R2 ]]; then
        extra_args+=(--task-contrastive-weight 1.0 --task-contrastive-margin 0.2)
    fi
    if [[ "${variant}" == R2 ]]; then
        extra_args+=(--stage-loss-weight 0.1)
    fi
    set +e
    CUDA_VISIBLE_DEVICES=0,1,2,3 "${python}" scripts/dualvln_mainline/real_overfit.py \
        --output-dir "${case_dir}/train" --run-id "${stage_id}_${variant}" --commit "${commit}" \
        --samples 8 --steps 80 --sharded --gradient-bypass --bypass-mode cross_attention \
        --bypass-scale 0.5 --experiment M2 --seed 23 --task-state-lr-scale 0.1 \
        --instruction-only-task-state "${extra_args[@]}" 2>&1 | tee "${case_dir}/train.log"
    train_status=${PIPESTATUS[0]}
    set -e
    [[ -f "${case_dir}/train/adapter_state.pt" ]] || { overall_status=1; continue; }
    guard_gpu || { overall_status=1; break; }
    set +e
    CUDA_VISIBLE_DEVICES=0,1,2,3 "${python}" scripts/dualvln_mainline/task_state_probe.py \
        --output-dir "${case_dir}/probe" --adapter "${case_dir}/train/adapter_state.pt" \
        --samples 32 --seed 23 --commit "${commit}" --instruction-only-task-state \
        2>&1 | tee "${case_dir}/probe.log"
    probe_status=${PIPESTATUS[0]}
    set -e
    [[ "${train_status}" -eq 0 && "${probe_status}" -eq 0 ]] || overall_status=1
done

set +e
"${python}" scripts/dualvln_mainline/analyze_task_state_repair.py \
    --stage-dir "${stage_dir}" --commit "${commit}" 2>&1 | tee "${stage_dir}/analysis.log"
analysis_status=${PIPESTATUS[0]}
set -e
sha256sum \
    scripts/dualvln_mainline/real_overfit.py \
    scripts/dualvln_mainline/task_state_probe.py \
    scripts/dualvln_mainline/analyze_task_state_repair.py \
    scripts/dualvln_mainline/task_state_repair_autorun.sh \
    internnav/model/basemodel/internvla_n1/internvla_n1.py \
    >"${stage_dir}/SOURCE_SHA256SUMS"
ln -sfn "${stage_id}" "${result_root}/latest_task_state_repair"
echo "Task-state repair summary: ${stage_dir}/summary.md"
exit "$((overall_status || analysis_status))"
