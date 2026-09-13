#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 0 ]]; then
    echo "usage: $0" >&2
    exit 2
fi

expected_root=/data/usr_data/yifeifeng/internnav/worktrees/dualvln-spatial-memory-v1
repo_root=$(git rev-parse --show-toplevel)
if [[ "${repo_root}" != "${expected_root}" ]] || [[ -n "$(git status --porcelain)" ]]; then
    echo "refusing non-isolated or dirty server worktree: ${repo_root}" >&2
    exit 2
fi

result_root=/data/usr_data/yifeifeng/internnav/dualvln_mainline/results/codex_latest_return
alignment=/data/usr_data/yifeifeng/internnav/dualvln_mainline/data/task_alignment/r2r_17DRP5sb8fy.json
raw=/data/usr_data/yifeifeng/internnav/hf_repos/InternData-N1/vln_ce/raw_data/r2r/train/train.json.gz
tasks=/data/usr_data/yifeifeng/internnav/dualvln_mainline/data/traj_data/r2r/17DRP5sb8fy/meta/tasks.jsonl
mkdir -p "${result_root}"
exec 9>"${result_root}/task_paraphrase_autorun.lock"
flock -n 9 || { echo "another task paraphrase experiment is active" >&2; exit 3; }

guard_gpu() {
    mapfile -t gpu_free < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    [[ ${#gpu_free[@]} -eq 4 ]] || return 1
    for gpu_index in 0 1 2 3; do
        free_mib=${gpu_free[${gpu_index}]// /}
        if [[ "${free_mib}" -lt 24000 ]]; then
            echo "GPU ${gpu_index} has ${free_mib} MiB free, below protected threshold" >&2
            return 1
        fi
    done
}

python=/home/yifeifeng/miniconda3/envs/internvla/bin/python
[[ -x "${python}" ]] && "${python}" -c 'import torch, pytest' >/dev/null 2>&1 || {
    echo "InternVLA Python is unavailable" >&2
    exit 3
}
commit=$(git rev-parse --short=12 HEAD)
stage_id="task_paraphrase_$(date -u +%Y%m%dT%H%M%SZ)_${commit}"
stage_dir=${result_root}/${stage_id}
mkdir -p "${stage_dir}/alignment"

finalize() {
    local exit_code=$?
    trap - EXIT
    if [[ "${exit_code}" -ne 0 && ! -f "${stage_dir}/metrics.json" ]]; then
        "${python}" scripts/dualvln_mainline/stage_report.py \
            --stage "R1P 官方路径复述监督" --run-id "${stage_id}" --commit "${commit}" \
            --exit-code "${exit_code}" --junit "${stage_dir}/junit.xml" --output-dir "${stage_dir}" \
            --note "实验在聚合分析前停止；已完成子阶段的原始日志和指标仍保留在本目录。"
    fi
    ln -sfn "${stage_id}" "${result_root}/latest_task_paraphrase"
    ln -sfn "${stage_id}" "${result_root}/latest"
    exit "${exit_code}"
}
trap finalize EXIT

set +e
"${python}" scripts/dualvln_mainline/build_r2r_task_alignment.py \
    --raw "${raw}" --tasks "${tasks}" --scene 17DRP5sb8fy --output "${alignment}" \
    2>&1 | tee "${stage_dir}/alignment/build.log"
alignment_status=${PIPESTATUS[0]}
set -e
[[ "${alignment_status}" -eq 0 ]] || exit "${alignment_status}"
cp "${alignment}" "${stage_dir}/alignment/manifest.json"
sha256sum "${raw}" "${tasks}" "${alignment}" >"${stage_dir}/alignment/SHA256SUMS"

"${python}" -m pytest -q \
    --junitxml="${stage_dir}/junit.xml" \
    tests/unit_test/test_task_alignment.py \
    tests/unit_test/test_evidence_memory.py \
    tests/unit_test/test_evidence_conditioning.py \
    tests/unit_test/test_internvla_evidence_forward.py \
    tests/unit_test/test_evidence_inference.py \
    tests/unit_test/test_trainable_policy.py \
    2>&1 | tee "${stage_dir}/protocol_test.log" || exit ${PIPESTATUS[0]}

run_train() {
    local case_name=$1
    local steps=$2
    shift 2
    local output_dir=${stage_dir}/${case_name}/train
    guard_gpu || return 3
    mkdir -p "${output_dir}"
    set +e
    CUDA_VISIBLE_DEVICES=0,1,2,3 "${python}" scripts/dualvln_mainline/real_overfit.py \
        --output-dir "${output_dir}" --run-id "${stage_id}_${case_name}" --commit "${commit}" \
        --samples 8 --steps "${steps}" --sharded --gradient-bypass --bypass-mode cross_attention \
        --bypass-scale 0.5 --experiment M2 --seed 23 --task-state-lr-scale 0.1 \
        --instruction-only-task-state --task-alignment-path "${alignment}" "$@" \
        2>&1 | tee "${stage_dir}/${case_name}/train.log"
    local status=${PIPESTATUS[0]}
    set -e
    return "${status}"
}

run_probe() {
    local case_name=$1
    guard_gpu || return 3
    mkdir -p "${stage_dir}/${case_name}/probe"
    set +e
    CUDA_VISIBLE_DEVICES=0,1,2,3 "${python}" scripts/dualvln_mainline/task_state_probe.py \
        --output-dir "${stage_dir}/${case_name}/probe" \
        --adapter "${stage_dir}/${case_name}/train/adapter_state.pt" \
        --samples 32 --seed 23 --commit "${commit}" --instruction-only-task-state \
        --task-alignment-path "${alignment}" 2>&1 | tee "${stage_dir}/${case_name}/probe.log"
    local status=${PIPESTATUS[0]}
    set -e
    return "${status}"
}

# A single update first validates the real manifest, collation, auxiliary loss and gradients.
run_train smoke 1 --task-contrastive-weight 1.0 --task-contrastive-margin 0.2 --smoke-only || exit $?

overall_status=0
run_train R0P 80 || overall_status=1
[[ "${overall_status}" -eq 0 ]] && run_probe R0P || overall_status=1
run_train R1P 80 --task-contrastive-weight 1.0 --task-contrastive-margin 0.2 || overall_status=1
[[ "${overall_status}" -eq 0 ]] && run_probe R1P || overall_status=1
[[ "${overall_status}" -eq 0 ]] || exit "${overall_status}"

set +e
"${python}" scripts/dualvln_mainline/analyze_task_paraphrase.py \
    --stage-dir "${stage_dir}" --commit "${commit}" 2>&1 | tee "${stage_dir}/analysis.log"
analysis_status=${PIPESTATUS[0]}
set -e
sha256sum \
    scripts/dualvln_mainline/build_r2r_task_alignment.py \
    scripts/dualvln_mainline/task_paraphrase_autorun.sh \
    scripts/dualvln_mainline/analyze_task_paraphrase.py \
    scripts/dualvln_mainline/real_overfit.py \
    scripts/dualvln_mainline/task_state_probe.py \
    internnav/dataset/internvla_n1_lerobot_dataset.py \
    internnav/model/basemodel/internvla_n1/internvla_n1.py \
    >"${stage_dir}/SOURCE_SHA256SUMS"
ln -sfn "${stage_id}" "${result_root}/latest_task_paraphrase"
ln -sfn "${stage_id}" "${result_root}/latest"
echo "R1P summary: ${stage_dir}/summary.md"
exit "${analysis_status}"
