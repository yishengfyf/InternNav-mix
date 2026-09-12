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
reference_dir=${result_root}/n2_real32_20260911T105430Z_8fd49d9cb484
if [[ ! -f "${reference_dir}/metrics.json" ]]; then
    echo "missing fixed seed-23 N2 reference: ${reference_dir}" >&2
    exit 3
fi
mkdir -p "${result_root}"
exec 9>"${result_root}/n2_robustness_autorun.lock"
if ! flock -n 9; then
    echo "another N2 robustness dispatcher is active" >&2
    exit 3
fi

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

python_bin=
for candidate in \
    /home/yifeifeng/anaconda3/envs/internvla/bin/python \
    /home/yifeifeng/miniconda3/envs/internvla/bin/python \
    /data/usr_data/yifeifeng/miniconda3/envs/internvla/bin/python; do
    if [[ -x "${candidate}" ]] && "${candidate}" -c 'import torch, pytest' >/dev/null 2>&1; then
        python_bin=${candidate}
        break
    fi
done
[[ -n "${python_bin}" ]] || { echo "no usable InternVLA Python" >&2; exit 3; }

commit=$(git rev-parse --short=12 HEAD)
stage_id="n2_robustness_$(date -u +%Y%m%dT%H%M%SZ)_${commit}"
stage_dir="${result_root}/${stage_id}"
mkdir -p "${stage_dir}"

set +e
"${python_bin}" -m pytest -q \
    tests/unit_test/test_evidence_memory.py \
    tests/unit_test/test_evidence_conditioning.py \
    tests/unit_test/test_internvla_evidence_forward.py \
    tests/unit_test/test_trainable_policy.py \
    2>&1 | tee "${stage_dir}/protocol_test.log"
protocol_status=${PIPESTATUS[0]}
set -e
[[ "${protocol_status}" -eq 0 ]] || exit "${protocol_status}"

run_case() {
    local output_dir=$1
    local experiment=$2
    local seed=$3
    shift 3
    guard_gpu || return 3
    mkdir -p "${output_dir}"
    set +e
    PYTHONPATH=/data/usr_data/yifeifeng/internnav/dualvln_mainline/python_deps \
        "${python_bin}" scripts/dualvln_mainline/real_overfit.py \
        --run-id "${stage_id}_$(basename "${output_dir}")" \
        --commit "${commit}" \
        --output-dir "${output_dir}" \
        --samples 32 --steps 200 --sharded --gradient-bypass \
        --bypass-mode cross_attention --bypass-scale 0.5 \
        --experiment "${experiment}" --seed "${seed}" "$@" \
        2>&1 | tee "${output_dir}/overfit.log"
    local status=${PIPESTATUS[0]}
    set -e
    sha256sum \
        scripts/dualvln_mainline/real_overfit.py \
        scripts/dualvln_mainline/analyze_n2_robustness.py \
        internnav/model/basemodel/internvla_n1/internvla_n1.py \
        internnav/model/basemodel/internvla_n1/evidence_memory.py \
        internnav/model/basemodel/internvla_n1/evidence_conditioning.py \
        >"${output_dir}/SOURCE_SHA256SUMS"
    return "${status}"
}

overall_status=0
for seed in 47 71; do
    for experiment in B2 M1 M2; do
        run_case "${stage_dir}/original/seed${seed}/${experiment}" "${experiment}" "${seed}" || overall_status=1
    done
done
[[ "${overall_status}" -eq 0 ]] || exit "${overall_status}"

for variant in scale01 unit_norm lr01; do
    extra_args=()
    case "${variant}" in
        scale01) extra_args=(--task-state-scale 0.1) ;;
        unit_norm) extra_args=(--normalize-task-state) ;;
        lr01) extra_args=(--task-state-lr-scale 0.1) ;;
    esac
    run_case "${stage_dir}/screen/${variant}" M2 23 "${extra_args[@]}" || overall_status=1
done
[[ "${overall_status}" -eq 0 ]] || exit "${overall_status}"

"${python_bin}" scripts/dualvln_mainline/analyze_n2_robustness.py \
    --reference-dir "${reference_dir}" --stage-dir "${stage_dir}" --select-only
selected=$(<"${stage_dir}/selected_variant.txt")
if [[ "${selected}" != original ]]; then
    for seed in 47 71; do
        extra_args=()
        case "${selected}" in
            scale01) extra_args=(--task-state-scale 0.1) ;;
            unit_norm) extra_args=(--normalize-task-state) ;;
            lr01) extra_args=(--task-state-lr-scale 0.1) ;;
            *) echo "unknown selected variant: ${selected}" >&2; exit 4 ;;
        esac
        run_case "${stage_dir}/validate/${selected}/seed${seed}" M2 "${seed}" "${extra_args[@]}" || overall_status=1
    done
fi

"${python_bin}" scripts/dualvln_mainline/analyze_n2_robustness.py \
    --reference-dir "${reference_dir}" --stage-dir "${stage_dir}"
analysis_status=$?
sha256sum \
    scripts/dualvln_mainline/n2_robustness_autorun.sh \
    scripts/dualvln_mainline/analyze_n2_robustness.py \
    >"${stage_dir}/SOURCE_SHA256SUMS"
ln -sfn "${stage_id}" "${result_root}/latest_n2_robustness"
echo "N2 robustness summary: ${stage_dir}/summary.md"
exit "$((overall_status || analysis_status))"
