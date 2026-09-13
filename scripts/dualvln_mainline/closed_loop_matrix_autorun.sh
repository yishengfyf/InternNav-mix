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
reference_root=${result_root}/n2_real32_20260911T105430Z_8fd49d9cb484
robustness_root=${result_root}/n2_robustness_20260912T030417Z_8b5e7d6e1ca1
declare -A adapters=(
    [B1]="${reference_root}/B1/adapter_state.pt"
    [B2]="${reference_root}/B2/adapter_state.pt"
    [M1]="${reference_root}/M1/adapter_state.pt"
    [M2Z]="${robustness_root}/screen/lr01/adapter_state.pt"
    [M2]="${robustness_root}/screen/lr01/adapter_state.pt"
)
[[ "$(<"${robustness_root}/closed_loop_ready")" == 1 ]] || { echo "N2 robustness gate is not passed" >&2; exit 3; }
for variant in B1 B2 M1 M2Z M2; do
    [[ -f "${adapters[$variant]}" ]] || { echo "adapter is missing for ${variant}" >&2; exit 3; }
done

mkdir -p "${result_root}"
exec 9>"${result_root}/closed_loop_matrix_autorun.lock"
flock -n 9 || { echo "another closed-loop matrix dispatcher is active" >&2; exit 3; }

free_mib=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
[[ -n "${free_mib}" && "${free_mib}" -ge 24000 ]] || {
    echo "GPU 0 has ${free_mib:-unknown} MiB free, below the protected 24000 MiB threshold" >&2
    exit 3
}

habitat_python=/home/yifeifeng/miniconda3/envs/habitat/bin/python
internvla_python=/home/yifeifeng/miniconda3/envs/internvla/bin/python
internvla_site=/home/yifeifeng/miniconda3/envs/internvla/lib/python3.9/site-packages
python_deps=/data/usr_data/yifeifeng/internnav/dualvln_mainline/python_deps
commit=$(git rev-parse --short=12 HEAD)
stage_id="closed_loop_matrix_$(date -u +%Y%m%dT%H%M%SZ)_${commit}"
stage_dir=${result_root}/${stage_id}
mkdir -p "${stage_dir}"

"${internvla_python}" -m pytest -q \
    tests/unit_test/test_evidence_memory.py \
    tests/unit_test/test_evidence_conditioning.py \
    tests/unit_test/test_internvla_evidence_forward.py \
    tests/unit_test/test_evidence_inference.py \
    2>&1 | tee "${stage_dir}/protocol_test.log" || exit ${PIPESTATUS[0]}

run_case() {
    local variant=$1
    local adapter=${adapters[$variant]:-}
    local case_dir=${stage_dir}/${variant}
    mkdir -p "${case_dir}"
    free_mib=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    [[ "${free_mib}" -ge 24000 ]] || { echo "GPU 0 dropped below protected threshold" >&2; return 3; }
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="${internvla_site}:${python_deps}" \
    DUALVLN_CLOSED_LOOP_VARIANT="${variant}" \
    DUALVLN_CLOSED_LOOP_OUTPUT="${case_dir}" \
    DUALVLN_EVIDENCE_ADAPTER="${adapter}" \
    DUALVLN_MAX_EPISODES="${DUALVLN_MAX_EPISODES:-4}" \
    DUALVLN_MAX_STEPS="${DUALVLN_MAX_STEPS:-24}" \
    DUALVLN_CLOSED_LOOP_SEED=23 \
        "${habitat_python}" scripts/eval/eval.py \
        --config scripts/dualvln_mainline/configs/habitat_short_smoke_cfg.py \
        2>&1 | tee "${case_dir}/eval.log"
}

overall_status=0
for variant in B0 B1 B2 M1 M2Z M2; do
    run_case "${variant}" || overall_status=1
done
if [[ "${overall_status}" -eq 0 ]]; then
    "${internvla_python}" scripts/dualvln_mainline/analyze_closed_loop_matrix.py --stage-dir "${stage_dir}" \
        2>&1 | tee "${stage_dir}/analysis.log" || overall_status=1
fi

sha256sum \
    scripts/dualvln_mainline/closed_loop_matrix_autorun.sh \
    scripts/dualvln_mainline/analyze_closed_loop_matrix.py \
    scripts/dualvln_mainline/configs/habitat_short_smoke_cfg.py \
    internnav/habitat_extensions/vln/habitat_vln_evaluator.py \
    internnav/model/basemodel/internvla_n1/internvla_n1.py \
    >"${stage_dir}/SOURCE_SHA256SUMS"
ln -sfn "${stage_id}" "${result_root}/latest_closed_loop_matrix"
echo "Closed-loop matrix summary: ${stage_dir}/summary.md"
exit "${overall_status}"
