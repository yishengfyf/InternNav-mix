#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 0 ]]; then
    echo "usage: $0" >&2
    exit 2
fi

expected_root=/data/usr_data/yifeifeng/internnav/worktrees/dualvln-spatial-memory-v1
repo_root=$(git rev-parse --show-toplevel)
if [[ "${repo_root}" != "${expected_root}" ]]; then
    echo "refusing to run outside ${expected_root}: ${repo_root}" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "refusing to run from a dirty server worktree" >&2
    exit 2
fi

result_root=/data/usr_data/yifeifeng/internnav/dualvln_mainline/results/codex_latest_return
mkdir -p "${result_root}"
exec 9>"${result_root}/n1_n2_autorun.lock"
if ! flock -n 9; then
    echo "another N1/N2 dispatcher owns ${result_root}/n1_n2_autorun.lock" >&2
    exit 3
fi

mapfile -t gpu_free < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
if [[ ${#gpu_free[@]} -ne 4 ]]; then
    echo "expected exactly four GPUs, found ${#gpu_free[@]}" >&2
    exit 3
fi
for gpu_index in 0 1 2 3; do
    free_mib=${gpu_free[${gpu_index}]// /}
    if [[ "${free_mib}" -lt 24000 ]]; then
        echo "GPU ${gpu_index} has ${free_mib} MiB free, below the protected 24000 MiB threshold" >&2
        exit 3
    fi
done

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
if [[ -z "${python_bin}" ]]; then
    echo "no candidate InternVLA Python provides both torch and pytest" >&2
    exit 3
fi

commit=$(git rev-parse --short=12 HEAD)
matrix_id="n2_real32_$(date -u +%Y%m%dT%H%M%SZ)_${commit}"
matrix_dir="${result_root}/${matrix_id}"
mkdir -p "${matrix_dir}"

set +e
"${python_bin}" -m pytest -q \
    tests/unit_test/test_evidence_memory.py \
    tests/unit_test/test_evidence_conditioning.py \
    tests/unit_test/test_internvla_evidence_forward.py \
    tests/unit_test/test_trainable_policy.py \
    2>&1 | tee "${matrix_dir}/protocol_test.log"
protocol_status=${PIPESTATUS[0]}
set -e
if [[ "${protocol_status}" -ne 0 ]]; then
    echo "N1/N2 protocol tests failed; no GPU experiment was started" >&2
    exit "${protocol_status}"
fi

overall_status=0
for experiment in B0 B1 B2 M1 M2; do
    run_dir="${matrix_dir}/${experiment}"
    mkdir -p "${run_dir}"
    set +e
    PYTHONPATH=/data/usr_data/yifeifeng/internnav/dualvln_mainline/python_deps \
        "${python_bin}" scripts/dualvln_mainline/real_overfit.py \
        --run-id "${matrix_id}_${experiment}" \
        --commit "${commit}" \
        --output-dir "${run_dir}" \
        --samples 32 \
        --steps 200 \
        --sharded \
        --gradient-bypass \
        --bypass-mode cross_attention \
        --bypass-scale 0.5 \
        --experiment "${experiment}" \
        2>&1 | tee "${run_dir}/overfit.log"
    run_status=${PIPESTATUS[0]}
    set -e
    sha256sum \
        scripts/dualvln_mainline/real_overfit.py \
        scripts/dualvln_mainline/compare_n2.py \
        internnav/dataset/internvla_n1_lerobot_dataset.py \
        internnav/model/basemodel/internvla_n1/internvla_n1.py \
        internnav/model/basemodel/internvla_n1/evidence_memory.py \
        internnav/model/basemodel/internvla_n1/evidence_conditioning.py \
        internnav/model/basemodel/internvla_n1/trainable.py \
        >"${run_dir}/SOURCE_SHA256SUMS"
    if [[ "${run_status}" -ne 0 ]]; then
        overall_status=1
    fi
done

"${python_bin}" scripts/dualvln_mainline/compare_n2.py --stage-dir "${matrix_dir}"
sha256sum \
    scripts/dualvln_mainline/n1_n2_autorun.sh \
    scripts/dualvln_mainline/compare_n2.py \
    >"${matrix_dir}/SOURCE_SHA256SUMS"
ln -sfn "${matrix_id}" "${result_root}/latest_n2"

echo "N1/N2 matrix: ${matrix_dir}"
echo "Result summary: ${matrix_dir}/summary.md"
exit "${overall_status}"
