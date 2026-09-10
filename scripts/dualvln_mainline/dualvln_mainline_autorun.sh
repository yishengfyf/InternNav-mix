#!/usr/bin/env bash
set -euo pipefail

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

result_root=/data/usr_data/yifeifeng/internnav/dualvln_mainline/results/codex_latest_return
commit=$(git rev-parse --short=12 HEAD)
run_id="protocol_$(date -u +%Y%m%dT%H%M%SZ)_${commit}"
run_dir="${result_root}/${run_id}"
mkdir -p "${run_dir}"

python_candidates=(
    /home/yifeifeng/anaconda3/envs/internvla/bin/python
    /home/yifeifeng/miniconda3/envs/internvla/bin/python
    /data/usr_data/yifeifeng/miniconda3/envs/internvla/bin/python
)
python_bin=
for candidate in "${python_candidates[@]}"; do
    if [[ -x "${candidate}" ]] && "${candidate}" -c 'import pytest, torch' >/dev/null 2>&1; then
        python_bin=${candidate}
        break
    fi
done

if [[ -z "${python_bin}" ]]; then
    echo "no candidate InternVLA Python provides both torch and pytest" | tee "${run_dir}/error.txt" >&2
    python3 scripts/dualvln_mainline/stage_report.py \
        --stage "P1 协议检查" \
        --run-id "${run_id}" \
        --commit "$(git rev-parse HEAD)" \
        --exit-code 3 \
        --junit "${run_dir}/junit.xml" \
        --output-dir "${run_dir}" \
        --note "服务器没有找到同时提供 torch 与 pytest 的候选 InternVLA Python。"
    exit 3
fi

set +e
{
    echo "run_id=${run_id}"
    echo "commit=$(git rev-parse HEAD)"
    echo "python=${python_bin}"
    "${python_bin}" -c 'import pytest, torch; print(f"torch={torch.__version__}"); print(f"pytest={pytest.__version__}")'
    "${python_bin}" -m pytest -q \
        --junitxml="${run_dir}/junit.xml" \
        tests/unit_test/test_evidence_memory.py \
        tests/unit_test/test_evidence_sequence.py \
        tests/unit_test/test_stage_report.py \
        tests/unit_test/test_trainable_policy.py
} 2>&1 | tee "${run_dir}/protocol_test.log"
test_exit=${PIPESTATUS[0]}
set -e

"${python_bin}" scripts/dualvln_mainline/stage_report.py \
    --stage "P1 协议检查" \
    --run-id "${run_id}" \
    --commit "$(git rev-parse HEAD)" \
    --exit-code "${test_exit}" \
    --junit "${run_dir}/junit.xml" \
    --output-dir "${run_dir}"

sha256sum \
    internnav/model/basemodel/internvla_n1/evidence_memory.py \
    internnav/model/basemodel/internvla_n1/evidence_sequence.py \
    internnav/model/basemodel/internvla_n1/trainable.py \
    tests/unit_test/test_evidence_memory.py \
    tests/unit_test/test_evidence_sequence.py \
    tests/unit_test/test_stage_report.py \
    tests/unit_test/test_trainable_policy.py \
    scripts/dualvln_mainline/stage_report.py \
    >"${run_dir}/SOURCE_SHA256SUMS"

ln -sfn "${run_id}" "${result_root}/latest"
exit "${test_exit}"
