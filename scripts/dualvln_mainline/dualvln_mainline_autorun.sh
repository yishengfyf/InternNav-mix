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
    exit 3
fi

{
    echo "run_id=${run_id}"
    echo "commit=$(git rev-parse HEAD)"
    echo "python=${python_bin}"
    "${python_bin}" -c 'import pytest, torch; print(f"torch={torch.__version__}"); print(f"pytest={pytest.__version__}")'
    "${python_bin}" -m pytest -q \
        tests/unit_test/test_evidence_memory.py \
        tests/unit_test/test_trainable_policy.py
} 2>&1 | tee "${run_dir}/protocol_test.log"

sha256sum \
    internnav/model/basemodel/internvla_n1/evidence_memory.py \
    internnav/model/basemodel/internvla_n1/trainable.py \
    tests/unit_test/test_evidence_memory.py \
    tests/unit_test/test_trainable_policy.py \
    >"${run_dir}/SOURCE_SHA256SUMS"

ln -sfn "${run_id}" "${result_root}/latest"
