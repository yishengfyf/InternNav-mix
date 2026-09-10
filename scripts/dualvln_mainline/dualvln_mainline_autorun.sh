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

for incomplete_dir in "${result_root}"/overfit_synthetic_*; do
    if [[ -d "${incomplete_dir}" && -f "${incomplete_dir}/overfit.log" && ! -f "${incomplete_dir}/metrics.json" ]]; then
        incomplete_id=$(basename "${incomplete_dir}")
        python3 scripts/dualvln_mainline/stage_report.py \
            --stage "P1 16 样本合成协议过拟合" \
            --run-id "${incomplete_id}" \
            --commit "unknown-before-report-repair" \
            --exit-code 1 \
            --junit "${incomplete_dir}/junit.xml" \
            --output-dir "${incomplete_dir}" \
            --note "运行在生成结构化指标前异常退出，原始错误见 overfit.log；后续运行已增加失败报告兜底。"
    fi
done
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
        tests/unit_test/test_evidence_history.py \
        tests/unit_test/test_evidence_conditioning.py \
        tests/unit_test/test_internvla_evidence_forward.py \
        tests/unit_test/test_evidence_sequence.py \
        tests/unit_test/test_encoder_lazy_imports.py \
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
    internnav/model/basemodel/internvla_n1/evidence_history.py \
    internnav/model/basemodel/internvla_n1/evidence_conditioning.py \
    internnav/model/basemodel/internvla_n1/evidence_sequence.py \
    internnav/model/basemodel/internvla_n1/internvla_n1.py \
    internnav/model/basemodel/internvla_n1/internvla_n1_arch.py \
    internnav/model/basemodel/internvla_n1/internvla_n1_policy.py \
    internnav/dataset/internvla_n1_lerobot_dataset.py \
    internnav/agent/internvla_n1_agent.py \
    internnav/trainer/internvla_n1_trainer.py \
    internnav/model/basemodel/internvla_n1/trainable.py \
    tests/unit_test/test_evidence_memory.py \
    tests/unit_test/test_evidence_history.py \
    tests/unit_test/test_evidence_conditioning.py \
    tests/unit_test/test_internvla_evidence_forward.py \
    tests/unit_test/test_evidence_sequence.py \
    tests/unit_test/test_encoder_lazy_imports.py \
    tests/unit_test/test_stage_report.py \
    tests/unit_test/test_trainable_policy.py \
    scripts/dualvln_mainline/stage_report.py \
    >"${run_dir}/SOURCE_SHA256SUMS"

ln -sfn "${run_id}" "${result_root}/latest"
if [[ "${test_exit}" -ne 0 ]]; then
    exit "${test_exit}"
fi

readiness_id="readiness_$(date -u +%Y%m%dT%H%M%SZ)_${commit}"
readiness_dir="${result_root}/${readiness_id}"
mkdir -p "${readiness_dir}"
"${python_bin}" scripts/dualvln_mainline/readiness_audit.py \
    --run-id "${readiness_id}" \
    --commit "$(git rev-parse HEAD)" \
    --output-dir "${readiness_dir}" \
    2>&1 | tee "${readiness_dir}/readiness.log"
sha256sum scripts/dualvln_mainline/readiness_audit.py >"${readiness_dir}/SOURCE_SHA256SUMS"
ln -sfn "${readiness_id}" "${result_root}/latest"

overfit_id="overfit_synthetic_$(date -u +%Y%m%dT%H%M%SZ)_${commit}"
overfit_dir="${result_root}/${overfit_id}"
mkdir -p "${overfit_dir}"
set +e
"${python_bin}" scripts/dualvln_mainline/synthetic_overfit.py \
    --run-id "${overfit_id}" \
    --commit "$(git rev-parse HEAD)" \
    --output-dir "${overfit_dir}" \
    2>&1 | tee "${overfit_dir}/overfit.log"
overfit_exit=${PIPESTATUS[0]}
set -e
if [[ ! -f "${overfit_dir}/metrics.json" ]]; then
    "${python_bin}" scripts/dualvln_mainline/stage_report.py \
        --stage "P1 16 样本合成协议过拟合" \
        --run-id "${overfit_id}" \
        --commit "$(git rev-parse HEAD)" \
        --exit-code "${overfit_exit}" \
        --junit "${overfit_dir}/junit.xml" \
        --output-dir "${overfit_dir}" \
        --note "运行在生成训练指标前异常退出，原始错误见 overfit.log。"
fi
sha256sum \
    scripts/dualvln_mainline/synthetic_overfit.py \
    internnav/model/basemodel/internvla_n1/evidence_memory.py \
    >"${overfit_dir}/SOURCE_SHA256SUMS"
ln -sfn "${overfit_id}" "${result_root}/latest"
if [[ "${overfit_exit}" -ne 0 ]]; then
    exit "${overfit_exit}"
fi

dependency_id="dependencies_$(date -u +%Y%m%dT%H%M%SZ)_${commit}"
dependency_dir="${result_root}/${dependency_id}"
mkdir -p "${dependency_dir}"
set +e
"${python_bin}" scripts/dualvln_mainline/bootstrap_dependencies.py \
    --run-id "${dependency_id}" \
    --commit "$(git rev-parse HEAD)" \
    --target /data/usr_data/yifeifeng/internnav/dualvln_mainline/python_deps \
    --output-dir "${dependency_dir}" \
    2>&1 | tee "${dependency_dir}/dependencies.log"
dependency_exit=${PIPESTATUS[0]}
set -e
sha256sum scripts/dualvln_mainline/bootstrap_dependencies.py >"${dependency_dir}/SOURCE_SHA256SUMS"
ln -sfn "${dependency_id}" "${result_root}/latest"
if [[ "${dependency_exit}" -ne 0 ]]; then
    exit "${dependency_exit}"
fi

data_id="data_bootstrap_$(date -u +%Y%m%dT%H%M%SZ)_${commit}"
data_dir="${result_root}/${data_id}"
mkdir -p "${data_dir}"
set +e
"${python_bin}" scripts/dualvln_mainline/bootstrap_minimal_data.py \
    --run-id "${data_id}" \
    --commit "$(git rev-parse HEAD)" \
    --data-root /data/usr_data/yifeifeng/internnav/dualvln_mainline/data/traj_data/r2r \
    --output-dir "${data_dir}" \
    2>&1 | tee "${data_dir}/bootstrap.log"
data_exit=${PIPESTATUS[0]}
set -e
sha256sum scripts/dualvln_mainline/bootstrap_minimal_data.py >"${data_dir}/SOURCE_SHA256SUMS"
ln -sfn "${data_id}" "${result_root}/latest"
if [[ "${data_exit}" -ne 0 ]]; then
    exit "${data_exit}"
fi

fixture_id="data_fixture_$(date -u +%Y%m%dT%H%M%SZ)_${commit}"
fixture_dir="${result_root}/${fixture_id}"
mkdir -p "${fixture_dir}"
set +e
PYTHONPATH=/data/usr_data/yifeifeng/internnav/dualvln_mainline/python_deps \
    "${python_bin}" scripts/dualvln_mainline/real_data_fixture.py \
    --run-id "${fixture_id}" \
    --commit "$(git rev-parse HEAD)" \
    --output-dir "${fixture_dir}" \
    2>&1 | tee "${fixture_dir}/fixture.log"
fixture_exit=${PIPESTATUS[0]}
set -e
sha256sum scripts/dualvln_mainline/real_data_fixture.py >"${fixture_dir}/SOURCE_SHA256SUMS"
ln -sfn "${fixture_id}" "${result_root}/latest"
if [[ "${fixture_exit}" -ne 0 ]]; then
    exit "${fixture_exit}"
fi

gpu3_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=3 | tr -d ' ')
if [[ "${gpu3_free}" -lt 26000 ]]; then
    echo "GPU 3 free memory ${gpu3_free} MiB is below the protected 26000 MiB threshold; real overfit deferred."
    exit 0
fi

real_overfit_id="overfit_real8_$(date -u +%Y%m%dT%H%M%SZ)_${commit}"
real_overfit_dir="${result_root}/${real_overfit_id}"
mkdir -p "${real_overfit_dir}"
set +e
CUDA_VISIBLE_DEVICES=3 \
PYTHONPATH=/data/usr_data/yifeifeng/internnav/dualvln_mainline/python_deps \
    "${python_bin}" scripts/dualvln_mainline/real_overfit.py \
    --run-id "${real_overfit_id}" \
    --commit "$(git rev-parse HEAD)" \
    --output-dir "${real_overfit_dir}" \
    2>&1 | tee "${real_overfit_dir}/overfit.log"
real_overfit_exit=${PIPESTATUS[0]}
set -e
sha256sum \
    scripts/dualvln_mainline/real_overfit.py \
    internnav/dataset/internvla_n1_lerobot_dataset.py \
    internnav/model/basemodel/internvla_n1/internvla_n1.py \
    internnav/model/basemodel/internvla_n1/evidence_memory.py \
    internnav/model/basemodel/internvla_n1/evidence_conditioning.py \
    internnav/model/basemodel/internvla_n1/trainable.py \
    internnav/model/basemodel/internvla_n1/internvla_n1_arch.py \
    internnav/model/encoder/__init__.py \
    >"${real_overfit_dir}/SOURCE_SHA256SUMS"
ln -sfn "${real_overfit_id}" "${result_root}/latest"
exit "${real_overfit_exit}"
