#!/usr/bin/env bash
set -uo pipefail

expected_root=/data/usr_data/yifeifeng/internnav/worktrees/dualvln-spatial-memory-v1
repo_root=$(git rev-parse --show-toplevel)
if [[ "${repo_root}" != "${expected_root}" ]] || [[ -n "$(git status --porcelain)" ]]; then
    echo "refusing non-isolated or dirty server worktree: ${repo_root}" >&2
    exit 2
fi

result_root=/data/usr_data/yifeifeng/internnav/dualvln_mainline/results/codex_latest_return
adapter=${result_root}/n2_robustness_20260912T030417Z_8b5e7d6e1ca1/screen/lr01/adapter_state.pt
[[ -f "${adapter}" ]] || { echo "selected adapter is missing" >&2; exit 3; }
mkdir -p "${result_root}"
exec 9>"${result_root}/task_state_probe_autorun.lock"
flock -n 9 || { echo "another task-state probe is active" >&2; exit 3; }

for gpu in 0 1 2 3; do
    free_mib=$(nvidia-smi --id="${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    [[ -n "${free_mib}" && "${free_mib}" -ge 24000 ]] || {
        echo "GPU ${gpu} has ${free_mib:-unknown} MiB free, below the protected 24000 MiB threshold" >&2
        exit 3
    }
done

python=/home/yifeifeng/miniconda3/envs/internvla/bin/python
commit=$(git rev-parse --short=12 HEAD)
stage_id="task_state_probe_$(date -u +%Y%m%dT%H%M%SZ)_${commit}"
stage_dir=${result_root}/${stage_id}
mkdir -p "${stage_dir}"
set +e
CUDA_VISIBLE_DEVICES=0,1,2,3 "${python}" scripts/dualvln_mainline/task_state_probe.py \
    --output-dir "${stage_dir}" \
    --adapter "${adapter}" \
    --samples 32 \
    --seed 23 \
    --commit "${commit}" \
    2>&1 | tee "${stage_dir}/probe.log"
status=${PIPESTATUS[0]}
set -e
sha256sum scripts/dualvln_mainline/task_state_probe.py scripts/dualvln_mainline/task_state_probe_autorun.sh \
    >"${stage_dir}/SOURCE_SHA256SUMS"
ln -sfn "${stage_id}" "${result_root}/latest_task_state_probe"
echo "Task-state probe summary: ${stage_dir}/summary.md"
exit "${status}"
