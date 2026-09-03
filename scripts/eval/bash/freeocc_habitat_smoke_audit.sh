#!/usr/bin/env bash
set -euo pipefail

# Reproducible offline smoke for the fixed 30-frame Habitat/R2R sequence.
# FreeOcc remains a separate process/environment and never feeds S1/S2 here.
INTERNNAV_ROOT="${INTERNNAV_ROOT:-/data/usr_data/yifeifeng/internnav/worktrees/freeocc-integration}"
FREEOCC_ROOT="${FREEOCC_ROOT:-/data/usr_data/yifeifeng/internnav/third_party/FreeOcc}"
FREEOCC_ENV="${FREEOCC_ENV:-/data/usr_data/yifeifeng/internnav/envs/freeocc}"
INPUT_DIR="${INPUT_DIR:-/data/usr_data/yifeifeng/internnav/freeocc_smoke_data/dhjEzFoUFzH_6763_30f}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/usr_data/yifeifeng/internnav/freeocc_smoke_outputs}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
PATCH_FILE="${INTERNNAV_ROOT}/patches/freeocc_habitat_audit_f84a0f0.patch"
COUNTS_PATCH_FILE="${INTERNNAV_ROOT}/patches/freeocc_habitat_audit_counts_v2.patch"
EXPECTED_FREEOCC_COMMIT="f84a0f0ce28146b703d4d5bb5e061dc9a80be04e"
STRICT_TAG="dhjEzFoUFzH_6763_30f_strict_audit1"
MV1_TAG="dhjEzFoUFzH_6763_30f_mv1_stride2_audit2"
NOMV_TAG="dhjEzFoUFzH_6763_30f_nomv_stride4_audit2"
STRICT_DIR="${OUTPUT_ROOT}/${STRICT_TAG}"

source /home/yifeifeng/miniconda3/etc/profile.d/conda.sh
conda activate "${FREEOCC_ENV}"
cd "${FREEOCC_ROOT}"

test "$(git rev-parse HEAD)" = "${EXPECTED_FREEOCC_COMMIT}"
test -f "${PATCH_FILE}"
test -f "${COUNTS_PATCH_FILE}"

if grep -Fq '[FreeOccAudit][filter]' src/depth_video.py; then
  echo "FreeOcc audit patch already applied"
else
  git apply --check "${PATCH_FILE}"
  git apply "${PATCH_FILE}"
  echo "FreeOcc audit patch applied"
fi

if grep -Fq 'mv_support_ge_1' src/depth_video.py; then
  echo "FreeOcc multiview support-count patch already applied"
else
  git apply --check "${COUNTS_PATCH_FILE}"
  git apply "${COUNTS_PATCH_FILE}"
  echo "FreeOcc multiview support-count patch applied"
fi

python -m py_compile src/datasets.py src/depth_video.py src/gaussian_mapping.py

# The exact launcher command is approval-whitelisted.  Auto mode advances one
# bounded diagnostic profile per invocation and never overwrites a completed
# profile.  The relaxed profiles diagnose compatibility; they are not safety
# configurations and never feed S1/S2.
if [ ! -f "${STRICT_DIR}/analysis/freeocc_mapping_summary.json" ]; then
  RUN_TAG="${STRICT_TAG}"
  PROFILE_LABEL="strict_mv2_stride1"
  PROFILE_KEY="strict"
  PROFILE_OVERRIDES=(
    mapping.online_opt.filter.multiview=true
    mapping.online_opt.filter.mv_count_th=2
    mapping.frame_gaussians_stride=1
  )
elif [ ! -f "${OUTPUT_ROOT}/${MV1_TAG}/analysis/freeocc_mapping_summary.json" ]; then
  RUN_TAG="${MV1_TAG}"
  PROFILE_LABEL="diagnostic_mv1_stride2"
  PROFILE_KEY="mv1_stride2"
  PROFILE_OVERRIDES=(
    mapping.online_opt.filter.multiview=true
    mapping.online_opt.filter.mv_count_th=1
    mapping.frame_gaussians_stride=2
  )
elif [ ! -f "${OUTPUT_ROOT}/${NOMV_TAG}/analysis/freeocc_mapping_summary.json" ]; then
  RUN_TAG="${NOMV_TAG}"
  PROFILE_LABEL="diagnostic_no_multiview_stride4"
  PROFILE_KEY="nomv_stride4"
  PROFILE_OVERRIDES=(
    mapping.online_opt.filter.multiview=false
    mapping.frame_gaussians_stride=4
  )
else
  echo "All bounded FreeOcc Habitat diagnostic profiles are already complete"
  RUN_TAG=""
  PROFILE_LABEL=""
  PROFILE_KEY=""
  PROFILE_OVERRIDES=()
fi

if [ -n "${RUN_TAG}" ]; then
  OUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
  mkdir -p "${OUT_DIR}"

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
  PYTHONUNBUFFERED=1 \
  HYDRA_FULL_ERROR=1 \
  PYTHONPATH=thirdparty/Trident:src/gs2occ/localagg_prob:. \
  python run.py \
    mode=mono \
    use_gt_poses=False \
    run_visualization=False \
    run_mapping_gui=False \
    run_loop_detection=False \
    mapping.enable_occ_eval=False \
    +t_start=0 \
    +t_stop=30 \
    data.input_folder="${INPUT_DIR}" \
    data.cam.H=480 \
    data.cam.W=640 \
    data.cam.H_out=256 \
    data.cam.W_out=320 \
    data.cam.fx=388.19104 \
    data.cam.fy=388.19104 \
    data.cam.cx=319.5 \
    data.cam.cy=239.5 \
    data.png_depth_scale=1000.0 \
    "${PROFILE_OVERRIDES[@]}" \
    hydra.run.dir="${OUT_DIR}" 2>&1 | tee "${OUT_DIR}/console.log"

  grep -q 'INFO: 30 images got!' "${OUT_DIR}/console.log"

  python "${INTERNNAV_ROOT}/scripts/eval/analyze_freeocc_mapping_audit.py" \
    --run-dir "${OUT_DIR}" \
    --expected-input-frames 30 \
    --rgb-dir "${INPUT_DIR}" \
    --profile-label "${PROFILE_LABEL}"

  HASH_TMP="/tmp/${RUN_TAG}_SHA256SUMS.$$"
  (cd "${OUT_DIR}" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum) > "${HASH_TMP}"
  mv "${HASH_TMP}" "${OUT_DIR}/SHA256SUMS"

  # Put the compact, user-facing bundle under the already approved strict
  # return root.  Full diagnostic runs remain isolated beside it on server.
  RETURN_DIR="${STRICT_DIR}/diagnostics/${PROFILE_KEY}"
  mkdir -p "${RETURN_DIR}/analysis" "${RETURN_DIR}/audit" "${RETURN_DIR}/mesh"
  cp "${OUT_DIR}/analysis/freeocc_mapping_summary.json" "${RETURN_DIR}/analysis/"
  cp "${OUT_DIR}/analysis/freeocc_filter_trajectory_audit.png" "${RETURN_DIR}/analysis/"
  cp "${OUT_DIR}/analysis/freeocc_rgb_semantic_gaussians.png" "${RETURN_DIR}/analysis/"
  cp "${OUT_DIR}/audit/freeocc_mapping_audit.json" "${RETURN_DIR}/audit/"
  cp "${OUT_DIR}/audit/trajectories.npz" "${RETURN_DIR}/audit/"
  cp "${OUT_DIR}/mesh/final_mono.ply" "${RETURN_DIR}/mesh/"
  cp "${OUT_DIR}/mesh/final_mono_raw.ply" "${RETURN_DIR}/mesh/"
  cp "${OUT_DIR}/config.yaml" "${RETURN_DIR}/"
  cp "${OUT_DIR}/console.log" "${RETURN_DIR}/"
  echo "FREEOCC_RUN_DIR=${OUT_DIR}"
fi

# Re-render every completed profile with corrected Habitat ground axes, build
# observed-surface GT from depth+pose, and refresh its compact return bundle.
AUDIT_RUN_DIRS=(
  "${STRICT_DIR}"
  "${OUTPUT_ROOT}/${MV1_TAG}"
  "${OUTPUT_ROOT}/${NOMV_TAG}"
)
AUDIT_PROFILE_LABELS=(
  strict_mv2_stride1
  diagnostic_mv1_stride2
  diagnostic_no_multiview_stride4
)
AUDIT_PROFILE_KEYS=(strict mv1_stride2 nomv_stride4)
COMPARE_ARGS=()

for i in "${!AUDIT_RUN_DIRS[@]}"; do
  AUDIT_DIR="${AUDIT_RUN_DIRS[$i]}"
  AUDIT_LABEL="${AUDIT_PROFILE_LABELS[$i]}"
  AUDIT_KEY="${AUDIT_PROFILE_KEYS[$i]}"
  if [ ! -f "${AUDIT_DIR}/audit/freeocc_mapping_audit.json" ]; then
    continue
  fi

  python "${INTERNNAV_ROOT}/scripts/eval/analyze_freeocc_mapping_audit.py" \
    --run-dir "${AUDIT_DIR}" \
    --expected-input-frames 30 \
    --rgb-dir "${INPUT_DIR}" \
    --profile-label "${AUDIT_LABEL}"

  PYTHONPATH="${INTERNNAV_ROOT}" python "${INTERNNAV_ROOT}/scripts/eval/evaluate_freeocc_habitat_gt.py" \
    --input-dir "${INPUT_DIR}" \
    --pred-ply "${AUDIT_DIR}/mesh/final_mono.ply" \
    --trajectory-npz "${AUDIT_DIR}/audit/trajectories.npz" \
    --out-dir "${AUDIT_DIR}/analysis" \
    --profile-label "${AUDIT_LABEL}"

  COMPARE_ARGS+=(--run "${AUDIT_KEY}=${AUDIT_DIR}")
  if [ "${AUDIT_KEY}" != strict ]; then
    RETURN_DIR="${STRICT_DIR}/diagnostics/${AUDIT_KEY}"
    mkdir -p "${RETURN_DIR}/analysis" "${RETURN_DIR}/audit" "${RETURN_DIR}/mesh"
    cp "${AUDIT_DIR}/analysis/freeocc_mapping_summary.json" "${RETURN_DIR}/analysis/"
    cp "${AUDIT_DIR}/analysis/freeocc_filter_trajectory_audit.png" "${RETURN_DIR}/analysis/"
    cp "${AUDIT_DIR}/analysis/freeocc_rgb_semantic_gaussians.png" "${RETURN_DIR}/analysis/"
    cp "${AUDIT_DIR}/analysis/habitat_gt_occ_metrics.json" "${RETURN_DIR}/analysis/"
    cp "${AUDIT_DIR}/analysis/habitat_observed_occ.npz" "${RETURN_DIR}/analysis/"
    cp "${AUDIT_DIR}/analysis/freeocc_rgb_pred_gt_occ.png" "${RETURN_DIR}/analysis/"
  fi

  HASH_TMP="/tmp/${AUDIT_KEY}_SHA256SUMS.$$"
  (cd "${AUDIT_DIR}" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum) > "${HASH_TMP}"
  mv "${HASH_TMP}" "${AUDIT_DIR}/SHA256SUMS"
done

if [ "${#COMPARE_ARGS[@]}" -gt 0 ]; then
  python "${INTERNNAV_ROOT}/scripts/eval/compare_freeocc_habitat_profiles.py" \
    "${COMPARE_ARGS[@]}" \
    --out-dir "${STRICT_DIR}/analysis"
fi

if [ -d "${STRICT_DIR}" ]; then
  HASH_TMP="/tmp/${STRICT_TAG}_SHA256SUMS.$$"
  (cd "${STRICT_DIR}" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum) > "${HASH_TMP}"
  mv "${HASH_TMP}" "${STRICT_DIR}/SHA256SUMS"
  echo "FREEOCC_RETURN_DIR=${STRICT_DIR}"
fi
