#!/usr/bin/env bash
set -euo pipefail

# Reproducible offline smoke for the fixed 30-frame Habitat/R2R sequence.
# FreeOcc remains a separate process/environment and never feeds S1/S2 here.
INTERNNAV_ROOT="${INTERNNAV_ROOT:-/data/usr_data/yifeifeng/internnav/worktrees/freeocc-integration}"
FREEOCC_ROOT="${FREEOCC_ROOT:-/data/usr_data/yifeifeng/internnav/third_party/FreeOcc}"
FREEOCC_ENV="${FREEOCC_ENV:-/data/usr_data/yifeifeng/internnav/envs/freeocc}"
INPUT_DIR="${INPUT_DIR:-/data/usr_data/yifeifeng/internnav/freeocc_smoke_data/dhjEzFoUFzH_6763_30f}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/usr_data/yifeifeng/internnav/freeocc_smoke_outputs}"
RUN_TAG="${RUN_TAG:-dhjEzFoUFzH_6763_30f_strict_audit1}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
OUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
PATCH_FILE="${INTERNNAV_ROOT}/patches/freeocc_habitat_audit_f84a0f0.patch"
EXPECTED_FREEOCC_COMMIT="f84a0f0ce28146b703d4d5bb5e061dc9a80be04e"

source /home/yifeifeng/miniconda3/etc/profile.d/conda.sh
conda activate "${FREEOCC_ENV}"
cd "${FREEOCC_ROOT}"

test "$(git rev-parse HEAD)" = "${EXPECTED_FREEOCC_COMMIT}"
test -f "${PATCH_FILE}"

if grep -Fq '[FreeOccAudit][filter]' src/depth_video.py; then
  echo "FreeOcc audit patch already applied"
else
  git apply --check "${PATCH_FILE}"
  git apply "${PATCH_FILE}"
  echo "FreeOcc audit patch applied"
fi

python -m py_compile src/datasets.py src/depth_video.py src/gaussian_mapping.py
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
  hydra.run.dir="${OUT_DIR}" 2>&1 | tee "${OUT_DIR}/console.log"

grep -q 'INFO: 30 images got!' "${OUT_DIR}/console.log"

python "${INTERNNAV_ROOT}/scripts/eval/analyze_freeocc_mapping_audit.py" \
  --run-dir "${OUT_DIR}" \
  --expected-input-frames 30

HASH_TMP="/tmp/${RUN_TAG}_SHA256SUMS.$$"
(cd "${OUT_DIR}" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "${HASH_TMP}"
mv "${HASH_TMP}" "${OUT_DIR}/SHA256SUMS"
echo "FREEOCC_RUN_DIR=${OUT_DIR}"
