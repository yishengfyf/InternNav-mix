#!/usr/bin/env bash
# Stage22E: compare free/occupied evidence inside the same obstacle height band.
set -euo pipefail

export STAGE22_FIXED_ROUTE_LABEL=stage22e_height_aligned_evidence_shadow11
export STAGE22_FIXED_ROUTE_CONFIG=scripts/eval/configs/habitat_dual_system_vlmap_stage22e_height_aligned_evidence_shadow_cfg.py
export STAGE22_FIXED_ROUTE_AUDIT_NAME=stage22e_height_aligned_evidence_shadow_audit.json
export STAGE22_REQUIRE_EVIDENCE=1
export STAGE22_REQUIRE_HEIGHT_EVIDENCE=1
export STAGE22_FIXED_ROUTE_EVAL_PORT=${STAGE22_FIXED_ROUTE_EVAL_PORT:-2569}
export STAGE22_FIXED_ROUTE_MASTER_PORT=${STAGE22_FIXED_ROUTE_MASTER_PORT:-2570}

exec bash scripts/eval/bash/stage22c_fixed_route_pitch_occ_shadow_replay11.sh "$@"
