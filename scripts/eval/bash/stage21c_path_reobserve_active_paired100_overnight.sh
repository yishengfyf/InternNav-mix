#!/usr/bin/env bash
# Stage21c paired100 convenience wrapper. The paired40 implementation is
# parameterized by STAGE21C_EXPECTED_EPISODES and keeps all audit logic shared.
set -euo pipefail

export STAGE21C_EXPECTED_EPISODES=100
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/stage21c_path_reobserve_active_paired40_overnight.sh" "$@"
