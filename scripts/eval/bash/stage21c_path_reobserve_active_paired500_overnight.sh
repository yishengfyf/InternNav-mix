#!/usr/bin/env bash
# Stage21c paired500 convenience wrapper. Active behavior and safety gates are
# identical to paired40/100; only the paired episode count changes.
set -euo pipefail

export STAGE21C_EXPECTED_EPISODES=500
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/stage21c_path_reobserve_active_paired40_overnight.sh" "$@"
