#!/usr/bin/env python3
"""Run the read-only Stage29 HSGM-inspired route bridge audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from internnav.utils.stage29_hsgm_recovery_bridge import audit, write_bev_ppm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fov-deg", type=float, default=135.0)
    parser.add_argument("--max-visible-distance-m", type=float, default=5.0)
    parser.add_argument("--gt-manifest", type=Path, default=None)
    parser.add_argument("--bev-dir", type=Path, default=None)
    parser.add_argument("--max-bev-events", type=int, default=12)
    args = parser.parse_args()
    expected = None
    if args.gt_manifest is not None:
        payload = json.loads(args.gt_manifest.read_text(encoding="utf-8"))
        expected = payload if isinstance(payload, list) else payload.get("selected_detector_events", [])
    report = audit(
        args.root,
        fov_deg=args.fov_deg,
        max_visible_distance_m=args.max_visible_distance_m,
        expected_events=expected,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.bev_dir is not None:
        groups = {}
        for row in report["records"]:
            key = tuple((row.get("event_key") or {}).get(field) for field in ("scene_id", "episode_id", "step_id"))
            groups.setdefault(key, []).append(row)
        for index, (key, rows) in enumerate(sorted(groups.items())):
            if index >= max(0, int(args.max_bev_events)):
                break
            scene, episode, step = key
            write_bev_ppm(args.bev_dir / f"{scene}_{episode}_{step}_route_bridge.ppm", rows)
        report["bev_exported_count"] = min(len(groups), max(0, int(args.max_bev_events)))
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
