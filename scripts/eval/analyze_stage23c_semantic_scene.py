"""Audit semantic anchors against independent Habitat scene annotations."""

import argparse
import json
from pathlib import Path


def _key(row):
    return str(row.get("scene_id")), int(row.get("episode_id"))


def _load_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_unique(paths):
    result = {}
    for path in paths:
        for row in _load_jsonl(path):
            result[_key(row)] = row
    return result


def analyze(run_root: Path, manifest: Path, output: Path, require_annotations: bool):
    expected = {_key(row): row for row in json.loads(manifest.read_text(encoding="utf-8"))}
    progress = _load_unique(run_root.glob("vlmap_safety_debug/*run_*/progress.json"))
    if not progress:
        progress = _load_unique(run_root.glob("vlmap_safety_debug/rank*_run_*/progress.json"))
    errors = []
    episodes = []
    annotation_available_count = 0
    valid_count = 0
    for key, _manifest_row in expected.items():
        row = progress.get(key)
        if row is None:
            errors.append(f"missing_progress:{key}")
            continue
        audit = row.get("stage23c_semantic_scene_audit") or {}
        if not audit.get("enabled"):
            errors.append(f"audit_disabled:{key}")
        if not audit.get("valid") and audit.get("reason") != "semantic_scene_unavailable":
            errors.append(f"invalid_audit:{key}:{audit.get('reason')}")
        if audit.get("annotation_available"):
            annotation_available_count += 1
        if audit.get("valid"):
            valid_count += 1
        if require_annotations and not audit.get("annotation_available"):
            errors.append(f"annotations_unavailable:{key}")
        if int(row.get("s2_loop_strict_active_applied_count", 0) or 0):
            errors.append(f"strict_active_action_violation:{key}")
        if int(row.get("s2_loop_path_reobserve_applied_count", 0) or 0):
            errors.append(f"path_reobserve_action_violation:{key}")
        episodes.append(
            {
                "scene_id": key[0],
                "episode_id": key[1],
                "episode_eval_seed": row.get("episode_eval_seed"),
                "steps": row.get("steps"),
                "success": row.get("success"),
                "anchor_count": audit.get("anchor_count"),
                "matched_anchor_count": audit.get("matched_anchor_count"),
                "category_agreement_count": audit.get("category_agreement_count"),
                "category_agreement_rate": audit.get("category_agreement_rate"),
                "nearest_distance_m": audit.get("nearest_distance_m"),
                "annotation_available": audit.get("annotation_available"),
                "reason": audit.get("reason"),
                "audit": audit,
            }
        )
    report = {
        "audit_name": "stage23c_semantic_scene",
        "integrity_passed": not errors,
        "expected_episode_count": len(expected),
        "completed_episode_count": len(episodes),
        "valid_episode_count": valid_count,
        "annotation_available_episode_count": annotation_available_count,
        "annotation_required": bool(require_annotations),
        "episodes": episodes,
        "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-annotations", action="store_true")
    args = parser.parse_args()
    analyze(args.run_root, args.manifest, args.output, args.require_annotations)


if __name__ == "__main__":
    main()
