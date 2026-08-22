#!/usr/bin/env python3
"""Build conservative Stage25 event GT-lite and audit detector coverage."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from internnav.utils.stage25_event_gt_review import (
    annotate_review_candidates, evaluate_detector_against_gt_lite,
    intervals_overlap,
)


def audit_categories(
    annotated: Sequence[Mapping[str, Any]], detector: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    categories: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in annotated:
        annotation = row.get("annotation") or {}
        detected = any(intervals_overlap(row, event) for event in detector)
        state = annotation.get("state")
        if annotation.get("auto_status") != "objective_confirmed":
            category = "abstain"
        elif state == "true_trap":
            category = "true_trap_detected" if detected else "true_trap_missed"
        elif state == "wrong_way_progress":
            category = "wrong_way_overlap" if detected else "wrong_way_protected"
        else:
            category = "other"
        categories[category].append({**row, "audit_category": category})
    return dict(categories)


def severity(row: Mapping[str, Any]) -> tuple:
    audit = row.get("offline_action_audit") or {}
    return (
        -int(row.get("duration_steps") or 0),
        -float(audit.get("collision_delta") or 0.0),
        str(row.get("scene_id")), str(row.get("episode_id")),
        int(row.get("onset_step") or 0),
    )


def select_audit_rows(
    categories: Mapping[str, Sequence[Mapping[str, Any]]], per_category: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for category in (
        "true_trap_missed", "true_trap_detected", "wrong_way_overlap",
        "wrong_way_protected", "abstain",
    ):
        selected.extend(dict(row) for row in sorted(categories.get(category, []), key=severity)[:per_category])
    return selected


def contact_sheet(
    rows: Sequence[Mapping[str, Any]], evidence_root: Path, output: Path,
    *, columns: int = 3, panel_size: tuple = (420, 280), caption_height: int = 54,
) -> None:
    if not rows:
        return
    cell_w, cell_h = panel_size[0], panel_size[1] + caption_height
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, sheet_rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        source = evidence_root / str(row["evidence_image"]).replace("event_evidence/", "")
        if source.exists():
            image = Image.open(source).convert("RGB")
            image.thumbnail(panel_size)
            x = (index % columns) * cell_w
            y = (index // columns) * cell_h
            sheet.paste(image, (x + (cell_w - image.width) // 2, y))
        else:
            x = (index % columns) * cell_w
            y = (index // columns) * cell_h
        annotation = row.get("annotation") or {}
        caption = (
            f"{row.get('audit_category')} | {row.get('split')} | "
            f"{row.get('scene_id')}/{row.get('episode_id')} "
            f"[{row.get('onset_step')},{row.get('end_step')}]\n"
            f"{annotation.get('state')} / {annotation.get('type')}"
        )
        draw.text((x + 4, y + panel_size[1] + 3), caption, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--detector-candidates", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-per-category", type=int, default=8)
    args = parser.parse_args()
    review = json.loads(args.review_manifest.read_text(encoding="utf-8"))
    detector_variants = json.loads(args.detector_candidates.read_text(encoding="utf-8"))
    annotated = annotate_review_candidates(review)
    evaluations = {
        name: evaluate_detector_against_gt_lite(events, annotated)
        for name, events in detector_variants.items()
        if name in {"D0", "D1", "D2", "D2_executed_rotation"}
    }
    categories = audit_categories(annotated, detector_variants["D2"])
    audit_rows = select_audit_rows(categories, args.audit_per_category)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "task": "stage25_conservative_event_gt_lite",
        "label_source": "detector_independent_executed_future_trajectory",
        "outcome_is_event_gt": False,
        "future_used_by_detector": False,
        "gt_lite_scope": "local_stagnation_and_wrong_way_only",
        "precision_status": "not_computed_until_unadjudicated_detector_events_are_reviewed",
        "manifest_count": len(annotated),
        "annotation_status_counts": dict(Counter(
            (row.get("annotation") or {}).get("auto_status") for row in annotated
        )),
        "state_counts": dict(Counter(
            (row.get("annotation") or {}).get("state") for row in annotated
        )),
        "split_counts": dict(Counter(row.get("split") for row in annotated)),
        "d2_audit_category_counts": {
            name: len(rows) for name, rows in categories.items()
        },
        "detectors": evaluations,
    }
    (args.output / "stage25_event_gt_lite_manifest.json").write_text(
        json.dumps(annotated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output / "stage25_event_gt_lite_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output / "stage25_event_gt_lite_audit_manifest.json").write_text(
        json.dumps(audit_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    contact_sheet(
        audit_rows, args.evidence_root,
        args.output / "stage25_event_gt_lite_audit_contact_sheet.png",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
