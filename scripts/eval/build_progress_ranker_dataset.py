"""Build a split-safe listwise dataset from Stage17 GT candidate labels.

This tool intentionally consumes only rows with ``label_status=ok``. The
current Stage17 angle labels are useful for a low-cost pipeline smoke test, but
they are not sufficient evidence for a final progress-value claim; every output
records its label source so later rollout-value data can replace them cleanly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import sys

TRAIN_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "train"
sys.path.insert(0, str(TRAIN_SCRIPT_DIR))

from progress_ranker_common import encode_candidate, feature_names


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{lineno}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{lineno}")
            yield value


def _split_for_episode(row: Dict[str, Any], val_ratio: float, seed: int) -> str:
    key = f"{row.get('scene_id')}|{row.get('episode_id')}|{seed}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / float(2**64)
    return "val" if bucket < val_ratio else "train"


def build_dataset(
    rows: Iterable[Dict[str, Any]],
    *,
    val_ratio: float,
    split_seed: int,
    require_angle_proxy_opt_in: bool,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    if not require_angle_proxy_opt_in:
        raise ValueError(
            "Refusing to build a training dataset from angle-only labels. "
            "Pass --allow-angle-proxy only for the Stage17 smoke experiment."
        )
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be in (0, 1)")

    outputs: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
    counts = Counter()
    feature_dim = len(feature_names())
    for row in rows:
        counts["input_rows"] += 1
        if row.get("label_status") != "ok":
            counts[f"drop_status={row.get('label_status')}"] += 1
            continue
        candidates = row.get("candidates") or []
        encoded = []
        labels = []
        candidate_ids = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            encoded.append(encode_candidate(candidate))
            labels.append(float(bool(candidate.get("gt_correct"))))
            candidate_ids.append(candidate.get("candidate_id"))
        if len(encoded) < 2:
            counts["drop_too_few_candidates"] += 1
            continue
        if not any(labels):
            counts["drop_no_positive"] += 1
            continue
        if any(len(item) != feature_dim for item in encoded):
            raise RuntimeError("Feature schema produced inconsistent dimensions")
        split = _split_for_episode(row, val_ratio, split_seed)
        outputs[split].append(
            {
                "scene_id": row.get("scene_id"),
                "episode_id": row.get("episode_id"),
                "step_id": row.get("step_id"),
                "label_source": "gt_direction_angle_proxy",
                "candidate_ids": candidate_ids,
                "features": encoded,
                "labels": labels,
            }
        )
        counts[f"kept_{split}"] += 1
        counts["positive_candidates"] += int(sum(labels))

    summary = {
        "label_source": "gt_direction_angle_proxy",
        "warning": "Angle labels are for Stage17 smoke training only; replace with rollout/progress labels before active claims.",
        "feature_names": feature_names(),
        "feature_dim": feature_dim,
        "split_seed": split_seed,
        "val_ratio": val_ratio,
        "counts": dict(counts),
    }
    return outputs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a listwise Stage17 progress-ranker dataset.")
    parser.add_argument("--labels", type=Path, required=True, help="Stage17 gt_candidate_labels.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--allow-angle-proxy", action="store_true")
    args = parser.parse_args()

    outputs, summary = build_dataset(
        _read_jsonl(args.labels),
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        require_angle_proxy_opt_in=args.allow_angle_proxy,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, items in outputs.items():
        path = args.output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
