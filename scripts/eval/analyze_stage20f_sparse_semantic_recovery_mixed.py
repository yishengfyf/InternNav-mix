"""Analyze Stage20f mixed semantic-recovery calibration logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_stage20d_sparse_semantic_recovery_active import analyze as _analyze_stage20d


def analyze(paths):
    summary = _analyze_stage20d(paths)
    summary["task"] = "stage20f_sparse_semantic_recovery_mixed"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Run root, vlmap_safety_debug dir, or active events JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the summary JSON.",
    )
    args = parser.parse_args()

    summary = analyze(args.paths)
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
