#!/usr/bin/env python3
"""Evaluate frozen InternVLA relevance features with episode-grouped folds."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def feature_matrix(data, variant):
    current, history = data["visual"][:, 0], data["visual"][:, 1]
    if variant == "pose_age":
        return data["metadata"]
    content = np.concatenate(
        (history - current, history * current, history * data["text"]), axis=1
    )
    if variant == "rgb_text":
        return content
    return np.concatenate((content, data["metadata"]), axis=1)


def evaluate(data, variant, seed, splits=4):
    targets, groups = data["targets"], data["episode_ids"]
    annotation_ids = data["annotation_ids"]
    features = feature_matrix(data, variant)
    fold_rows, hits, pairwise = [], [], []
    splitter = GroupKFold(n_splits=min(splits, len(set(groups))))
    for fold, (train, test) in enumerate(splitter.split(features, targets, groups)):
        components = max(1, min(16, len(train) - 1, features.shape[1]))
        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=components, whiten=False, random_state=seed),
            LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=2000, random_state=seed
            ),
        )
        model.fit(features[train], targets[train])
        scores = model.predict_proba(features[test])[:, 1]
        fold_hits, fold_pairwise = [], []
        for annotation_id in sorted(set(annotation_ids[test])):
            mask = annotation_ids[test] == annotation_id
            truth, sample_scores = targets[test][mask], scores[mask]
            if not truth.any() or truth.all():
                continue
            fold_hits.append(int(truth[np.argmax(sample_scores)] == 1))
            positive, negative = sample_scores[truth == 1], sample_scores[truth == 0]
            fold_pairwise.extend(float(pos > neg) for pos in positive for neg in negative)
        hits.extend(fold_hits)
        pairwise.extend(fold_pairwise)
        fold_rows.append(
            {
                "fold": fold,
                "train_episodes": len(set(groups[train])),
                "test_episodes": len(set(groups[test])),
                "cards": len(fold_hits),
                "top1_hit": float(np.mean(fold_hits)) if fold_hits else None,
                "pairwise_accuracy": float(np.mean(fold_pairwise)) if fold_pairwise else None,
            }
        )
    return {
        "variant": variant,
        "seed": seed,
        "top1_hit": float(np.mean(hits)),
        "pairwise_accuracy": float(np.mean(pairwise)),
        "evaluated_cards": len(hits),
        "folds": fold_rows,
    }


def write_svg(summary, path):
    variants = [("随机", summary["random_top1"])] + [
        (item["variant"], item["top1_hit_mean"]) for item in summary["variants"]
    ]
    rows = []
    for index, (label, value) in enumerate(variants):
        y = 65 + index * 42
        rows.append(
            f'<text x="20" y="{y + 17}">{label}</text>'
            f'<rect x="130" y="{y}" width="390" height="22" fill="#edf2f7"/>'
            f'<rect x="130" y="{y}" width="{390 * value:.1f}" height="22" fill="#3878c7"/>'
            f'<text x="530" y="{y + 17}">{value:.1%}</text>'
        )
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="620" height="250">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="20" y="31" font-size="20" font-weight="700">Frozen feature relevance probe</text>'
        + "".join(rows) + "</svg>\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="运行冻结 InternVLA 特征 relevance probe")
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[23, 47, 71])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = np.load(args.features)
    data = {key: loaded[key] for key in loaded.files}
    positive_cards = np.unique(data["annotation_ids"][data["targets"] == 1])
    useful = np.isin(data["annotation_ids"], positive_cards)
    data = {key: value[useful] for key, value in data.items()}
    card_ids = np.unique(data["annotation_ids"])
    random_top1 = float(
        np.mean(
            [data["targets"][data["annotation_ids"] == item].mean() for item in card_ids]
        )
    )
    runs = [
        evaluate(data, variant, seed)
        for variant in ("pose_age", "rgb_text", "rgb_text_pose")
        for seed in args.seeds
    ]
    variants = []
    for variant in ("pose_age", "rgb_text", "rgb_text_pose"):
        selected = [run for run in runs if run["variant"] == variant]
        variants.append(
            {
                "variant": variant,
                "top1_hit_mean": float(np.mean([run["top1_hit"] for run in selected])),
                "top1_hit_std": float(np.std([run["top1_hit"] for run in selected])),
                "pairwise_accuracy_mean": float(
                    np.mean([run["pairwise_accuracy"] for run in selected])
                ),
            }
        )
    best = max(variants, key=lambda item: item["top1_hit_mean"])
    passed = (
        best["top1_hit_mean"] > random_top1 + 0.05
        and best["pairwise_accuracy_mean"] > 0.55
    )
    summary = {
        "schema_version": 1,
        "status": "passed" if passed else "not_passed",
        "scope": "single scene; episode-grouped 4-fold; diagnostic only",
        "candidate_examples": len(data["targets"]),
        "cards": len(card_ids),
        "episodes": len(set(data["episode_ids"])),
        "positive_examples": int(data["targets"].sum()),
        "random_top1": random_top1,
        "variants": variants,
        "runs": runs,
        "gate": "best top1 > random + 0.05 and pairwise > 0.55",
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_svg(summary, args.output_dir / "metrics.svg")
    status = "通过" if passed else "未通过"
    report = f"""# 冻结 InternVLA 特征 relevance probe

- 状态：**{status}**
- 范围：单场景、{summary['episodes']} 个 episode、{summary['cards']} 张含正例卡片；按 episode 四折隔离
- 随机多正例 Top-1：{random_top1:.1%}
- 最佳：`{best['variant']}`，Top-1 {best['top1_hit_mean']:.1%} ± {best['top1_hit_std']:.1%}，pairwise {best['pairwise_accuracy_mean']:.1%}

## 分析边界

本结果只判断冻结特征是否含可读出的人工 relevance 信号，不更新 InternVLA，也不代表导航或跨场景泛化。门槛要求最佳 Top-1 高于随机 5 个百分点且 pairwise 超过 55%。
"""
    (args.output_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
