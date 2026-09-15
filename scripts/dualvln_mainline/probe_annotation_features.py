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


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def attach_targets(data, annotation_path):
    annotations = {row["annotation_id"]: row for row in read_jsonl(annotation_path)}
    targets = []
    missing = set()
    for annotation_id, label in zip(data["annotation_ids"], data["candidate_labels"]):
        row = annotations.get(str(annotation_id))
        if row is None:
            missing.add(str(annotation_id))
            targets.append(0)
            continue
        preferred = set(row.get("annotation", {}).get("preferred_evidence", []))
        targets.append(int(str(label) in preferred))
    if missing:
        raise ValueError("特征中存在人工文件未覆盖的 annotation_id: " + ", ".join(sorted(missing)))
    return {**data, "targets": np.asarray(targets, dtype=np.int64)}


def feature_matrix(data, variant):
    current, history = data["visual"][:, 0], data["visual"][:, 1]
    if variant == "position":
        return np.stack(
            [data["candidate_labels"] == label for label in ("A", "B", "C")], axis=1
        ).astype(np.float32)
    if variant == "pose_age":
        return data["metadata"]
    content = np.concatenate((history - current, history * current), axis=1)
    if variant == "rgb_only":
        return content
    content = np.concatenate((content, history * data["text"]), axis=1)
    if variant == "rgb_text":
        return content
    return np.concatenate((content, data["metadata"]), axis=1)


def pairwise_examples(features, targets, annotation_ids, indices):
    examples, labels = [], []
    for annotation_id in sorted(set(annotation_ids[indices])):
        card = indices[annotation_ids[indices] == annotation_id]
        positive, negative = card[targets[card] == 1], card[targets[card] == 0]
        for positive_index in positive:
            for negative_index in negative:
                difference = features[positive_index] - features[negative_index]
                examples.extend((difference, -difference))
                labels.extend((1, 0))
    if not examples:
        raise ValueError("训练折没有可用的正负候选对")
    return np.asarray(examples), np.asarray(labels)


def evaluate(data, variant, seed, splits=4, protocol="group4", objective="pointwise"):
    targets, groups = data["targets"], data["episode_ids"]
    annotation_ids = data["annotation_ids"]
    features = feature_matrix(data, variant)
    fold_rows, hits, pairwise, items = [], [], [], []
    splitter = GroupKFold(n_splits=min(splits, len(set(groups))))
    for fold, (train, test) in enumerate(splitter.split(features, targets, groups)):
        if objective == "pairwise":
            fit_features, fit_targets = pairwise_examples(
                features, targets, annotation_ids, train
            )
        else:
            fit_features, fit_targets = features[train], targets[train]
        components = max(1, min(16, len(fit_features) - 1, features.shape[1]))
        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=components, whiten=False, random_state=seed),
            LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=2000, random_state=seed
            ),
        )
        model.fit(fit_features, fit_targets)
        scores = model.decision_function(features[test])
        fold_hits, fold_pairwise = [], []
        for annotation_id in sorted(set(annotation_ids[test])):
            mask = annotation_ids[test] == annotation_id
            truth, sample_scores = targets[test][mask], scores[mask]
            if not truth.any() or truth.all():
                continue
            fold_hits.append(int(truth[np.argmax(sample_scores)] == 1))
            positive, negative = sample_scores[truth == 1], sample_scores[truth == 0]
            item_pairwise = [float(pos > neg) for pos in positive for neg in negative]
            fold_pairwise.extend(item_pairwise)
            sample_indices = np.flatnonzero(mask)
            items.append(
                {
                    "annotation_id": str(annotation_id),
                    "episode_id": str(groups[test][sample_indices[0]]),
                    "top1_hit": fold_hits[-1],
                    "pairwise_accuracy": float(np.mean(item_pairwise)),
                    "random_top1": float(np.mean(truth)),
                }
            )
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
        "protocol": protocol,
        "objective": objective,
        "top1_hit": float(np.mean(hits)),
        "pairwise_accuracy": float(np.mean(pairwise)),
        "evaluated_cards": len(hits),
        "folds": fold_rows,
        "items": items,
    }


def poisson_binomial_tail(probabilities, observed):
    distribution = [1.0]
    for probability in probabilities:
        updated = [0.0] * (len(distribution) + 1)
        for count, mass in enumerate(distribution):
            updated[count] += mass * (1.0 - probability)
            updated[count + 1] += mass * probability
        distribution = updated
    return float(sum(distribution[observed:]))


def bootstrap_interval(items, field, seed, samples=4000):
    rng = np.random.default_rng(seed)
    episodes = sorted({item["episode_id"] for item in items})
    by_episode = {
        episode: [item[field] for item in items if item["episode_id"] == episode]
        for episode in episodes
    }
    estimates = []
    for _ in range(samples):
        selected = rng.choice(episodes, size=len(episodes), replace=True)
        values = [value for episode in selected for value in by_episode[episode]]
        estimates.append(float(np.mean(values)))
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


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
        '<svg xmlns="http://www.w3.org/2000/svg" width="620" height="335">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="20" y="31" font-size="20" font-weight="700">Frozen feature relevance probe</text>'
        + "".join(rows) + "</svg>\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="运行冻结 InternVLA 特征 relevance probe")
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[23, 47, 71])
    parser.add_argument("--objective", choices=("pointwise", "pairwise"), default="pointwise")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = np.load(args.features)
    data = {key: loaded[key] for key in loaded.files}
    data = attach_targets(data, args.annotations)
    positive_cards = np.unique(data["annotation_ids"][data["targets"] == 1])
    useful = np.isin(data["annotation_ids"], positive_cards)
    data = {key: value[useful] for key, value in data.items()}
    card_ids = np.unique(data["annotation_ids"])
    random_top1 = float(
        np.mean(
            [data["targets"][data["annotation_ids"] == item].mean() for item in card_ids]
        )
    )
    variant_names = ("position", "pose_age", "rgb_only", "rgb_text", "rgb_text_pose")
    runs = [
        evaluate(data, variant, seed, splits=4, protocol="group4", objective=args.objective)
        for variant in variant_names
        for seed in args.seeds
    ]
    loeo_runs = [
        evaluate(
            data,
            variant,
            seed,
            splits=len(set(data["episode_ids"])),
            protocol="leave_one_episode_out",
            objective=args.objective,
        )
        for variant in variant_names
        for seed in args.seeds
    ]
    variants = []
    for variant in variant_names:
        selected = [run for run in runs if run["variant"] == variant]
        loeo_selected = [run for run in loeo_runs if run["variant"] == variant]
        representative = selected[0]
        observed_hits = sum(item["top1_hit"] for item in representative["items"])
        chance_tail = poisson_binomial_tail(
            [item["random_top1"] for item in representative["items"]], observed_hits
        )
        variants.append(
            {
                "variant": variant,
                "top1_hit_mean": float(np.mean([run["top1_hit"] for run in selected])),
                "top1_hit_std": float(np.std([run["top1_hit"] for run in selected])),
                "pairwise_accuracy_mean": float(
                    np.mean([run["pairwise_accuracy"] for run in selected])
                ),
                "leave_one_episode_out_top1_mean": float(
                    np.mean([run["top1_hit"] for run in loeo_selected])
                ),
                "leave_one_episode_out_pairwise_mean": float(
                    np.mean([run["pairwise_accuracy"] for run in loeo_selected])
                ),
                "top1_episode_bootstrap_95ci": bootstrap_interval(
                    representative["items"], "top1_hit", args.seeds[0]
                ),
                "chance_tail_probability": chance_tail,
            }
        )
    method_variants = [item for item in variants if item["variant"] in {"rgb_only", "rgb_text", "rgb_text_pose"}]
    best = max(method_variants, key=lambda item: item["top1_hit_mean"])
    pose = next(item for item in variants if item["variant"] == "pose_age")
    if args.objective == "pairwise":
        passed = (
            best["top1_hit_mean"] > random_top1 + 0.05
            and best["leave_one_episode_out_top1_mean"] > random_top1 + 0.05
            and best["pairwise_accuracy_mean"] > pose["pairwise_accuracy_mean"] + 0.05
            and best["leave_one_episode_out_pairwise_mean"]
            > pose["leave_one_episode_out_pairwise_mean"] + 0.05
        )
    else:
        passed = (
            best["top1_hit_mean"] > random_top1 + 0.05
            and best["pairwise_accuracy_mean"] > 0.55
            and best["leave_one_episode_out_top1_mean"] > random_top1 + 0.05
            and best["leave_one_episode_out_pairwise_mean"] > 0.55
            and best["top1_hit_mean"] > pose["top1_hit_mean"] + 0.05
        )
    evidence_strength = (
        "promising_but_underpowered"
        if passed and (best["chance_tail_probability"] > 0.10 or best["top1_episode_bootstrap_95ci"][0] <= random_top1)
        else "supported" if passed else "not_supported"
    )
    summary = {
        "schema_version": 1,
        "status": "passed" if passed else "not_passed",
        "scope": "single scene; episode-grouped 4-fold; diagnostic only",
        "objective": args.objective,
        "candidate_examples": len(data["targets"]),
        "cards": len(card_ids),
        "episodes": len(set(data["episode_ids"])),
        "positive_examples": int(data["targets"].sum()),
        "random_top1": random_top1,
        "variants": variants,
        "runs": runs,
        "leave_one_episode_out_runs": loeo_runs,
        "evidence_strength": evidence_strength,
        "gate": (
            "pairwise: group4 and leave-one-episode-out top1 > random + 0.05; pairwise accuracy > pose_age + 0.05"
            if args.objective == "pairwise"
            else "pointwise: group4 and leave-one-episode-out top1 > random + 0.05, pairwise > 0.55; group4 top1 > pose_age + 0.05"
        ),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_svg(summary, args.output_dir / "metrics.svg")
    status = "通过" if passed else "未通过"
    report = f"""# 冻结 InternVLA 特征 relevance probe

- 状态：**{status}**
- 训练目标：`{args.objective}`
- 范围：单场景、{summary['episodes']} 个 episode、{summary['cards']} 张含正例卡片；按 episode 四折隔离
- 随机多正例 Top-1：{random_top1:.1%}
- 最佳：`{best['variant']}`，Top-1 {best['top1_hit_mean']:.1%} ± {best['top1_hit_std']:.1%}，pairwise {best['pairwise_accuracy_mean']:.1%}
- Leave-one-episode-out：Top-1 {best['leave_one_episode_out_top1_mean']:.1%}，pairwise {best['leave_one_episode_out_pairwise_mean']:.1%}
- episode bootstrap 95% 区间：{best['top1_episode_bootstrap_95ci'][0]:.1%}--{best['top1_episode_bootstrap_95ci'][1]:.1%}
- 精确同卡随机尾概率：{best['chance_tail_probability']:.3f}
- 证据强度：`{evidence_strength}`

## 分析边界

本结果只判断冻结特征是否含可读出的人工 relevance 信号，不更新 InternVLA，也不代表导航或跨场景泛化。工程门槛同时要求四折与 leave-one-episode-out 高于随机，并优于 pose/age；统计强度另由 bootstrap 区间和精确随机尾概率标记。
"""
    (args.output_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
