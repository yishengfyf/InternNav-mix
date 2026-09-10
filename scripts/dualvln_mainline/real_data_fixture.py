#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
ISOLATED_DEPS = Path("/data/usr_data/yifeifeng/internnav/dualvln_mainline/python_deps")
DATA_ROOT = Path("/data/usr_data/yifeifeng/internnav/dualvln_mainline/data/traj_data")
CHECKPOINT = Path("/home/yifeifeng/workspace/InternNav/checkpoints/InternVLA-N1")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ISOLATED_DEPS))
os.environ["INTERNNAV_TRAJ_DATA_ROOT"] = str(DATA_ROOT)


def write_report(args, report):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics = report.get("metrics", {})
    summary = f"""# 阶段结果简报

- 阶段：`P1 真实训练样本 fixture`
- 运行：`{args.run_id}`
- 提交：`{args.commit}`
- 状态：`{'通过' if report['status'] == 'passed' else '失败'}`

|指标|结果|
|---|---:|
|可用 pixel-goal 样本|{metrics.get('dataset_samples', 0)}|
|选中 episode|{metrics.get('episode_id', -1)}|
|当前 frame|{metrics.get('current_frame_id', -1)}|
|历史帧数|{metrics.get('history_count', 0)}|
|输入图像数|{metrics.get('image_count', 0)}|
|Evidence 占位符|{metrics.get('evidence_placeholders', 0)}|
|轨迹监督帧|{metrics.get('trajectory_frames', 0)}|
|耗时（秒）|{metrics.get('duration_s', 0):.3f}|

## 简述与分析

{report['analysis']}

样本与 batch 形状、因果断言和异常信息见 `metrics.json`，原始终端信息见 `fixture.log`。
"""
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    passed = report["status"] == "passed"
    color = "#2f855a" if passed else "#c53030"
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="620" height="140"><rect width="620" height="140" fill="#fff"/><text x="24" y="32" font-size="21" font-weight="700">P1 真实训练样本 fixture</text><rect x="24" y="60" width="520" height="26" fill="{color}"/><text x="556" y="79" font-size="15">{"通过" if passed else "失败"}</text><text x="24" y="116" font-size="13">train split | 单个 R2R scene | 不含 val-unseen</text></svg>\n'
    (args.output_dir / "metrics.svg").write_text(svg, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    start = time.monotonic()
    report = {
        "schema_version": 1,
        "run_id": args.run_id,
        "git_commit": args.commit,
        "status": "failed",
        "metrics": {},
        "analysis": "真实训练样本 fixture 尚未通过。",
    }
    try:
        import torch
        from transformers import AutoProcessor, AutoTokenizer

        from internnav.dataset.internvla_n1_lerobot_dataset import (
            EVIDENCE_TOKEN_INDEX,
            TRAJ_TOKEN_INDEX,
            DataCollatorForSupervisedDataset,
            NavPixelGoalDataset,
        )

        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True, use_fast=False)
        tokenizer.model_max_length = 2048
        processor = AutoProcessor.from_pretrained(CHECKPOINT, local_files_only=True)
        data_args = SimpleNamespace(
            vln_dataset_use="r2r_125cm_0_30",
            video_max_total_pixels=1664 * 28 * 28,
            video_min_total_pixels=256 * 28 * 28,
            model_type="internvla-n1",
            sample_step=4,
            predict_step_num=32,
            pixel_goal_only=True,
            num_future_steps=4,
            num_history=4,
            image_processor=processor.image_processor,
            transform_train=None,
            use_evidence_memory=True,
            max_pixels=224 * 224,
            min_pixels=224 * 224,
        )
        dataset = NavPixelGoalDataset(tokenizer, data_args)
        selected_index = next(
            index
            for index, entry in enumerate(dataset.list_data_dict)
            if entry[7][0] >= 4 and entry[9] is not None
        )
        entry = dataset.list_data_dict[selected_index]
        sample = dataset[selected_index]
        collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, num_evidence_tokens=4)
        batch = collator([sample])
        current_frame = entry[7][0]
        frame_ids = sample["evidence_frame_ids"]
        assert torch.all(frame_ids < current_frame)
        assert torch.all(sample["evidence_ages"] > 0)
        assert int(batch["evidence_valid_mask"].sum()) == sample["evidence_history_count"]
        evidence_count = int(batch["input_ids"].eq(EVIDENCE_TOKEN_INDEX).sum())
        trajectory_count = int(batch["input_ids"].eq(TRAJ_TOKEN_INDEX).sum())
        assert evidence_count == 4
        assert trajectory_count == 4
        assert torch.all(batch["labels"][batch["input_ids"].eq(EVIDENCE_TOKEN_INDEX)].eq(-100))
        assert torch.all(batch["labels"][batch["input_ids"].eq(TRAJ_TOKEN_INDEX)].eq(-100))
        report["status"] = "passed"
        report["metrics"] = {
            "dataset_samples": len(dataset),
            "selected_index": selected_index,
            "episode_id": entry[0],
            "current_frame_id": current_frame,
            "history_frame_ids": frame_ids.tolist(),
            "history_count": sample["evidence_history_count"],
            "image_count": sample["evidence_image_count"],
            "evidence_placeholders": evidence_count,
            "trajectory_placeholders": trajectory_count,
            "trajectory_frames": int(batch["video_frame_num"][0]),
            "input_shape": list(batch["input_ids"].shape),
            "pixel_values_shape": list(batch["pixel_values"].shape),
            "traj_images_shape": list(batch["traj_images"].shape),
            "traj_poses_shape": list(batch["traj_poses"].shape),
            "duration_s": time.monotonic() - start,
        }
        report["analysis"] = "真实 R2R train 样本已通过 processor、dataset 与 collator；历史均来自当前帧之前，evidence/traj 标签均被屏蔽。该阶段未加载 7B 参数。"
    except Exception as error:
        report["metrics"]["duration_s"] = time.monotonic() - start
        report["error"] = f"{type(error).__name__}: {error}"
        report["analysis"] = "真实样本读取或 batch 协议失败，不能进入真实过拟合；原始异常已保留。"
    write_report(args, report)
    print(f"real_data_fixture={report['status']} duration_s={report['metrics'].get('duration_s', 0):.3f}")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
