#!/usr/bin/env python3
"""四卡受限模型分片的单 batch 前向/反向安全检查。"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
ISOLATED_DEPS = Path("/data/usr_data/yifeifeng/internnav/dualvln_mainline/python_deps")
DATA_ROOT = Path("/data/usr_data/yifeifeng/internnav/dualvln_mainline/data/traj_data")
CHECKPOINT = Path("/home/yifeifeng/workspace/InternNav/checkpoints/InternVLA-N1")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ISOLATED_DEPS))
os.environ["INTERNNAV_TRAJ_DATA_ROOT"] = str(DATA_ROOT)
os.environ["INTERNNAV_CHECKPOINT_ROOT"] = str(CHECKPOINT.parent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {
        "schema_version": 1,
        "run_id": args.run_id,
        "git_commit": args.commit,
        "status": "failed",
        "metrics": {},
        "analysis": "四卡模型分片 smoke 未完成。",
    }
    try:
        import torch
        from transformers import AutoProcessor, AutoTokenizer

        from internnav.dataset.internvla_n1_lerobot_dataset import (
            DataCollatorForSupervisedDataset,
            NavPixelGoalDataset,
        )
        from internnav.model.basemodel.internvla_n1.internvla_n1 import (
            InternVLAN1ForCausalLM,
            InternVLAN1ModelConfig,
        )
        from internnav.model.basemodel.internvla_n1.trainable import configure_trainable_parameters
        from scripts.dualvln_mainline.real_overfit import summarize_gradients

        if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
            raise RuntimeError(f"需要至少四张 CUDA 卡，实际为 {torch.cuda.device_count()}")
        torch.manual_seed(23)
        torch.cuda.manual_seed_all(23)
        max_memory = {index: "20GiB" for index in range(4)}
        max_memory["cpu"] = "64GiB"

        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True, use_fast=False)
        tokenizer.model_max_length = 1024
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
            num_history=2,
            image_processor=processor.image_processor,
            transform_train=None,
            use_evidence_memory=True,
            max_pixels=224 * 224,
            min_pixels=224 * 224,
        )
        dataset = NavPixelGoalDataset(tokenizer, data_args)
        selected = [
            index
            for index, entry in enumerate(dataset.list_data_dict)
            if entry[7][0] >= 2 and entry[9] is not None
        ][:1]
        if len(selected) != 1:
            raise RuntimeError("真实 fixture 中没有找到可用的单个 pixel-goal 样本")
        batch = DataCollatorForSupervisedDataset(tokenizer=tokenizer, num_evidence_tokens=4)(
            [dataset[selected[0]]]
        )

        config = InternVLAN1ModelConfig.from_pretrained(CHECKPOINT, local_files_only=True)
        config.use_evidence_memory = True
        config.evidence_task_dim = 512
        config.evidence_bottleneck_dim = 512
        config.num_evidence_tokens = 4
        config.evidence_num_heads = 8
        config.evidence_num_stages = 4
        config.evidence_dropout = 0.0
        config.s2_loss_weight = 1.0
        config.trajectory_loss_weight = 1.0
        config.use_cache = False
        model = InternVLAN1ForCausalLM.from_pretrained(
            CHECKPOINT,
            config=config,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory=max_memory,
        )
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        trainable = configure_trainable_parameters(
            model,
            ("model.task_state_estimator.*", "model.evidence_memory.*", "model.cond_projector.*", "model.latent_queries"),
        )
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        input_device = model.get_input_embeddings().weight.device
        batch = {key: value.to(input_device) if hasattr(value, "to") else value for key, value in batch.items()}
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**batch)
        output.loss.backward()
        gradient_audit = summarize_gradients(model, torch)
        nonfinite = sum(item["nonfinite_gradients"] for item in gradient_audit.values())
        per_gpu = {
            str(index): {
                "allocated_mib": torch.cuda.memory_allocated(index) / 1024**2,
                "reserved_mib": torch.cuda.memory_reserved(index) / 1024**2,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(index) / 1024**2,
            }
            for index in range(4)
        }
        report["status"] = "passed" if torch.isfinite(output.loss) and nonfinite == 0 else "failed"
        report["metrics"] = {
            "sample_index": selected[0],
            "loss": float(output.loss.detach()),
            "s2_loss": float(output.s2_loss.detach()),
            "trajectory_loss": float(output.trajectory_loss.detach()),
            "nonfinite_gradients": nonfinite,
            "input_device": str(input_device),
            "device_map": getattr(model, "hf_device_map", {}),
            "per_gpu_memory": per_gpu,
            "trainable_parameters": trainable.trainable_parameters,
            "duration_s": time.monotonic() - started,
        }
        report["analysis"] = (
            "四卡分片单 batch 前向与反向均有限，可进入 8--32 样本受限过拟合。"
            if report["status"] == "passed"
            else "分片加载成功，但单 batch 前向或反向仍有非有限值；保留梯度审计，暂不启动正式过拟合。"
        )
    except Exception as error:
        report["metrics"]["duration_s"] = time.monotonic() - started
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (args.output_dir / "summary.md").write_text(
        "# 四卡模型分片单 batch smoke\n\n"
        f"- 状态：{'通过' if report['status'] == 'passed' else '失败'}\n"
        f"- 运行：`{args.run_id}`\n"
        f"- 结论：{report['analysis']}\n\n"
        "完整设备映射、显存和梯度审计见 `metrics.json`。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, default=str))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
