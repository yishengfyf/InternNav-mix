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
    parser.add_argument("--attention", default="flash_attention_2", choices=("flash_attention_2", "eager", "sdpa"))
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--loss", default="total", choices=("total", "s2", "trajectory"))
    parser.add_argument("--gradient-bypass", action="store_true")
    parser.add_argument("--evidence", default="on", choices=("on", "off", "detached"))
    parser.add_argument("--no-trajectory", action="store_true")
    parser.add_argument("--no-s2", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--check-checkpoint", action="store_true")
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
            use_evidence_memory=args.evidence != "off",
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
        batch = DataCollatorForSupervisedDataset(
            tokenizer=tokenizer, num_evidence_tokens=4 if args.evidence != "off" else 0
        )(
            [dataset[selected[0]]]
        )
        if args.no_trajectory:
            for key in ("traj_poses", "traj_images", "traj_depths", "video_frame_num"):
                batch.pop(key, None)
        if args.no_s2:
            batch.pop("labels", None)

        config = InternVLAN1ModelConfig.from_pretrained(CHECKPOINT, local_files_only=True)
        config.use_evidence_memory = args.evidence != "off"
        config.evidence_task_dim = 512
        config.evidence_bottleneck_dim = 512
        config.num_evidence_tokens = 4
        config.evidence_num_heads = 8
        config.evidence_num_stages = 4
        config.evidence_dropout = 0.0
        config.s2_loss_weight = 1.0
        config.trajectory_loss_weight = 1.0
        config.use_cache = False
        config.evidence_gradient_bypass = args.gradient_bypass
        config.evidence_gradient_bypass_scale = 0.1
        config.evidence_detach_tokens = args.evidence == "detached"
        config.numeric_diagnostics = True
        model_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
        model = InternVLAN1ForCausalLM.from_pretrained(
            CHECKPOINT,
            config=config,
            local_files_only=True,
            torch_dtype=model_dtype,
            attn_implementation=args.attention,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory=max_memory,
        )
        module_diagnostics = {}

        def summarize_tensor(tensor):
            detached = tensor.detach()
            finite = torch.isfinite(detached)
            values = detached[finite].float()
            return {
                "shape": list(detached.shape),
                "dtype": str(detached.dtype),
                "device": str(detached.device),
                "finite": bool(finite.all()),
                "nonfinite": int(detached.numel() - finite.sum().item()),
                "min": float(values.min()) if values.numel() else None,
                "max": float(values.max()) if values.numel() else None,
                "max_abs": float(values.abs().max()) if values.numel() else None,
            }

        def record_output(name):
            def hook(_module, _inputs, output):
                if torch.is_tensor(output):
                    module_diagnostics[name] = summarize_tensor(output)
            return hook

        if args.evidence != "off":
            estimator = model.get_model().task_state_estimator
            estimator.visual_projection.register_forward_hook(record_output("task.visual_projection"))
            estimator.text_projection.register_forward_hook(record_output("task.text_projection"))
            for index, layer in enumerate(estimator.fusion):
                layer.register_forward_hook(record_output(f"task.fusion.{index}"))
        if not args.no_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        if not args.forward_only:
            model.enable_input_require_grads()
        allowlist = ["model.cond_projector.*", "model.latent_queries"]
        if args.evidence != "off":
            allowlist[:0] = ["model.task_state_estimator.*", "model.evidence_memory.*"]
        trainable = configure_trainable_parameters(
            model,
            tuple(allowlist),
        )
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        if args.evidence != "off":
            model.get_model().reset_evidence_parameters()
        trainable_parameter_diagnostics = {
            name: summarize_tensor(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        checkpoint_nonfinite = 0
        checkpoint_parameters = 0
        if args.check_checkpoint:
            for parameter in model.parameters():
                checkpoint_parameters += parameter.numel()
                checkpoint_nonfinite += parameter.numel() - int(torch.isfinite(parameter).sum())
        input_device = model.get_input_embeddings().weight.device
        batch = {key: value.to(input_device) if hasattr(value, "to") else value for key, value in batch.items()}
        model.eval() if args.forward_only else model.train()
        grad_context = torch.no_grad() if args.forward_only else torch.enable_grad()
        with grad_context:
            if args.dtype == "bfloat16":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(**batch)
            else:
                output = model(**batch)
        selected_loss = {
            "total": output.loss,
            "s2": output.s2_loss,
            "trajectory": output.trajectory_loss,
        }[args.loss]
        if selected_loss is None:
            raise RuntimeError(f"所选 {args.loss} loss 在当前 case 中不存在")
        if not args.forward_only:
            selected_loss.backward()
        gradient_audit = {} if args.forward_only else summarize_gradients(model, torch)
        nonfinite = sum(item["nonfinite_gradients"] for item in gradient_audit.values())
        numeric_diagnostics = getattr(model, "numeric_diagnostics", {})
        first_nonfinite = next(
            (name for name, values in numeric_diagnostics.items() if not values["finite"]),
            None,
        )
        per_gpu = {
            str(index): {
                "allocated_mib": torch.cuda.memory_allocated(index) / 1024**2,
                "reserved_mib": torch.cuda.memory_reserved(index) / 1024**2,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(index) / 1024**2,
            }
            for index in range(4)
        }
        report["status"] = "passed" if first_nonfinite is None and torch.isfinite(selected_loss) and nonfinite == 0 else "failed"
        report["metrics"] = {
            "sample_index": selected[0],
            "loss": float(output.loss.detach()) if output.loss is not None else None,
            "s2_loss": float(output.s2_loss.detach()) if output.s2_loss is not None else None,
            "trajectory_loss": float(output.trajectory_loss.detach()) if output.trajectory_loss is not None else None,
            "backward_loss": args.loss,
            "forward_only": args.forward_only,
            "evidence": args.evidence,
            "trajectory_enabled": not args.no_trajectory,
            "s2_enabled": not args.no_s2,
            "valid_s2_labels": int(batch["labels"].ne(-100).sum()) if "labels" in batch else 0,
            "valid_trajectory_frames": int(batch["video_frame_num"].sum()) if "video_frame_num" in batch else 0,
            "first_nonfinite_tensor": first_nonfinite,
            "finite_checks": numeric_diagnostics,
            "module_finite_checks": module_diagnostics,
            "trainable_parameter_checks": trainable_parameter_diagnostics,
            "checkpoint_parameters_checked": checkpoint_parameters,
            "checkpoint_nonfinite": checkpoint_nonfinite,
            "nonfinite_gradients": nonfinite,
            "gradient_audit": gradient_audit,
            "input_device": str(input_device),
            "attention": args.attention,
            "gradient_checkpointing": not args.no_gradient_checkpointing,
            "dtype": args.dtype,
            "gradient_bypass": args.gradient_bypass,
            "gradient_bypass_scale": 0.1,
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
