#!/usr/bin/env python3
"""Extract frozen InternVLA features for evidence-relevance probes."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ISOLATED_DEPS = Path("/data/usr_data/yifeifeng/internnav/dualvln_mainline/python_deps")
if ISOLATED_DEPS.is_dir():
    sys.path.insert(0, str(ISOLATED_DEPS))

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from internnav.model.basemodel.internvla_n1.evidence_conditioning import pool_visual_features
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM, InternVLAN1ModelConfig


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="提取人工标注卡片的冻结 InternVLA 特征")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-pixels", type=int, default=224 * 224)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.manifest)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, use_fast=False)
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    config = InternVLAN1ModelConfig.from_pretrained(args.checkpoint, local_files_only=True)
    config.use_evidence_memory = False
    model = InternVLAN1ForCausalLM.from_pretrained(
        args.checkpoint,
        config=config,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model.eval()
    device = next(model.visual.parameters()).device
    visual, text, annotation_ids, episode_ids = [], [], [], []
    candidate_labels, targets, metadata = [], [], []

    with torch.inference_mode():
        for index, row in enumerate(rows):
            entries = [("CURRENT", row["current_image_path"], None)] + [
                (candidate["label"], candidate["image_path"], candidate) for candidate in row["candidates"]
            ]
            images = [Image.open(args.base_dir / image_path).convert("RGB") for _, image_path, _ in entries]
            inputs = processor(images=images, return_tensors="pt", max_pixels=args.max_pixels)
            pixels = inputs["pixel_values"].to(device=device, dtype=model.visual.dtype)
            grids = inputs["image_grid_thw"].to(device)
            image_tokens = model.visual(pixels, grid_thw=grids)
            pooled = pool_visual_features(
                image_tokens, grids, config.vision_config.spatial_merge_size
            ).float().cpu()
            tokenized = tokenizer(row["instruction"], return_tensors="pt", add_special_tokens=True)
            input_ids = tokenized["input_ids"].to(model.get_input_embeddings().weight.device)
            instruction = model.get_input_embeddings()(input_ids).float().mean(dim=1).cpu()[0]
            for candidate_index, (label, _, candidate) in enumerate(entries[1:], 1):
                pose = candidate["relative_pose"]
                visual.append(torch.stack((pooled[0], pooled[candidate_index])).numpy())
                text.append(instruction.numpy())
                annotation_ids.append(row["annotation_id"])
                episode_ids.append(str(row["episode_id"]))
                candidate_labels.append(label)
                targets.append(0)
                metadata.append(
                    [
                        candidate["age"], pose["dx"], pose["dy"], pose["distance"],
                        np.sin(pose["dyaw_rad"]), np.cos(pose["dyaw_rad"]),
                    ]
                )
            print(
                json.dumps(
                    {
                        "processed": index + 1,
                        "total": len(rows),
                        "gpu_peak_mb": torch.cuda.max_memory_allocated() / 2**20,
                    }
                ),
                flush=True,
            )

    np.savez_compressed(
        args.output_dir / "features.npz",
        visual=np.asarray(visual, dtype=np.float32),
        text=np.asarray(text, dtype=np.float32),
        metadata=np.asarray(metadata, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.int64),
        annotation_ids=np.asarray(annotation_ids),
        episode_ids=np.asarray(episode_ids),
        candidate_labels=np.asarray(candidate_labels),
    )
    metrics = {
        "schema_version": 1,
        "status": "completed",
        "rows": len(rows),
        "candidate_examples": len(targets),
        "feature_dim": int(visual[0].shape[-1]),
        "checkpoint": str(args.checkpoint),
        "checkpoint_config_sha256": sha256(args.checkpoint / "config.json"),
        "source_manifest_sha256": sha256(args.manifest),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_peak_mb": torch.cuda.max_memory_allocated() / 2**20,
    }
    (args.output_dir / "extraction_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
