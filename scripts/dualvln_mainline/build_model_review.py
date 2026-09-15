#!/usr/bin/env python3
"""Build an annotation-blind review subset and four-image sheets."""
import argparse
import hashlib
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def fit_image(path, width=560, height=350):
    image = Image.open(path).convert("RGB")
    image.thumbnail((width, height))
    canvas = Image.new("RGB", (width, height), "#111827")
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def make_sheet(row, base_dir, target):
    tile_w, tile_h, header, footer = 560, 350, 175, 54
    output = Image.new("RGB", (1160, header + 2 * (tile_h + footer) + 25), "white")
    draw = ImageDraw.Draw(output)
    draw.text((20, 14), row["annotation_id"], fill="#111827", font=load_font(22))
    instruction = textwrap.wrap("Instruction: " + row.get("instruction", "").strip(), width=100)[:3]
    draw.multiline_text((20, 47), "\n".join(instruction), fill="#111827", font=load_font(19), spacing=4)
    context = row.get("context", {})
    draw.text((20, 139), "Expert decision: %s  target=%s" % (context.get("expert_action"), context.get("expert_local_goal")), fill="#7c2d12", font=load_font(19))
    panels = [("CURRENT", row["current_image_path"], None)] + [(c["label"], c["image_path"], c) for c in row.get("candidates", [])]
    for index, (label, image_path, candidate) in enumerate(panels):
        x, y = 20 + (index % 2) * 570, header + (index // 2) * (tile_h + footer)
        output.paste(fit_image(base_dir / image_path, tile_w, tile_h), (x, y))
        if candidate is None:
            caption = "CURRENT observation"
        else:
            pose = candidate.get("relative_pose", {})
            caption = "History %s | frame %s | age %s | dist %.2fm | yaw %.1f deg" % (label, candidate.get("frame_id"), candidate.get("age"), pose.get("distance", 0), pose.get("dyaw_deg", 0))
        draw.text((x, y + tile_h + 12), caption, fill="#111827", font=load_font(16))
    output.save(target, quality=92)


def main():
    parser = argparse.ArgumentParser(description="生成不含人工答案的模型独立复核包")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", default="20260915")
    args = parser.parse_args()
    data = read_jsonl(args.manifest)
    selected = sorted(data, key=lambda r: hashlib.sha256((args.seed + ":" + r["annotation_id"]).encode()).hexdigest())[:args.count]
    sheets = args.output_dir / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    output = []
    for row in selected:
        name = row["annotation_id"] + ".jpg"
        make_sheet(row, args.base_dir, sheets / name)
        output.append({**row, "review_sheet": "sheets/" + name, "annotation": {"need_history": "", "preferred_evidence": [], "evidence_role": "", "misleading_evidence": [], "confidence": "", "short_reason": ""}, "reviewer_id": "model_reviewer", "review_protocol": "annotation_blind_v1"})
    (args.output_dir / "review_template.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in output) + "\n", encoding="utf-8")
    summary = "# 模型独立复核包\n\n- 样本数：%d\n- 抽样 seed：`%s`\n- 输入：candidate manifest，不含人工 annotation\n- 复核者：`model_reviewer`，不得作为第二位人工标注者\n\n完成后另存为 `model_review.jsonl`，只用于一致性和分歧分析。\n" % (len(output), args.seed)
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print("生成模型复核样本 %d 条：%s" % (len(output), args.output_dir))


if __name__ == "__main__":
    main()
