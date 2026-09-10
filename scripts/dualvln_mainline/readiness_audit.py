#!/usr/bin/env python3
import argparse
import importlib.util
import json
import shutil
import subprocess
from html import escape
from pathlib import Path


DATASET_CANDIDATES = (
    Path("/data/usr_data/yifeifeng/internnav/worktrees/dualvln-spatial-memory-v1/traj_data"),
    Path("/data/usr_data/yifeifeng/internnav/traj_data"),
    Path("/data/usr_data/yifeifeng/internnav/hf_repos/InternData-N1/vln_n1/traj_data"),
    Path("/home/yifeifeng/workspace/InternNav/traj_data"),
)
RAW_DATA_ROOT = Path("/data/usr_data/yifeifeng/internnav/hf_repos/InternData-N1/vln_ce/raw_data")
CHECKPOINT = Path("/home/yifeifeng/workspace/InternNav/checkpoints/InternVLA-N1")


def gpu_state():
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        fields = [int(field.strip()) for field in line.split(",")]
        if len(fields) == 5:
            rows.append(dict(zip(("index", "total_mib", "used_mib", "free_mib", "utilization_pct"), fields)))
    return rows


def write_svg(report, path):
    checks = report["checks"]
    rows = []
    for index, (label, ready) in enumerate(checks.items()):
        y = 72 + index * 34
        color = "#2f855a" if ready else "#c53030"
        rows.append(
            f'<circle cx="32" cy="{y}" r="8" fill="{color}"/>'
            f'<text x="52" y="{y + 5}" font-size="14">{escape(label)}</text>'
            f'<text x="520" y="{y + 5}" font-size="14" font-weight="600">{"就绪" if ready else "未就绪"}</text>'
        )
    height = 100 + len(rows) * 34
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="620" height="{height}">'
        f'<rect width="620" height="{height}" fill="#ffffff"/>'
        f'<text x="24" y="31" font-size="21" font-weight="700">P1 真实实验就绪审计：{escape(report["status_cn"])}</text>'
        f'<text x="24" y="51" font-size="13" fill="#4a5568">commit {escape(report["git_commit"][:12])}</text>'
        + "".join(rows)
        + "</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {name: [str(root / name) for root in DATASET_CANDIDATES if (root / name).is_dir()] for name in ("r2r", "rxr", "scalevln")}
    raw_data = {name: (RAW_DATA_ROOT / name).is_dir() for name in ("r2r", "rxr", "scalevln")}
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in ("torch", "pytest", "transformers", "diffusers", "flash_attn", "pyarrow", "peft", "deepspeed", "decord", "torchcodec")
    }
    gpus = gpu_state()
    gpu_available = any(gpu["free_mib"] >= 24000 and gpu["utilization_pct"] <= 20 for gpu in gpus)
    disk_data = shutil.disk_usage("/data/usr_data/yifeifeng/internnav")
    disk_home = shutil.disk_usage("/home/yifeifeng/workspace")
    checks = {
        "checkpoint": CHECKPOINT.is_dir(),
        "至少一个 LeRobot 训练集": any(datasets.values()),
        "pyarrow": packages["pyarrow"],
        "视频解码器": packages["decord"] or packages["torchcodec"],
        "flash-attn": packages["flash_attn"],
        "空闲且低负载 GPU": gpu_available,
        "/data 可用空间不少于 100 GiB": disk_data.free >= 100 * 1024**3,
    }
    ready = all(checks.values())
    blockers = [label for label, value in checks.items() if not value]
    report = {
        "schema_version": 1,
        "stage": "P1 真实过拟合与短闭环就绪审计",
        "run_id": args.run_id,
        "git_commit": args.commit,
        "status": "ready" if ready else "blocked",
        "status_cn": "就绪" if ready else "未就绪",
        "checks": checks,
        "blockers": blockers,
        "datasets": datasets,
        "raw_data": raw_data,
        "packages": packages,
        "gpus": gpus,
        "disk": {
            "data_free_gib": round(disk_data.free / 1024**3, 2),
            "home_free_gib": round(disk_home.free / 1024**3, 2),
        },
        "analysis": (
            "真实 8--32 样本过拟合和短闭环具备启动条件。"
            if ready
            else "当前不启动真实训练；阻塞项为：" + "、".join(blockers) + "。原始标注存在不等于 LeRobot 轨迹训练样本存在。"
        ),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "config.json").write_text(
        json.dumps({"dataset_candidates": [str(path) for path in DATASET_CANDIDATES], "checkpoint": str(CHECKPOINT)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = f"""# 阶段结果简报

- 阶段：`{report['stage']}`
- 运行：`{args.run_id}`
- 提交：`{args.commit}`
- 状态：`{report['status_cn']}`

## 检查结果

|条件|结果|
|---|---|
""" + "".join(f"|{label}|{'就绪' if value else '未就绪'}|\n" for label, value in checks.items()) + f"""

## 简述与分析

{report['analysis']}

GPU 明细、包探测、数据候选路径和磁盘信息见 `metrics.json`；可视化见 `metrics.svg`。
"""
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    write_svg(report, args.output_dir / "metrics.svg")
    print(f"readiness={report['status']} blockers={','.join(blockers) if blockers else 'none'}")


if __name__ == "__main__":
    main()
