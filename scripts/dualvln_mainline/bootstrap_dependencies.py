#!/usr/bin/env python3
import argparse
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path


REQUIREMENTS = ("pyarrow==17.0.0", "decord==0.6.0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.target.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    sys.path.insert(0, str(args.target))
    already_ready = all((args.target / package).exists() for package in ("pyarrow", "decord"))
    command = [sys.executable, "-m", "pip", "install", "--target", str(args.target), "--no-deps", *REQUIREMENTS]
    completed = subprocess.CompletedProcess(command, 0) if already_ready else subprocess.run(command, check=False, text=True)
    versions = {}
    error = None
    for package in ("pyarrow", "decord"):
        try:
            module = importlib.import_module(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
    passed = completed.returncode == 0 and error is None
    duration = time.monotonic() - start
    report = {
        "schema_version": 1,
        "stage": "P1 隔离依赖准备",
        "run_id": args.run_id,
        "git_commit": args.commit,
        "status": "passed" if passed else "failed",
        "exit_code": 0 if passed else 1,
        "target": str(args.target),
        "requirements": REQUIREMENTS,
        "install_skipped": already_ready,
        "versions": versions,
        "duration_s": duration,
        "error": error,
        "analysis": "依赖仅安装到新主线专属目录，没有修改共享 conda 环境。" if passed else "隔离依赖准备失败，真实 dataset 仍不可加载。",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = f"""# 阶段结果简报

- 阶段：`{report['stage']}`
- 运行：`{args.run_id}`
- 提交：`{args.commit}`
- 状态：`{'通过' if passed else '失败'}`
- 目标目录：`{args.target}`
- 版本：`{json.dumps(versions, ensure_ascii=False)}`
- 耗时：`{duration:.3f}` 秒

## 简述与分析

{report['analysis']}
"""
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    color = "#2f855a" if passed else "#c53030"
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="620" height="130"><rect width="620" height="130" fill="#fff"/><text x="24" y="32" font-size="21" font-weight="700">P1 隔离依赖准备</text><rect x="24" y="60" width="520" height="26" fill="{color}"/><text x="558" y="79" font-size="15">{"通过" if passed else "失败"}</text></svg>\n'
    (args.output_dir / "metrics.svg").write_text(svg, encoding="utf-8")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
