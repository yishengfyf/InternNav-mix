#!/usr/bin/env python3
import argparse
import json
from html import escape
from pathlib import Path
from typing import Dict
from xml.etree import ElementTree


def status_label(status: str) -> str:
    return {"passed": "通过", "failed": "失败"}.get(status, status)


def summarize_junit(path: Path) -> Dict[str, float]:
    if not path.is_file():
        return {"tests": 0, "passed": 0, "failures": 0, "errors": 0, "skipped": 0, "duration_s": 0.0}

    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("./testsuite"))
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "duration_s": 0.0}
    for suite in suites:
        totals["tests"] += int(suite.attrib.get("tests", 0))
        totals["failures"] += int(suite.attrib.get("failures", 0))
        totals["errors"] += int(suite.attrib.get("errors", 0))
        totals["skipped"] += int(suite.attrib.get("skipped", 0))
        totals["duration_s"] += float(suite.attrib.get("time", 0.0))
    totals["passed"] = max(0, totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"])
    return totals


def build_report(stage: str, run_id: str, commit: str, exit_code: int, junit_path: Path, note: str) -> dict:
    metrics = summarize_junit(junit_path)
    executed = metrics["tests"] - metrics["skipped"]
    metrics["pass_rate"] = metrics["passed"] / executed if executed else 0.0
    status = "passed" if exit_code == 0 and metrics["failures"] == 0 and metrics["errors"] == 0 else "failed"
    if status == "passed":
        analysis = "本阶段协议检查全部通过，可以继续进入受这些协议约束的集成工作；该结果不代表导航收益已经成立。"
    elif metrics["tests"]:
        analysis = "本阶段存在失败或错误，应先定位失败用例并修复；在门槛恢复前不进入后续训练。"
    else:
        analysis = "测试未正常产生可解析指标，应先修复运行环境或入口；当前不能判断协议是否通过。"
    if note:
        analysis = f"{analysis} 补充：{note}"
    return {
        "schema_version": 1,
        "stage": stage,
        "run_id": run_id,
        "git_commit": commit,
        "status": status,
        "exit_code": exit_code,
        "metrics": metrics,
        "analysis": analysis,
    }


def write_svg(report: dict, path: Path) -> None:
    metrics = report["metrics"]
    bars = [
        ("通过", metrics["passed"], "#2f855a"),
        ("失败", metrics["failures"], "#c53030"),
        ("错误", metrics["errors"], "#b7791f"),
        ("跳过", metrics["skipped"], "#718096"),
    ]
    maximum = max(1, *(value for _, value, _ in bars))
    rows = []
    for index, (label, value, color) in enumerate(bars):
        y = 78 + index * 38
        width = 430 * value / maximum
        rows.append(
            f'<text x="24" y="{y + 17}" font-size="15">{escape(label)}</text>'
            f'<rect x="82" y="{y}" width="430" height="22" fill="#edf2f7" rx="3"/>'
            f'<rect x="82" y="{y}" width="{width:.1f}" height="22" fill="{color}" rx="3"/>'
            f'<text x="526" y="{y + 17}" font-size="15" font-weight="600">{value}</text>'
        )
    title = escape(f'{report["stage"]}：{status_label(report["status"])}')
    subtitle = escape(f'commit {report["git_commit"][:12]} | pass rate {metrics["pass_rate"]:.1%}')
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="620" height="250" viewBox="0 0 620 250">'
        '<rect width="620" height="250" fill="#ffffff"/>'
        f'<text x="24" y="32" font-size="21" font-weight="700">{title}</text>'
        f'<text x="24" y="57" font-size="13" fill="#4a5568">{subtitle}</text>' + "".join(rows) + '</svg>\n'
    )
    path.write_text(svg, encoding="utf-8")


def write_report(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = report["metrics"]
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = f"""# 阶段结果简报

- 阶段：`{report['stage']}`
- 运行：`{report['run_id']}`
- 提交：`{report['git_commit']}`
- 状态：`{status_label(report['status'])}`
- 退出码：`{report['exit_code']}`

|指标|结果|
|---|---:|
|测试总数|{metrics['tests']}|
|通过|{metrics['passed']}|
|失败|{metrics['failures']}|
|错误|{metrics['errors']}|
|跳过|{metrics['skipped']}|
|通过率|{metrics['pass_rate']:.1%}|
|测试耗时（秒）|{metrics['duration_s']:.3f}|

## 简述与分析

{report['analysis']}

可视化见 `metrics.svg`，原始测试结构化结果见 `junit.xml`。
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    write_svg(report, output_dir / "metrics.svg")


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 DualVLN 自动实验阶段结束报告")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--junit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    report = build_report(args.stage, args.run_id, args.commit, args.exit_code, args.junit, args.note)
    write_report(report, args.output_dir)
    metrics = report["metrics"]
    print("=== 阶段结果简述 ===")
    print(f"阶段: {report['stage']} | 状态: {status_label(report['status'])} | 退出码: {report['exit_code']}")
    print(
        f"测试: {metrics['tests']} | 通过: {metrics['passed']} | 失败: {metrics['failures']} | "
        f"错误: {metrics['errors']} | 跳过: {metrics['skipped']} | 通过率: {metrics['pass_rate']:.1%}"
    )
    print(f"分析: {report['analysis']}")
    print(f"报告目录: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
