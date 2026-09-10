import json

from scripts.dualvln_mainline.stage_report import build_report, write_report


def test_stage_report_writes_chinese_summary_metrics_and_svg(tmp_path):
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuites><testsuite tests="5" failures="1" errors="0" skipped="1" time="1.25"/></testsuites>',
        encoding="utf-8",
    )

    report = build_report("P1 协议检查", "run-1", "abcdef123456", 1, junit, "用于测试")
    write_report(report, tmp_path)

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    svg = (tmp_path / "metrics.svg").read_text(encoding="utf-8")
    assert metrics["status"] == "failed"
    assert metrics["metrics"]["passed"] == 3
    assert metrics["metrics"]["pass_rate"] == 0.75
    assert "简述与分析" in summary
    assert "P1 协议检查" in svg


def test_missing_junit_produces_explicit_failed_report(tmp_path):
    report = build_report("P1", "run-2", "abcdef", 3, tmp_path / "missing.xml", "环境缺少依赖")

    assert report["status"] == "failed"
    assert report["metrics"]["tests"] == 0
    assert "不能判断" in report["analysis"]
