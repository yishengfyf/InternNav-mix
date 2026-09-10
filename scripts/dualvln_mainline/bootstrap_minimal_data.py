#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import tarfile
import time
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "InternRobotics/InternData-N1"
FILENAME = "vln_ce/traj_data/r2r/17DRP5sb8fy.tar.gz"
EXPECTED_SHA256 = "15d2df50021a159abcf21018247e4f04eb0ffe263987375a9882baaaef4e1bd9"
EXPECTED_BYTES = 1_538_017_940


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(output_dir, report):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics = report["metrics"]
    summary = f"""# 阶段结果简报

- 阶段：`P1 最小真实数据准备`
- 运行：`{report['run_id']}`
- 提交：`{report['git_commit']}`
- 状态：`{'通过' if report['status'] == 'passed' else '失败'}`

|指标|结果|
|---|---:|
|归档字节数|{metrics.get('archive_bytes', 0)}|
|归档 SHA256 匹配|{metrics.get('sha256_match', False)}|
|归档成员数|{metrics.get('archive_members', 0)}|
|解包文件数|{metrics.get('extracted_files', 0)}|
|耗时（秒）|{metrics.get('duration_s', 0):.3f}|

## 简述与分析

{report['analysis']}

来源、目标路径和异常信息见 `metrics.json`，下载终端信息见 `bootstrap.log`。
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    values = [("大小", metrics.get("archive_bytes", 0) == EXPECTED_BYTES), ("哈希", metrics.get("sha256_match", False)), ("解包", metrics.get("extracted_files", 0) > 0)]
    rows = "".join(
        f'<text x="24" y="{80 + i * 40}" font-size="15">{label}</text><rect x="100" y="{63 + i * 40}" width="420" height="22" fill="{("#2f855a" if passed else "#c53030")}"/><text x="535" y="{80 + i * 40}" font-size="14">{("通过" if passed else "失败")}</text>'
        for i, (label, passed) in enumerate(values)
    )
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="620" height="210"><rect width="620" height="210" fill="#fff"/><text x="24" y="32" font-size="21" font-weight="700">P1 最小真实数据准备</text>{rows}</svg>\n'
    (output_dir / "metrics.svg").write_text(svg, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    report = {
        "schema_version": 1,
        "run_id": args.run_id,
        "git_commit": args.commit,
        "status": "failed",
        "source": {"repo_id": REPO_ID, "revision": "main", "filename": FILENAME},
        "target": str(args.data_root),
        "metrics": {},
        "analysis": "数据准备尚未完成。",
    }
    try:
        download_root = args.data_root.parent.parent / "downloads"
        cache_root = args.data_root.parent.parent / "hf_cache"
        archive = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=FILENAME,
                repo_type="dataset",
                revision="main",
                local_dir=download_root,
                cache_dir=cache_root,
            )
        )
        archive_bytes = archive.stat().st_size
        archive_hash = sha256(archive)
        if archive_bytes != EXPECTED_BYTES or archive_hash != EXPECTED_SHA256:
            raise RuntimeError(f"archive identity mismatch: bytes={archive_bytes} sha256={archive_hash}")

        marker = args.data_root / ".17DRP5sb8fy.complete"
        member_count = 0
        if not marker.is_file():
            args.data_root.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive, "r:gz") as bundle:
                members = bundle.getmembers()
                member_count = len(members)
                root = args.data_root.resolve()
                for member in members:
                    if member.issym() or member.islnk():
                        raise RuntimeError(f"archive links are not accepted: {member.name}")
                    target = (args.data_root / member.name).resolve()
                    if os.path.commonpath((str(root), str(target))) != str(root):
                        raise RuntimeError(f"unsafe archive member: {member.name}")
                bundle.extractall(args.data_root)
            marker.write_text(EXPECTED_SHA256 + "\n", encoding="ascii")
        extracted_files = sum(1 for path in args.data_root.rglob("*") if path.is_file() and path != marker)
        if extracted_files == 0:
            raise RuntimeError("archive extraction produced no files")
        report["status"] = "passed"
        report["metrics"] = {
            "archive_bytes": archive_bytes,
            "archive_sha256": archive_hash,
            "sha256_match": True,
            "archive_members": member_count,
            "extracted_files": extracted_files,
            "duration_s": time.monotonic() - start,
        }
        report["analysis"] = "已按 gated 仓库 main revision 选择性准备单个 R2R scene；这是训练 split 的最小工程样本，不包含 val-unseen 审计数据。"
    except Exception as error:
        report["metrics"]["duration_s"] = time.monotonic() - start
        report["error"] = f"{type(error).__name__}: {error}"
        report["analysis"] = "最小真实数据准备失败，不能进入真实过拟合；原始原因已保留。"
    write_report(args.output_dir, report)
    print(f"data_bootstrap={report['status']} duration_s={report['metrics'].get('duration_s', 0):.3f}")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
