#!/usr/bin/env python3
"""Selectively download and safely extract R2R scenes for annotation expansion."""
import argparse
import hashlib
import json
import os
import tarfile
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "InternRobotics/InternData-N1"
REPO_PREFIX = "vln_ce/traj_data/r2r"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive, destination, scene):
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise RuntimeError(f"归档不允许链接: {member.name}")
            parts = Path(member.name).parts
            if not parts or parts[0] != scene:
                raise RuntimeError(f"归档成员不属于预期场景 {scene}: {member.name}")
            target = (destination / member.name).resolve()
            if os.path.commonpath((str(root), str(target))) != str(root):
                raise RuntimeError(f"归档路径越界: {member.name}")
        bundle.extractall(destination)
    return len(members)


def write_report(output_dir, report):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = report["scenes"]
    summary_rows = "\n".join(
        f"|{row['scene']}|{row['archive_bytes']}|{row['archive_sha256'][:12]}|{row['extracted_files']}|{row['status']}|"
        for row in rows
    )
    summary = f"""# R2R 多场景标注数据准备简报

- 状态：`{'通过' if report['status'] == 'passed' else '失败'}`
- 场景数：`{len(rows)}`
- 新下载字节：`{report['downloaded_bytes']}`
- 已核验归档字节：`{report['verified_archive_bytes']}`
- 总耗时：`{report['duration_s']:.1f}` 秒

|场景|归档字节|SHA256 前缀|解包文件|状态|
|---|---:|---|---:|---|
{summary_rows}

所有归档均按 Hugging Face LFS 元数据校验大小和 SHA256，并在解包前检查路径越界与链接。数据仅来自 R2R train 轨迹场景，不含 val-seen/val-unseen。
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    bars = []
    for index, row in enumerate(rows):
        y = 62 + index * 34
        width = min(500, max(2, row["archive_bytes"] / 1024**3 * 300))
        color = "#2f855a" if row["status"] == "passed" else "#c53030"
        bars.append(
            f'<text x="20" y="{y + 16}" font-size="13">{row["scene"]}</text>'
            f'<rect x="150" y="{y}" width="{width:.1f}" height="20" fill="{color}"/>'
            f'<text x="{160 + width:.1f}" y="{y + 15}" font-size="12">{row["archive_bytes"] / 1024**3:.2f} GiB</text>'
        )
    height = 95 + len(rows) * 34
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="820" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="20" y="30" font-size="20" font-weight="700">R2R 多场景归档核验</text>'
        + "".join(bars)
        + "</svg>\n"
    )
    (output_dir / "metrics.svg").write_text(svg, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="准备用于第二批人工标注的 R2R train 场景")
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    args.archive_root.mkdir(parents=True, exist_ok=True)
    args.data_root.mkdir(parents=True, exist_ok=True)
    repo_files = HfApi().list_repo_tree(REPO_ID, REPO_PREFIX, repo_type="dataset", expand=True)
    metadata = {Path(item.path).name: item for item in repo_files if item.path.endswith(".tar.gz")}
    results = []
    downloaded_bytes = 0
    verified_archive_bytes = 0
    try:
        for scene in args.scenes:
            filename = f"{scene}.tar.gz"
            item = metadata.get(filename)
            if item is None or item.lfs is None:
                raise RuntimeError(f"仓库缺少带 LFS 哈希的场景归档: {filename}")
            expected_local = args.archive_root / REPO_PREFIX / filename
            archive_was_present = expected_local.is_file() and expected_local.stat().st_size == item.size
            archive = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=f"{REPO_PREFIX}/{filename}",
                    repo_type="dataset",
                    revision="main",
                    local_dir=args.archive_root,
                )
            )
            actual_size = archive.stat().st_size
            actual_sha = sha256(archive)
            if actual_size != item.size or actual_sha != item.lfs.sha256:
                raise RuntimeError(
                    f"{scene} 归档身份不一致: size={actual_size}, sha256={actual_sha}"
                )
            verified_archive_bytes += actual_size
            if not archive_was_present:
                downloaded_bytes += actual_size
            marker = args.data_root / f".{scene}.complete"
            member_count = 0
            if not marker.is_file() or marker.read_text(encoding="ascii").strip() != actual_sha:
                member_count = safe_extract(archive, args.data_root, scene)
                marker.write_text(actual_sha + "\n", encoding="ascii")
            scene_root = args.data_root / scene
            extracted_files = sum(1 for path in scene_root.rglob("*") if path.is_file())
            if extracted_files == 0:
                raise RuntimeError(f"{scene} 解包后没有文件")
            results.append(
                {
                    "scene": scene,
                    "status": "passed",
                    "archive": str(archive),
                    "archive_bytes": actual_size,
                    "archive_sha256": actual_sha,
                    "archive_members_extracted": member_count,
                    "extracted_files": extracted_files,
                }
            )
        report = {
            "schema_version": 1,
            "status": "passed",
            "scope": "R2R train scenes for causal annotation candidates",
            "repo_id": REPO_ID,
            "scenes": results,
            "downloaded_bytes": downloaded_bytes,
            "verified_archive_bytes": verified_archive_bytes,
            "duration_s": time.monotonic() - start,
        }
    except Exception as error:
        report = {
            "schema_version": 1,
            "status": "failed",
            "scope": "R2R train scenes for causal annotation candidates",
            "repo_id": REPO_ID,
            "scenes": results,
            "downloaded_bytes": downloaded_bytes,
            "verified_archive_bytes": verified_archive_bytes,
            "duration_s": time.monotonic() - start,
            "error": f"{type(error).__name__}: {error}",
        }
    write_report(args.output_dir, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
