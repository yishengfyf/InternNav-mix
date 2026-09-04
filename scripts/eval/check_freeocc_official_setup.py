#!/usr/bin/env python3
"""Preflight FreeOcc against its official README and our A6000 baseline."""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeocc-root", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.freeocc_root.resolve()
    checks: dict[str, object] = {}

    checks["root"] = str(root)
    checks["environment_yaml"] = (root / "environment.yaml").is_file()
    checks["slam_config"] = (root / "configs/slam.yaml").is_file()
    checks["droid_checkpoint"] = (root / "pretrained/droid.pth").is_file()
    checks["sam_checkpoint"] = (root / "pretrained/sam_vit_b_01ec64.pth").is_file()
    checks["cuda_extensions"] = {}
    modules = (
        "torch", "droid_backends", "lietorch", "simple_knn._C",
        "diff_gaussian_rasterization", "local_aggregate_prob._C",
        "mmcv", "mmengine", "mmseg", "pytorch3d", "torch_scatter",
    )
    for name in modules:
        try:
            importlib.import_module(name)
            checks["cuda_extensions"][name] = "ok"
        except Exception as exc:  # report all failures in one pass
            checks["cuda_extensions"][name] = f"error: {type(exc).__name__}: {exc}"
    try:
        status = subprocess.run(
            ["git", "-C", str(root), "submodule", "status", "--recursive"],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
    except Exception as exc:
        status = [f"error: {exc}"]
    checks["submodules"] = status
    slam_text = (root / "configs/slam.yaml").read_text(encoding="utf-8") if checks["slam_config"] else ""
    checks["mono_depth_declared"] = "mono_depth:" in slam_text
    python_hits = subprocess.run(
        ["grep", "-RIl", "--include=*.py", "mono_depth", str(root / "src"), str(root / "run.py")],
        capture_output=True, text=True,
    ).stdout.splitlines()
    checks["mono_depth_python_consumers"] = python_hits
    checks["mono_depth_implemented"] = bool(python_hits)
    checks["official_occ_voxel_size_m"] = 0.08
    checks["notes"] = [
        "README verified stack is torch 2.9.0+cu128; the project A6000 baseline may remain on its already-tested torch 2.5.1+cu121 build.",
        "RTX A6000 uses TORCH_CUDA_ARCH_LIST=8.6; README's 12.0 example targets newer GPUs.",
        "A YAML mono_depth value is not functional unless mono_depth_python_consumers is non-empty.",
    ]
    output = json.dumps(checks, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    print(output)
    failed_imports = [v for v in checks["cuda_extensions"].values() if v != "ok"]
    return 1 if failed_imports else 0


if __name__ == "__main__":
    raise SystemExit(main())
