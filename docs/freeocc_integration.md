# FreeOcc monocular 分支接入说明

本分支 (`freeocc-integration`) 只增加外接适配和离线审计，不改变
`vlmap-safety-mvp` 的 DualVLN、SparseOcc、S1 队列或安全阈值。FreeOcc 使用
自己的 SLAM/Gaussian/Trident 环境，不能直接安装到 InternNav 的
`habiinter` 环境中。

## 适配边界

`internnav.utils.freeocc_adapter.FreeOccWorldMemory` 提供：

```python
memory.reset(scene_id, episode_id)
memory.update(rgb_uint8, step_id, pose=None)  # 只保存 RGB 和 ledger
memory.run_external()                         # 显式启动外部 FreeOcc
memory.load_occupancy("occ.npz")
memory.query_semantics("door")
```

`pose` 只写入 `pose_present` 诊断字段，不能用于在线安全判断。FreeOcc 单目
模式内部仍由 DROID-SLAM 估计相机运动；它不是无位姿的全局建图。

## 在服务器上准备独立环境

```bash
git clone --recurse-submodules https://github.com/the-masses/FreeOcc.git /data/usr_data/yifeifeng/internnav/third_party/FreeOcc
cd /data/usr_data/yifeifeng/internnav/third_party/FreeOcc
conda env create -f environment.yaml
conda activate freeocc
# 按仓库 README 安装 torch/cu128、mmcv、Trident 及 CUDA extensions
```

FreeOcc README 当前验证组合为 Python 3.11、PyTorch 2.9/cu128；不要复用
InternNav 的 Python 3.9/habitat 环境。先在 FreeOcc 仓库执行其 import checks，
再运行本项目桥接脚本。

## RGB 序列与 OCC 结果

把 Habitat RGB 保存为 `rgb/<scene>/<step>.png`，并为 FreeOcc 的 dataset
生成 `intrinsic/intrinsic_color.txt`。单目运行时不提供 depth/pose；若为了
对照提供 pose，只能作为离线 ablation。FreeOcc 的最终 `occ.npz` 至少需要
`pred`、可选 `valid_mask`、`voxel_origin`、`voxel_size`。

## GT 验证和可视化

```bash
MPLBACKEND=Agg python scripts/eval/freeocc_habitat_bridge.py \
  --pred-npz /path/to/FreeOcc/outputs/.../occ.npz \
  --gt-npz /path/to/mp3d_gt_occ.npz \
  --rgb-dir /path/to/rgb/scene \
  --out-dir /path/to/audit/scene
```

输出：

- `freeocc_metrics.json`：occupancy IoU、按标签 semantic IoU、mIoU、网格元数据；
- `freeocc_rgb_occ_overview.png`：中间 RGB 帧与 3D occupied semantic voxels
  的并排图，便于快速发现坐标、尺度和语义附着错误。

脚本要求预测和 GT 体素网格形状一致；origin/voxel size 不一致时只报警，
不会偷偷重采样。更复杂的 Sim(3) 对齐必须单独离线完成，不能回写在线地图。

## 接入 DualVLN 的后续步骤

1. 先只对固定 MP3D RGB 序列运行 FreeOcc，统计 SLAM 漂移、尺度误差、FPS、
   显存和语义类别稳定性。
2. 再把 `FreeOccWorldMemory.update()` 放入 evaluator 的观测旁路，仍以独立
   进程/低频关键帧运行；S2/S1 不读取其未经审计的结果。
3. 只有完成 scene-disjoint GT 审计后，才允许把门、墙、地面等语义作为
   recovery candidate 的排序或提示证据；SparseOcc 仍是唯一在线安全权威。
4. 将来若把 OCC/BEV token 送入 S2，必须训练 map adapter/LoRA；Frozen S2
   不应直接接收未见过的 tensor 或 overlay。

FreeOcc 的开放词汇 Trident 可以复用，但需通过名称列表/文本查询，不应把整条
R2R instruction 直接当作类别。DiScene/GPOcc 的闭集头不能直接替代它。
