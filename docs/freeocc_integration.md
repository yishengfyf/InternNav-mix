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

FreeOcc 当前 `environment.yaml` 固定 Python 3.10、NumPy 1.26.4；上游 README
推荐 PyTorch 2.9/cu128。我们的服务器兼容性基线实际为 Python 3.10、
PyTorch 2.5.1+cu121、系统 nvcc 12.8、RTX A6000 (sm_86)，所需四个 CUDA
扩展、`localagg_prob`、PyTorch3D、Torch Scatter 和 Trident 均已通过 import。
这是项目实测组合，不代表上游官方矩阵。不要复用 InternNav 的 Python
3.9/habitat 环境；两个进程通过落盘序列/产物通信。

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

## 2026-09-03 首轮 Habitat mono smoke 与诊断补丁

固定输入是 `dhjEzFoUFzH/6763` 的 30 帧 RGB，depth 和 Habitat pose 只作
离线 GT/loader 审计；运行配置保持 `mode=mono`、`use_gt_poses=False`、
`multiview=True`。上游 commit 为
`f84a0f0ce28146b703d4d5bb5e061dc9a80be04e`。首次链路成功完成 DROID、
Backend BA、Trident 和 Gaussian Mapper，约 1.55 FPS，但
`mesh/final_mono.ply` 只有 3 个顶点，不能用于 OCC 精度结论。

代码检查确认两处尾帧问题和一处分配维度问题：

1. `BaseDataset.load_poses()` 在全有效 pose 时返回 `-1`，使 30 帧固定变为
   29 帧；遇到无效 pose 时返回 `i + 1` 还会让三种输入长度不一致。
2. Mapper final drain 在只消费 0--15 后把 `last_idx` 强制设为 21，跳过
   DROID 关键帧 16--20。
3. 合并 Gaussian 后的 `_features_rest`、`max_radii2D` 和
   `xyz_gradient_accum` 使用最后一帧 `Ni`，而不是总点数。

版本化补丁位于
`patches/freeocc_habitat_audit_f84a0f0.patch`。除修复上述问题外，它会输出：

- `mesh/final_mono_raw.ply` 与 Sim3 对齐后的 `mesh/final_mono.ply`；
- `audit/trajectories.npz`、estimated/GT c2w 文本轨迹；
- `audit/freeocc_mapping_audit.json`，记录 raw disparity、multiview、
  uncertainty/inlier、static mask、finite geometry 和最终 Gaussian 数量；
- 即使 `mapping.enable_occ_eval=False`，也仅用相机轨迹计算 Sim3，不再错误依赖
  ScanNet `gt_scene_data`。

分析命令：

```bash
python scripts/eval/analyze_freeocc_mapping_audit.py \
  --run-dir /path/to/freeocc/run \
  --expected-input-frames 30
```

生成 `analysis/freeocc_mapping_summary.json` 和
`analysis/freeocc_filter_trajectory_audit.png`。下一轮先用原严格参数重跑，
只有证据显示 `filter_collapse` 时才做 `multiview=False` 离线诊断对照；该对照
不改变 `unknown != free`，也不授予 FreeOcc 在线安全权威。
