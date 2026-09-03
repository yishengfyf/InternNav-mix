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

### 严格基线结果与 audit2 诊断设计

严格 30 帧基线最终完整读取 30 张 RGB，DROID 选择 22 个关键帧，Mapper 也
消费了全部 22 个关键帧，尾帧修复生效。DROID 原始有效 disparity 为
1,802,240 像素，但 `multiview=True, mv_count_th=2` 后只剩 9 个，最终 PLY
也只有 9 个 Gaussian。根因因此定位在 DROID `depth_filter` 的多视图一致性
层，而不是 Trident、动态掩码、非有限几何或 Mapper final drain。DROID
轨迹与 Habitat GT 做仅供审计的 Sim(3) 对齐后 RMSE 为 0.3818 m；GT 路径
2.25 m，而对齐后 DROID 路径仍为 4.09 m，表明全局尺度对齐不能消除局部漂移。

`freeocc_habitat_smoke_audit.sh` 的后续无参数调用按以下顺序，每次只推进一个
有界 profile：

1. `diagnostic_mv1_stride2`：仍启用多视图过滤，但把支持视图阈值从 2 降为 1；
   Gaussian stride 设为 2 控制显存。此实验用于判断失败是“完全没有跨视图
   支持”还是“阈值过严”。
2. `diagnostic_no_multiview_stride4`：关闭多视图过滤并用 stride 4 将理论上限
   从约 180 万 Gaussian 降到约 11.3 万。它只验证 Trident/Gaussian/导出链路，
   不能据此宣称几何可信。

增量补丁 `patches/freeocc_habitat_audit_counts_v2.patch` 额外记录每帧
`depth_filter` 的 `count>=1/2/3`、均值和最大值；这能避免仅凭最终阈值后数量
猜测过滤行为。分析器新增：三张输入 RGB、Sim(3) 对齐语义 Gaussian 透视图、
语义俯视图和 GT/DROID 轨迹的六宫格图
`analysis/freeocc_rgb_semantic_gaussians.png`，并在 JSON 中保存 11 类语义点数。
所有图均带有 offline diagnostic 标记。

第二轮实测结果进一步收敛了问题：`mv_count_th=1, stride=2` 仅保留 840 个
Gaussian，且对齐后 bbox 最大跨度约 300 m；关闭多视图过滤后虽能稳定导出
102,566 个带语义 Gaussian，但 bbox 最大跨度仍约 414 m。许多 DROID 深度
达到 100--1000 m，而 Habitat GT 相机仅移动 2.25 m。因此无过滤结果只证明
Trident、Gaussian 构建和 PLY 导出可运行，不能证明空间建图有效。

检查上游执行路径还发现：`configs/slam.yaml` 中声明的
`mono_depth: metric3d-vit_giant2` 没有任何 Python 代码读取；mono tracking 会把
Dataset 已加载的 depth 设为 `None`，随后 `DepthVideo.get_mapping_item()` 又把
DROID 的 `est_depth` 同时当作 `depth_prior`。也就是说当前发布代码的 mono
模式没有真正使用 Metric3D 单目深度先验。后续适配必须显式补齐 RGB→metric
depth 接口，或先用 Habitat depth 做仅供定位的 oracle ablation，不能继续把
调低一致性阈值当作修复。

`scripts/eval/evaluate_freeocc_habitat_gt.py` 使用保存的 Habitat depth 与 c2w
pose 构造离线 observed-surface GT，以 0.10 m 网格比较 Sim(3) 对齐后的
Gaussian center occupancy proxy，输出 exact IoU、0.10/0.20/0.50 m 的表面
precision/recall/F1 及双向距离。其产物是：

- `analysis/habitat_gt_occ_metrics.json`；
- `analysis/habitat_observed_occ.npz`（稀疏 occupied indices、origin、shape、
  voxel size；没有把未观测体素写成 free）；
- `analysis/freeocc_rgb_pred_gt_occ.png`（RGB、预测、GT、overlay）。

当前序列没有保存 Habitat semantic sensor 输出和 instance-to-category 映射，
所以几何 OCC 可量化，semantic mIoU 必须明确保持 `null`；不能用 FreeOcc 自己
的 Trident 标签伪造 semantic GT。跨 profile 汇总由
`scripts/eval/compare_freeocc_habitat_profiles.py` 生成。

### audit3：定位 mono 失败来自深度还是位姿

在接入真正的 RGB 单目深度网络前，先运行两个严格隔离的 oracle ablation：

1. `oracle_habitat_depth_estimated_pose_stride4`：给 DROID/Mapper Habitat depth，
   仍由 DROID 估计 pose。它回答“有正确尺度深度后，估计位姿是否足以建图”。
2. `oracle_habitat_depth_and_pose_stride4`：同时给 Habitat depth 和 GT pose，
   作为 FreeOcc Gaussian+Trident 后半链路在这段数据上的上界。

两者都明确不是 RGB-only 候选，也不能接入 DualVLN；depth/pose 只在独立进程
中作为诊断输入。若第二项好而第一项差，下一步需要同时解决 RGB-only pose；
若两项都好，则优先补接 Metric3D 等单目 metric-depth；若第二项仍差，问题在
相机约定、Mapper 或语义 Gaussian，而不是 DROID。该分解避免盲目下载大模型
后才发现下游坐标链路仍错误。

完整服务器实验保存在并列 run 目录中；为了复用固定自动审批链路，紧凑产物
还会复制到严格基线的 `diagnostics/<profile>/` 下，再重新生成排除清单自身的
`SHA256SUMS`。这不会改变任何 DualVLN 运行文件。

### audit4：坐标约定与姿态退化复核

全 oracle（Habitat depth + pose）的 71,596 个 Gaussian 全部位于 GT 网格内；
预测中心到 Habitat observed surface 的平均距离为 0.0037 m，10 cm precision
为 98.77%。但 0.10 m 网格内只有 15,119 个唯一预测体素，对比 198,954 个
GT observed-surface 体素，10 cm recall 只有 10.61%，exact IoU 为 7.30%。因此
图上最明显的差异主要是 Gaussian center proxy 的稀疏覆盖，而不是整体刚体
坐标错位；当前结果还不是完整 free/occupied voxel volume。

为了排除预测端和 GT 端共同重复同一轴错误造成的“假自洽”，
`scripts/eval/audit_habitat_camera_convention.py` 在 135 个跨帧对上执行实测深度
重投影。直接把保存 pose 当作 OpenCV c2w 时，10% 深度一致率为 87.03%，一致
区域 RGB L1 为 0.0402；额外施加 Habitat camera 到 OpenCV 的
`diag(1,-1,-1)` 翻转后，二者分别退化到 46.77% 和 0.1765。保存的 pose 因此
已经是 `+z forward, +y image-down` 的 OpenCV c2w，不能再翻转。产物为
`analysis/habitat_camera_convention_audit.json/png`。

真正的严重错误位于 RGB-only DROID pose。位置 Sim(3) 只能对齐相机中心，
不能修复朝向。新增的首帧归一化相对旋转审计显示：严格 mono 最终姿态误差
约 170.58°；即使输入 Habitat 真值深度、仍让 DROID 估计 pose，误差也在
timestamp 14 突然跳到约 58.94°，并在 timestamp 26 跳到约 110.31°。相应
全 oracle 的最大相对旋转误差只有 0.038°。这解释了“depth oracle + estimated
pose”相机中心 RMSE 仅 0.103 m、表面 F1@0.50m 却只有 0.0053 的矛盾。

当前结论是：FreeOcc 后半段的 depth→Gaussian→语义 PLY 坐标链路在 oracle
条件下可以工作；Habitat/FreeOcc 坐标转换不是主故障。下一阶段不能只补单目
metric depth，还必须修复/替换 DROID 在原地大角度转向时的姿态跟踪，并把
每帧深度和相邻位姿作为联合门控。Trident 语义目前也只能定性展示：该序列
未录制 semantic sensor GT，且可视化中 `table/floor/ceiling` 明显占比过高，
在补录 instance-to-category GT 前不得报告 semantic mIoU。

补充边界检查还发现：短窗口经 DROID motion filter 后若保留的关键帧数不超过
mapper warmup，原 `GaussianMapper.__call__(the_end=True)` 的两个结束分支都不
匹配，会无限忙循环。`patches/freeocc_short_sequence_finalization.patch` 让结束
阶段无延迟地消费现有关键帧并正常退出；此类结果仍会标记为 under-warmup
诊断，不能据此评估正常长序列精度。
