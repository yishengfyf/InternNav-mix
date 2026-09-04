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

### audit5：多 episode 密度/精度复核（2026-09-03）

为区分“FreeOcc 后端本身稀疏”和“输入帧/过滤策略造成的稀疏”，在四个
DualVLN replay episode 上使用相同的 Habitat depth + GT camera pose oracle，
输出分辨率 256×320、`multiview=true`、`mv_count_th=2`。r47 额外跑了
`frame_gaussians_stride=1`，其余使用工程折中 `stride=2`。所有指标都只比较
匹配帧产生的 observed surface；未观测体素不计为 free，semantic mIoU 保持
`null`（replay 没有 Habitat semantic sensor/实例类别真值）。

| episode/profile | frames / mapper cameras | Gaussian centers | exact surface IoU | F1 @ 0.10 m | F1 @ 0.20 m | precision / recall @ 0.10 m |
|---|---:|---:|---:|---:|---:|---:|
| r47 stride=1 | 100 / 100 | 7,125,925 | 0.658 | 0.854 | 0.900 | 0.950 / 0.775 |
| SN83 stride=2 | 120 / 101 | 1,589,243 | 0.475 | 0.722 | 0.782 | 0.974 / 0.574 |
| 5q7 stride=2 | 120 / 102 | 1,807,107 | 0.544 | 0.776 | 0.828 | 0.987 / 0.639 |
| PuK stride=2 | 120 / 120 | 1,867,225 | 0.233 | 0.444 | 0.548 | 0.952 / 0.290 |

结论：stride=1 确实能把覆盖率推高，但单个 PLY 约 0.8 GB；stride=2 将
PLY 压到约 170–210 MB，预测中心的 10 cm precision 仍为 95–99%，但不同
场景的 recall 受可见表面、运动路径和多视图支持显著影响（PuK 的 observed
surface 更大，recall 下降并不等价于坐标系错误）。四个 profile 的预测点均
100% 落在 GT 评估网格内，且 GT-pose 轨迹 RMSE < 1e-6 m，说明本轮 oracle
实验没有发现新的坐标变换错误。当前“画面稀疏”主要是 Gaussian 表示和
`stride`/关键帧覆盖问题，不是 FreeOcc 单目后端的固定上限。

`scripts/eval/summarize_freeocc_replay_results.py` 会生成上述 JSON/柱状图；
`scripts/eval/compact_freeocc_voxels.py` 可把百万级 Gaussian 合并为 10 cm
语义体素（通常约数万体素），作为受困判断的只读摘要，避免把完整 PLY 放入
DualVLN 进程。该摘要仍标记 shadow-only，不改变 SparseOcc 的安全权威。

本轮还修复并验证了 final-drain no-progress 卡死：5q7 旧进程在
`last_idx=112` 重复循环；应用 `freeocc_final_drain_progress_v2.patch` 后打印
`Final drain made no progress; exporting current map.` 并正常导出。补丁必须
应用在服务器的 FreeOcc 源码目录，而不是 InternNav worktree。

### audit6：真实 RGB-only 反例（2026-09-03）

为验证 oracle 结论是否能迁移到真实输入，使用 SN83 的前 60 帧运行
`mode=mono, use_gt_poses=false`（不把 replay depth/pose 送入 FreeOcc）。该
序列仍能完成 DROID、Backend、Trident 和 PLY 导出，但审计显示：

- 多视图过滤后仅保留 247,596 / 3,686,400 像素（6.7%），最终 Gaussian 约
  21,476 个；
- 相机姿态相对误差在 timestamp 26 突然约 56°，timestamp 36 后累计约
  140–144°；
- 仅做离线 Sim(3) 后相机中心 RMSE 为 0.729 m，10 cm precision/recall/F1
  分别约 0.108 / 0.004 / 0.008，exact surface IoU 约 0.0013。

这组结果与 GT depth+GT pose 的 0.475 IoU、0.722 F1@0.10m 形成明确对照：
当前上游 `mono_depth: metric3d-vit_giant2` 只是 YAML 字符串，代码没有加载
Metric3D checkpoint；而 DROID 在大角度原地旋转时也会发生姿态跳变。因此
下一步必须先做“稳定姿态/初始化 + 真正的 RGB→metric-depth”两条链路的隔离
实验，再考虑把 FreeOcc 作为 DualVLN 在线 worker。仅调低 `mv_count_th` 会
增加错误点，不能视为修复。

### audit7：延迟初始化的 RGB-only 对照（2026-09-03）

同一 SN83 序列再从第 20 帧开始取 60 帧（跳过最初的纯旋转/初始化段），仍
使用 `mode=mono, use_gt_poses=false`。相较从第 0 帧开始的实验，姿态相对误差
最大值降至约 27°、中心 RMSE 降至 0.065 m，Gaussian 增至 362,695，观察表面
IoU/F1@0.10m 提升至 0.034/0.117；但这仍远低于 oracle，说明“等待足够视差
后再启动 DROID”是有效缓解而非完整解决方案。适配 DualVLN 时可以把前 N 帧
作为 warm-up、只缓存 RGB，并在检测到平移/视差后启动 FreeOcc worker；同时
保留姿态质量门控，异常时冻结地图而不是继续写入错误坐标。

补充的 `multiview=false + stride=4` 诊断（同一 20–80 帧窗口）把 Gaussian
数量调整为约 266k，10 cm recall/F1 提高到 0.189/0.209，但 precision 降到
0.235，且姿态最大相对误差约 89.6°。这验证了关闭多视图确实能“变密”，却会
把错误深度和错误姿态一起写入地图；因此工程默认仍应保留多视图过滤，把密度
优化放在关键帧调度、体素合并和真实 metric-depth 上，而不是关闭一致性约束。

## 体素分辨率约定

FreeOcc 当前代码调用 EmbodiedOcc/Occ-ScanNet GT 时固定使用 0.08 m voxel；
InternNav 的 SparseOcc 与候选规划默认使用 0.05 m 二维 cell。因此接入时保留
三个有明确职责的尺度：

- 0.08 m：FreeOcc 原生 3D OCC、与官方 ScanNet/EmbodiedOcc 结果公平比较；
- 0.10 m：跨 episode 的低成本语义长期记忆和可视化；
- 0.05 m：将 3D voxel 投影到现有 BEV 后的候选规划接口。

10 cm voxel 适合辅助判断房间结构、墙/门/家具分布，但不应直接裁决机器人能否
穿过窄通道。投影到 5 cm BEV 时需要按机器人 footprint 加上半个 3D voxel
对角线做保守膨胀，且 SparseOcc 仍是安全权威。实验报告必须同时注明 voxel
size；不同分辨率的 exact IoU 不可直接横向比较。

## RGB-only metric depth 适配

上游 README 截至 commit `f84a0f0` 只完整提供 RGB-D 数据和运行说明。
`configs/slam.yaml` 虽声明 `mono_depth: metric3d-vit_giant2`，但 `run.py` 与
`src/` 没有读取该字段，也没有 Metric3D/ZoeDepth/Depth Anything/Lotus 的安装
或权重说明。原生 `mode=mono` 还会显式丢弃 dataset depth。因此不能通过补一
个 Hydra 参数来启用 metric depth。

本分支新增 `prepare_freeocc_metric_depth.py`，复用 InternNav 已有的
Depth Anything V2 Metric Hypersim Small 模型，把 RGB 预测为毫米 PNG。推理
仍然只输入 RGB；Habitat depth 仅可通过 `--audit-gt-depth` 计算 AbsRel/RMSE/
delta1，不参与预测。生成的数据随后使用 FreeOcc 的成熟 RGB-D 路径，但必须
标记为 `rgb+pseudo-depth`，不能和真实 RGB-D/oracle 混称。

示例：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 python scripts/eval/prepare_freeocc_metric_depth.py \
  --input-dir /path/to/replay_episode \
  --output-dir /path/to/replay_episode_da2_metric \
  --checkpoint checkpoints/depth_anything_v2_metric_hypersim_vits.pth \
  --audit-gt-depth --copy-audit-poses
```

`check_freeocc_official_setup.py` 会统一检查 README 所需配置、checkpoint、Python
包和 CUDA 扩展，并明确报告 `mono_depth` 是否存在 Python consumer。A6000 的
编译架构应为 8.6；README 中的 `TORCH_CUDA_ARCH_LIST=12.0` 不是 A6000 配置。

### audit8：先行完成的 RGB-only 深度/姿态基线（2026-09-04）

按照“先验证输入质量，再决定是否替换前端”的顺序，本轮没有把伪深度接入
FreeOcc，也没有继续运行 MASt3R-SLAM。使用 replay 中保存的 Habitat depth 仅
做离线审计，预测阶段只读取 RGB。审计脚本同时修复了 replay 中同一数字帧同时
存在 `.jpg` 和 `.png` 时的重复计数问题（commit `3fef267`），每个 frame stem
现在只推理一次，优先选用 PNG。

Depth Anything V2 Metric Hypersim Small（输入 518）结果：

| episode / window | frames | AbsRel | RMSE (m) | delta1 | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| SN83 20--80 | 60 | 0.355 | 1.053 | 0.495 | 18.44 |
| 5q7 full | 120 | 0.309 | 1.008 | 0.371 | 18.09 |
| PuK full | 120 | 0.502 | 1.522 | 0.317 | 18.64 |
| r47 full | 107 | 0.453 | 1.283 | 0.199 | 16.70 |

Metric3D ViT-Giant2（`metric_depth_vit_giant2_800k.pth`）结果：

| episode / window | frames | AbsRel | RMSE (m) | delta1 | FPS | peak CUDA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SN83 20--80 | 60 | 0.283 | 0.545 | 0.681 | 0.83 | 7.94 GB |
| 5q7 full | 120 | 0.074 | 0.336 | 0.974 | 0.85 | 7.94 GB |
| PuK full | 120 | 0.102 | 0.529 | 0.952 | 0.85 | 7.94 GB |
| r47 full | 107 | 0.066 | 0.211 | 0.974 | 0.86 | 7.94 GB |

Metric3D 明显优于当前 DA2 Small，但 SN83 原地转向窗口仍有较大的深度误差，
且 Giant2 约 0.85 FPS，不能直接逐步阻塞 DualVLN。它适合作为低频异步深度
worker 或离线消融，不应被误认为已经解决姿态问题。

已有 FreeOcc pose audit 显示，GT depth + DROID pose 在 timestamp 14/26
分别出现约 59/110 度相对旋转跳变；GT depth + GT pose 的最大误差仅约 0.038
度。结合本轮深度结果，当前优先级明确为：先修复原地转向时的 RGB-only pose
跟踪/初始化，再以 Metric3D 作为深度增强对照；在此之前不替换 FreeOcc 前端、
不接管 SparseOcc 安全判断。
