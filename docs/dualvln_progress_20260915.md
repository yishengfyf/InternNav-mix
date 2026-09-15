# DualVLN 新主线进展记录（2026-09-15）

## 本次目标

承接人工证据标注 Pilot，先完成只读数据审计和 relevance/readout probe，再决定是否把标签接入 evidence reader。人工原始文件不覆盖，当前单场景 pilot 不宣称泛化。

## 已完成状态核对

- 隔离分支：`dualvln-spatial-memory-v1`，HEAD `3d14f5d`。
- 人工文件：`/home/siat-xt523/Downloads/vln/dualvln_annotations_annotator_01.jsonl`。
- 40 条记录、20 个 episode、单场景 `17DRP5sb8fy`。
- `need_history`：yes 24、no 15、uncertain 1；24 条 yes 中 16 条选择两张正例。
- 当前没有运行中的 DualVLN 训练/闭环进程。

## 本次代码修改

- 新增 `scripts/dualvln_mainline/validate_annotations.py`：检查枚举互斥、Top-2 限制、因果帧、图像路径、annotator 字段及 candidate manifest 漂移；生成中文 `summary.md`、`metrics.json`、`metrics.svg`。
- 新增 `scripts/dualvln_mainline/analyze_annotations.py`：按 episode 聚合，比较最近帧、最近位姿、较旧帧和随机卡片基线；支持多正例并输出中文结果。

## 决策门槛

1. 校验必须零错误；否则不接训练。
2. probe 仅作为标注质量/可学习性诊断。它不能证明 S2 已理解历史，也不能替代跨场景验证。
3. 只有在 episode 隔离的 probe 中人工选择优于随机且不是单纯位置偏差，才生成 reader 监督转换；`uncertain` 保留为忽略项。
4. 下一步若门槛满足，先做冻结视觉特征的 relevance reader 小 probe，不覆盖主 checkpoint；若不满足，先扩展第二场景或做双人复核。

## 本次运行结果

- 校验报告：`results/dualvln_mainline/codex_latest_return/annotation_validation_20260915/`。
- 校验状态：通过；40/40 条有效，模板字段漂移 0，错误 0，警告 0。
- relevance 报告：`results/dualvln_mainline/codex_latest_return/annotation_relevance_probe_20260915/`。
- 24 条 `need_history=yes`（覆盖 16 个 episode）中，人工允许的多正例总数为 40。
- 最近帧 Top-1 命中 54.2%，最近位姿 54.2%，较旧帧 45.8%，随机卡片基线 55.6%；最近帧比随机低 1.4 个百分点。

## 当前决策

数据格式和因果审计已达到门槛。简单年龄/位姿启发式没有超过随机，这不能推出人工标签不可用；它说明人工选择不能被“越近越好”的位置偏差解释。真正的下一层门槛改为：冻结 InternVLA 视觉特征在 episode 隔离的 relevance probe 中能否稳定高于同卡片随机多正例基线。因此本轮暂不启动 S2/evidence reader 训练、不进入短闭环，也不把自动模型判断混入人工文件。

新增冻结特征两阶段入口：`extract_annotation_features.py` 在单张空闲 GPU 上只做 InternVLA 视觉塔与词嵌入前向，缓存当前/历史 RGB、指令和 pose/age；`probe_annotation_features.py` 在缓存上按 episode 四折、多 seed 比较 `pose_age`、`rgb_text` 与 `rgb_text_pose`。预设门槛为最佳 Top-1 高于随机 5 个百分点且 pairwise accuracy 超过 55%。通过后才实现多正例 reader loss；未通过则先做独立复核或第二场景 pilot。

隐私边界：服务器只读取不含人工答案的 candidate manifest 并生成冻结特征；人工 JSONL 不上传。特征回传本地后，`probe_annotation_features.py` 才按 `annotation_id + candidate label` 关联人工多正例并计算指标。

## 冻结特征结果与优化（提交 `adaa20f`）

- 服务器 GPU 1 完成 40 张卡片、120 个候选的冻结前向；视觉特征维度 3584，峰值显存 16026.95 MiB，未更新模型。
- 特征目录：`annotation_frozen_features_20260915_adaa20f/`；本地 probe 目录：`annotation_frozen_probe_robust_20260915_adaa20f/`。
- 按 episode 四折：位置显示基线 37.5%，pose/age 54.2%，RGB-only 70.8%，RGB+指令 66.7%，RGB+指令+pose 66.7%。
- leave-one-episode-out：RGB-only Top-1 70.8%、pairwise 64.6%；同卡随机 Top-1 为 55.6%。
- RGB-only 精确同卡随机尾概率为 0.083，episode bootstrap 95% 区间为 50.0%--88.5%。

结论：视觉内容信号同时通过四折与 leave-one-episode-out 工程门槛，并明显优于 pose/age；但样本量小，bootstrap 区间仍覆盖随机基线，证据等级记为 `promising_but_underpowered`。下一步允许进入不改 S2 checkpoint 的多正例 reader 机制试验；暂不进入闭环或宣称跨场景有效。由于 RGB+指令没有优于 RGB-only，首个 reader 试验必须保留 RGB-only 对照，不能把收益归因于任务条件化。

多正例 pairwise 机制复验中，RGB-only 的四折与 leave-one-episode-out Top-1 均为 66.7%，pairwise accuracy 均为 70.8%；对应 pose/age 为 62.5%/60.4% 和 58.3%/52.1%。工程门槛通过，但精确随机尾概率 0.174、bootstrap 区间 45.8%--85.7%，仍保持 `promising_but_underpowered`。

据此新增 evidence reader 监督协议：每条历史 target 为正例 1、负例 0、忽略 -1；一张卡片有多个正例时最大化落在整个正例集合上的平均 attention mass，不强制唯一 Top-1；`need_history=no` 时将 null evidence 作为正目标；`uncertain` 全部忽略。该 loss 独立输出为 `evidence_relevance_loss`，以显式权重接入总 loss，并保留 S2/trajectory 原始指标。

本地已生成最小监督 manifest，共 40 条、9129 字节，SHA256 为 `50f4e278aa36f8db8cf77c34892bcc55ea57a3eae49e2678a030a286ef8b59ed`。文件不包含指令、图像路径、理由、标注者或时间，只保留卡片/候选 ID 与 `1/0/-1` target；它仍属于未公开研究监督数据，上传服务器前需要针对该具体文件的明确批准。

新增 `train_annotation_reader.py`：直接读取冻结 3584 维视觉特征，以 4 折 episode 隔离、seed 23/47/71 比较 content、spatial、task-spatial；只训练 128 维轻量 memory/estimator，不加载或更新 S2 checkpoint。输出验证 relevance loss、正例质量、yes Top-1、pairwise 和 no/null accuracy，并保留 JSON、中文摘要、SVG 与逐运行曲线。
