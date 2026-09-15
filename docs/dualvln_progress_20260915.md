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

## Task-state 稳定化消融

固定平衡四折、seed 23/47/71、20 steps 和学习率 3e-4 后，原 task-spatial 的验证 loss 为 1.7997、跨运行标准差 0.3822、训练/验证 gap 1.6897。task 分支 0.1 倍学习率将验证 loss 降至 1.4634，冻结 estimator 降至 1.5483，但仍保留明显过拟合。task-state 单位范数将验证 loss 降至 0.9953、标准差降至 0.0923、gap 降至 0.4658；yes Top-1、pairwise、no/null 分别为 73.6%、68.8%、68.8%。归一化再叠加 0.1 倍 task 学习率的验证 loss 为 1.0076，没有进一步收益。

同预算 content/spatial 的验证 loss 分别为 1.0304/1.0281，yes Top-1 均为 69.4%，pairwise 为 65.3%/66.0%，no/null 为 60.4%/62.5%。因此“仅归一化 task-state”在当前单场景 40 卡片上通过了不劣于普通历史 reader 的机制门槛，主要修复的是 task query 尺度失衡。该证据仍不足以宣称跨场景或导航收益。

现有闭环脚本加载的是更早的完整训练权重，不会加载本轮 reader-only 参数，不能直接用于本轮结论。下一步先把 `evidence_normalize_task_state=true` 和人工 relevance targets 接入可保存 adapter 的正式 S2 小样本训练；先通过 relevance、S2、trajectory 三类损失和冻结参数审计，再进入 1--4 episode 配对短闭环。

## Reader-only 矩阵与预算扫描

初始按 episode ID 取模的 120-step 矩阵显示：content/spatial/task-spatial 的训练 loss 分别约 0.115/0.109/0.0001，而验证 loss 为 2.691/2.511/3.612，确认明显记忆训练集；各折 yes/no 从 5/5 到 8/1，也使 no/null accuracy 从 100% 到 0% 波动。

随后改为确定性的标签分层 episode 四折：每折 yes/no/uncertain 分别为 6/4/0、6/3/1、6/4/0、6/4/0。固定其他条件扫描后：

- 20 steps：content/spatial/task-spatial 验证 loss 1.030/1.028/1.800；
- 40 steps：1.143/1.122/2.281；
- 80 steps：1.989/1.877/2.666。

因此 20 steps 是唯一使 content/spatial 验证 loss 低于初始约 1.10 的预算。task-spatial 在 20 steps 时训练 loss 已约 0.11、验证 loss 却为 1.80，说明 task 分支容量过强。下一轮固定 20 steps，只做 task 参数 0.1 倍学习率、task-state 单位范数、冻结 estimator 三项单变量实验。

## 集成 Reader 诊断、早停扫描与多场景扩展

正式联合训练 adapter 的逐卡读出显示：8 个训练和 8 个 episode-held-out 样本的 Yes 全局 Top-1 均为 0，pairwise 分别为 40.0%/35.7%，而 No/null Top-1 均为 100%。历史权重归一化熵约 0.94，三张历史之间近似均匀；此前 relevance loss 的下降主要来自 `need_history=no` 卡片的 null 偏置，不能称为学会选证据。结果见 `annotation_reader_eval_joint_n20_v2_20260915_38ff89a/`。

为区分容量不足和数据不足，新增显式 loss 权重及 adapter 暖启动接口，并做相同 8+8 episode 隔离的 relevance-only 扫描：

|预算|训练 relevance|held-out relevance|判定|
|---:|---:|---:|---|
|20 steps|`1.1259 -> 0.9376`|`1.2104 -> 1.1007`|仅有早期方向性信号|
|40 steps|`1.1260 -> 0.1419`|`1.2104 -> 2.8336`|过拟合|
|80 steps|`1.1260 -> 0.000088`|`1.2104 -> 6.7051`|严重记忆训练集|

20 步 adapter 的 held-out 历史内部 Top-1 为 75%（逐卡随机期望 50%），但含 null 的全局 Top-1 仍为 0，pairwise 为 35.7%，历史需求二分类为 50%。40/80 步训练拟合继续增强但 held-out 快速恶化，因此不再在单场景 8 样本上增加训练步数，也不从该 adapter 启动新闭环。训练入口的未来通过门槛已修正为：存在 held-out relevance 时必须同时改善，避免只凭训练集下降误判。

原联合 adapter 的 4-episode 归因矩阵见 `closed_loop_matrix_20260915T095005Z_38ff89ad714f/`：B0/B1/B2/M1/M2Z/M2 的 SR/SPL 全为 0；B1 后的 NE 变化仍不能归因于历史，M2 与 M2Z 没有可辨识任务条件化增量。当前停止扩大闭环。

服务器随后从官方 `InternRobotics/InternData-N1` 选择性下载并核验 6 个较小 R2R train 场景，共 2,471,101,485 字节、90 个 episode；LFS 大小/SHA256、安全解包均通过，报告为 `multiscene_bootstrap_20260915_38ff89a/`。自动挖掘和合并得到第二批 104 张因果卡片，覆盖 6 场景、52 episode，ID 冲突、未来泄漏和缺图均为 0，目录为 `annotation_multiscene_candidates_20260915_38ff89a/merged/`。

多场景协议已改用 `(scene_id, episode_id, current_frame_id)` 唯一键，旧单场景 manifest 保持兼容；相关 evidence、dataset、推理和候选测试为 `49 passed`。第二批当前只有候选、没有自动伪标签。下一门槛是完成跨场景标注与独立复核，再按 scene-held-out 而非仅 episode-held-out 训练 reader；通过后才恢复联合 S2 训练和 1--4 episode 配对闭环。

## 多场景候选 v3 与模型盲审

提交 `87a6c36` 固化了 scene-aware relevance 协议、联合 reader 诊断、闭环归因、多场景准备工具和 held-out relevance 门槛；服务器隔离 worktree 的相关回归为 `59 passed`。提交 `96fa690` 进一步增加候选帧黑区质量过滤，候选选择测试为 `7 passed`。旧实验 worktree 保持原样，新建干净 detached worktree 运行本轮工具，未修改服务器正式工作树。

对第二批 104 张旧候选进行 18 张 annotation-blind 模型复核，结果为 yes/no/uncertain=`1/10/7`。这说明原时间分层候选仍以同一路段近邻视角为主，任务相关历史密度偏低。模型复核文件明确标记 `reviewer_id=model_reviewer` 和 `training_supervision_allowed=false`，只用于候选筛查，不是第二位人工标注，也不得转为训练监督。

候选器随后改为“近邻连续帧 + 跨朝向/动作阶段帧 + 有效位移锚点”，并拒绝黑像素占比超过 25% 的当前或历史画面。相同 6 场景、104 张、52 episode 的 v3 与旧版对比如下：

|指标|旧时间分层|v3 阶段/空间分层|
|---|---:|---:|
|历史平均年龄|7.35 帧|12.34 帧|
|历史平均位移|1.09 m|1.59 m|
|位移至少 2 m|9.3%|35.6%|
|黑区超过 25% 的历史候选|12|0|
|同一 18 张盲审 yes/no/uncertain|1/10/7|6/7/5|

完整对比见 `annotation_candidate_strategy_comparison_20260915/`。盲审改善只证明候选更值得人工判断，不能替代人工真值。`XcA2TqTSSAj` 仍存在无法由黑区规则捕获的白墙和几何破损，因此不进入首轮人工批次。最终 curated 包为 `annotation_multiscene_candidates_v3_20260915_96fa690/curated/`：5 个新场景、84 张、42 episode、因果违规 0；与既有 40 张人工 pilot 合计 124 张。

当前停止条件是人工监督而非算力：完成 curated 包人工填写后，先校验 schema/因果性并抽取 20--30% 独立人工复核，再生成 scene-aware relevance manifest 和 scene-held-out reader 对照。只有跨场景 reader 明显超过逐卡随机、Yes 历史选择不塌缩且 No/null 正常，才恢复联合 S2 小样本训练；之后才允许 1--4 episode 配对闭环。
