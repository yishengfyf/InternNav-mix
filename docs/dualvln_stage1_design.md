# DualVLN 第一阶段：任务条件化空间证据记忆

状态：实施设计，不是实验结果。

基线：分支 `dualvln-spatial-memory-v1`，来源为
`upstream/main@7a5c62400ac45b313d9b709c740b64191556a242`。

## 1. 研究问题与范围

第一阶段检验：使用估计的任务状态对历史读取进行条件化后，因果空间历史是否更有价值。实验必须将该作用与继续训练、增加历史图像、仅增加空间元数据等解释分开。

首版保留 S2 原生输出，包括像素目标、转向、向下观察和 STOP；同时保留 S1 原生 NextDiT 轨迹接口。完整 OCC、独立恢复代理、新 backbone 和可靠性学习不在本阶段同时引入。

## 2. 当前代码约束

- `NavPixelGoalDataset` 从因果历史前缀中均匀选取最多 `num_history` 张 RGB，但目前只把它们作为普通图像列表交给 S2，没有返回历史帧编号、相对位姿、年龄或质量。
- `DataCollatorForSupervisedDataset` 在序列尾部追加 4 个轨迹 token，并用 `t_s_pos` 记录起点。插入 evidence token 时必须同步更新截断、labels、attention mask、position IDs 和 `t_s_pos`。
- `InternVLAN1ForCausalLM.forward` 替换图像 token 和轨迹 token 的 embedding，再读取 `t_s_pos:t_s_pos+n_query` 的 S2 hidden states 作为 NextDiT 条件。
- 当前 `forward` 在存在轨迹标签时计算 trajectory MSE，但不计算 S2 输出的 token 交叉熵。第一阶段必须分别输出 `s2_loss`、`trajectory_loss` 及其加权总 loss。
- `generate_latents` 会在生成后重复执行 S2。训练 forward、生成 prefill 和 latent 提取必须共享同一 evidence 注入逻辑，只修改其中一条路径无效。
- 当前 `set_model` 通过宽泛 substring 解冻 NextDiT、RGB、resampler、projector 和 latent query。第一阶段改为失败即关闭的显式参数白名单，并报告准确的可训练参数量。
- policy 虽然声明 `pose_list`，但没有随 RGB 历史同步写入 pose。在线 evidence store 必须原子地更新 RGB、pose、时间和质量。

## 3. Evidence 接口协议

所有运行时字段必须在当前决策时刻或之前可获得；仅训练可用的标签与运行时输入严格分离。

|字段|形状|含义|
|---|---|---|
|`history_features`|`B x N x Dv`|因果历史观测的池化特征|
|`relative_poses`|`B x N x 4`|当前坐标系下的 `dx`、`dy`、`sin(dyaw)`、`cos(dyaw)`|
|`ages`|`B x N x 1`|非负决策步年龄|
|`qualities`|`B x N x 2`|可观测的位姿/测量质量，不包含 GT 成功信号|
|`valid_mask`|`B x N`|真实条目与 padding 的区分|
|`task_state`|`B x Dt`|由指令、当前观测和因果历史估计的任务状态|
|输出 `tokens`|`B x K x H`|固定数量、位于 S2 hidden space 的 evidence tokens|

writer 在不读取任务状态的前提下组合观测特征和空间元数据。reader 将 task-state query 加到可学习 evidence queries 上，再对历史条目做 cross-attention。始终有效的 null evidence 使空历史输入有明确定义。模块同时返回历史读取权重、null 权重、阶段 logits 和空历史标记，供审计使用。

为避免把 S2 的 `3584` 维 hidden size 直接用作 Transformer reader 宽度，首版固定采用 `3584 -> 512` 的 memory bottleneck，并将 reader 输出经独立线性层投影回 `3584` 维 S2 token 空间。按 `task_dim=512`、4 个 evidence token 和 8 个 attention heads 计算，memory 模块约 730 万参数；单元测试设置 700--800 万参数门槛，防止配置回退为约两亿参数的宽 reader。

训练 LeRobot pose 与在线仿真 observation 统一编码为当前机器人平面坐标系下的 `[dx, dy, sin(dyaw), cos(dyaw)]`。在线 `globalgps/globalrotation` 属于仿真理想位姿/理想里程计条件，后续与估计里程计、纯视觉条件分开报告，不将它描述为真实部署必然可得输入。

task-state 训练监督可以来自训练标注或经过 mask 的软伪标签，但推理 API 不接受 GT progress。有效 evidence 的年龄不得为负；调用方必须保证输入仅来自因果历史前缀。

## 4. S2 接入顺序

1. 按每张历史图像的 `image_grid_thw` 池化视觉 token，不能把当前图像 token 写入历史存储。
2. 使用指令 embedding、当前观测池化特征和上一时刻递归状态估计 `task_state`；初始状态可学习，并在 episode 开始时重置。
3. 由 `TaskConditionedEvidenceMemory` 读取固定 `K` 个 evidence tokens。
4. 在训练与推理 prompt 的当前观测前插入 `K` 个注册的 evidence placeholder，再用读取结果替换其 embedding。
5. embedding 替换和序列 bookkeeping 只能有一套实现，并同时用于训练 forward、生成 prefill 和 `generate_latents`。
6. evidence label 固定为 `IGNORE_INDEX`。插入后重新计算多模态 RoPE，逐样本断言 image/evidence/traj token 数量；不能根据 batch padding 后的偏移猜测 evidence 位置。
7. 分别计算带 mask 的 S2 token CE 与 trajectory flow-matching loss；记录 latent 均值、标准差、范数及相对冻结基线的余弦偏移。

选择显式 placeholder 而不是直接追加匿名 embedding，是因为 Qwen 多模态 RoPE 和 generation cache 都依赖对齐的显式序列；该方案也便于测试训练与推理的一致性。

## 5. 首轮可训练参数

默认首轮只训练：

- evidence writer、任务状态估计器、条件化 reader、阶段头和 evidence-to-S2 投影；
- `peft` 环境具备后启用的 S2 LoRA 参数；
- 作为初始 latent bridge 的 latent queries 与 `cond_projector`。

视觉塔、RGB model、NextDiT action encoder/decoder 和 NextDiT transformer 首轮冻结。梯度仍通过冻结的 NextDiT 回传给 S2 bridge。只有当固定目标 S1 对照正常、完整 S2-to-S1 轨迹却退化时，才从最小 bridge 组件开始扩大训练范围。

## 6. 公平对照

|编号|历史路径|空间元数据|任务条件化读取|回答的问题|
|---|---|---|---|---|
|B0|checkpoint 默认路径|否|否|干净基线能否复现|
|B1|checkpoint 默认路径|否|否|收益是否只是继续训练|
|B2|等量 `K` 个历史 token|否|否|收益是否只是更多上下文/token|
|M1|evidence memory|是|否|空间关系是否有独立价值|
|M2|evidence memory|是|是|任务条件化是否有独立价值|

B1/B2/M1/M2 使用相同的新训练数据、更新步数、历史帧、token 数量、优化预算和随机种子。仅用于验证的历史 audit 数据不得转成训练样本。

## 7. 验证门槛

1. CPU 协议测试：shape、padding 不变性、空历史、因果年龄、任务条件化、梯度、显式训练白名单及错误白名单拒绝。
2. Dataset fixture：证明 frame id 和元数据只来自前缀，且训练/推理历史选择器一致。
3. 小模型集成：证明 evidence、image、labels、RoPE 和 `t_s_pos` 始终对齐。
4. 在 8--32 个经授权训练样本上过拟合：S2 与 trajectory loss 均下降；evidence/bridge 梯度非零，冻结参数无梯度。
5. 在 1--4 个开发 episode 上做短闭环 smoke；GPU 空闲后再做 200--500 update profiling。

只有在训练数据来源与 split manifest 固定、隔离环境依赖齐全、共享 GPU 可用后，才启动更大规模运行。

## 8. 阶段结果输出约定

每个自动实验阶段无论成功、失败还是提前停止，都必须保留可追溯的结束报告：

- `metrics.json`：机器可读的核心指标、运行状态、commit、退出码和耗时；
- `summary.md`：中文简述、关键指标表和一段边界明确的结果分析；
- 至少一个可视化文件，默认使用无需额外依赖的 `metrics.svg`，训练阶段再增加 loss、成功率、SPL、延迟或显存曲线；
- 原始日志、配置、数据/checkpoint manifest 与源码哈希；
- 终端末尾打印 3--8 行关键指标及简短分析，便于自动任务结束时直接检查。

报告中的“通过”只表示当前阶段预先定义的门槛通过，不等于方法有效或论文假设成立。失败报告同样保留，禁止只回传成功运行。
