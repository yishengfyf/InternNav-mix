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

在 N0 已确认原始 input-token evidence 无有效梯度后，N2 首轮统一采用冻结 Qwen 的 late-adapter 诊断路径，具体实现固定为：

- B0：不插入 evidence、不更新参数，只记录原 checkpoint 在固定样本上的损失；
- B1：只保留可学习 null evidence，屏蔽真实历史、空间元数据与 task-state，作为无历史的适配器对照；
- B2：读取真实历史视觉内容，但将位姿、年龄、质量和 task-state 置零；
- M1：读取历史视觉内容和空间元数据，但将 task-state 置零；
- M2：读取历史视觉内容、空间元数据，并使用由当前视觉和因果 prompt 估计的 task-state。

四个可训练组均使用相同的 4 个 evidence token、cross-attention late adapter、初始化、seed、学习率、样本顺序和更新数。B1 的含义相应收窄为“无历史的冻结特征适配器”，不能描述成完整 Qwen 继续训练。该对照首先回答训练信号与小样本拟合差异，不直接等价于未见场景导航收益。

训练 forward 和 `generate_latents` 必须调用同一 late evidence adapter。特别是 cross-attention 模式下，推理不得退化为 evidence 均值，否则闭环结果不属于训练时验证的模型。

### 6.1 多 seed 与 task-state 优化门槛

单次 seed 的 loss 下降百分比不能决定是否进入闭环，因为不同消融的随机初始化会改变初始 loss。N2 稳定性阶段固定使用 seed `23/47/71`，以最终 loss 的跨 seed 均值和逐 seed 一致性判断：

- M2 平均最终总 loss 与 S2 loss 均不得超过 B2 的 `101%`；
- M2 平均最终 trajectory loss 不得超过 B2 的 `105%`；
- 至少 `2/3` seed 的 M2 最终总 loss 不超过对应 B2 的 `101%`；
- 所有运行必须梯度 finite、冻结参数无梯度并完成预定更新数。

若原 M2 未通过，先在 seed 23 筛选最小优化干预：task-state `0.1` 缩放、task-state 单位范数、estimator `0.1x` 学习率。按最终总 loss 选择一项，再在 seed 47/71 复验；不并行叠加多个改动，以保持归因清晰。只有复验通过上述门槛，才允许进入 1--4 episode 短闭环。

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

## 9. 短闭环归因协议

首个 B0/M2-lr01 配对 episode 只用于打通真实闭环。两组均未成功，M2 的终点导航误差由 `7.514 m` 降至 `4.613 m`，但 M2 在第 0 步尚无历史时已经输出与 B0 不同的动作。这说明行为差异可能来自 null evidence、late adapter、latent queries 或继续训练，而不能直接归因于空间历史和 task-state。

后续 4-episode smoke 固定比较六组：

|组别|权重与推理设置|主要归因|
|---|---|---|
|B0|原 checkpoint，无 evidence|原始导航基线|
|B1|B1 权重，null ablation|adapter/继续训练本身|
|B2|B2 权重，content ablation|历史 RGB 内容|
|M1|M1 权重，spatial ablation|相对位姿、年龄和质量|
|M2Z|M2-lr01 权重，推理时 task-state 置零|同权重下移除 task query 的因果对照|
|M2|M2-lr01 权重，task-spatial ablation|完整原型|

所有组运行排序后完全相同的 R2R val-unseen episodes，并保留每次 evidence 调用的有效历史数、task-state 范数、stage logits、read weights 和 null weights。read/null 权重用于确认是否真实读取历史；stage logits 在没有可靠阶段标签前只作为分布探针，不报告阶段准确率。4 个 episode 仍只用于接口、行为方向和归因诊断，不能作为 val-unseen 泛化结果。

当前闭环输入协议明确标为：S2 使用当前/历史 RGB、指令和 Habitat GPS/compass 派生的理想相对位姿；S1/局部执行继续使用 RGB-D 与 pose。因而该版本属于 `RGB+Pose / RGB-D+Pose` 系统，不属于严格 single-RGB-only。纯 RGB、RGB+pose、RGB-D+pose 的正式分协议对照在闭环归因稳定后单独实施。

## 10. Task-state 语义与指令敏感性探针

4-episode 归因矩阵中，M2 与同权重、推理时将 task-state 置零的 M2Z 动作和导航指标完全相同；同时历史读取归一化熵接近 1。扩大闭环前，先在 R2R train 上运行不更新参数的 task-state 探针，避免把一个存在的张量误称为已经形成的任务阶段推理。

探针固定使用 M2-lr01 seed 23 权重和 32 个真实因果样本。样本来自 32 个不同 episode，在路线帧进度四分位中各取 8 个；四折划分在各区间内均衡。对每个相同图像与历史输入分别运行原指令和来自其他样本的确定性错配指令，并记录：

- 原/错配指令 task-state 余弦和相对 L2 变化；
- memory read weights 的平均 L1 变化和 stage argmax 翻转率；
- stage logits 与粗进度四分位的最佳置换准确率；
- 每折只在训练样本上拟合的 8 维 PCA-ridge 进度预测 MAE，以及训练折均值常数基线 MAE。

32 个样本只构成单 scene 的低成本诊断，不构成语义证明。路线帧进度也不是自然语言子任务阶段真值：即使可解码，也可能来自当前视觉、历史长度或位置线索；只有原/错配指令对照可以提供有限的指令敏感性证据。若 task-state 余弦大于 `0.995`、读取权重 L1 变化小于 `0.01` 且 stage 翻转率小于 `0.1`，判为对指令近乎不敏感。若进度探针不优于常数基线的 `95%` 且 stage 最佳置换准确率不超过 `0.40`，判为未观察到粗进度结构。

探针较弱时不再重复扩大现有闭环，而转向显式但推理时可获得的状态学习：先设计因果进度或指令子句对齐监督及对应视觉-only 对照，再重训 task-state。探针较强时也只进入跨 episode/scene 复验和视觉-only 对照，不直接宣称模型理解任务阶段。

## 11. Task-state 修复实验

首轮探针满足“对指令近乎不敏感”的判据：原/错配指令 task-state 余弦为 `0.999996`，读取权重 L1 变化为 `0.000832`，stage argmax 翻转率为 0。修复首先针对两个可区分原因：固定 prompt 平均造成指令稀释，以及原生 S2/trajectory loss 没有迫使 task-state 保存任务身份。

输入侧新增独立的 instruction token ids/mask。task-state estimator 只池化原始导航指令，不再把固定系统模板、输出要求和图像占位符共同平均；训练、generation prefill 与 `generate_latents` 使用同一结构化字段。旧 adapter 默认保持原协议，新实验通过显式 config 开关启用，禁止静默改变已有闭环复现路径。

监督侧分两项：

- 指令错配分离：当前转换 shard 每条路线只有一条可核验指令，不能虚构同路线复述。正项因此只使用同一结构化指令的一致状态，负项使用不同路线指令；margin 只要求相同视觉下的错配指令 task-state 与原状态分离。该目标只检验模型是否保留路线级指令差异，不等同于语义匹配或阶段理解。未来只有恢复原始 R2R `path_id` 到多条人工指令的无歧义映射后，才升级为真正的复述正样本。
- 软进度代理：训练时根据参考路线中当前因果帧的位置，对四个进度锚点生成相邻插值软标签并监督现有 stage head。该标签不进入推理，且始终称为 route-progress proxy，不称为自然语言子任务阶段。

固定 32 样本、seed 23 和相同更新预算依次比较：R0（instruction-only，无新辅助 loss）、R1（R0 + 任务身份对比）、R2（R1 + 软进度代理）。先要求所有梯度 finite、冻结参数无梯度且 S2/trajectory 不劣于已有 B2 工程容差；再用独立错配指令探针要求 task-state、read weights 或同权重 M2Z/M2 行为出现可重复差异。若 R1 不能产生指令敏感性，停止增加进度 loss，回到表示与负样本设计；若 R1 成立而 R2 只增加位置可解码性，则保留两者分离表述，不把进度代理包装成语义收益。

首轮 8 样本机制筛选中，R1 虽增大 estimator 输出差异但 read L1 仍低于 `0.001`，R2 的训练进度 loss 下降却未改善独立进度 probe。后续单一修复 R1Q 将错配 margin 施加在 `evidence_memory.task_projection(task_state)` 的实际 reader query 空间，使梯度同时约束 estimator 与 reader 投影；R1Q 不启用进度代理。若它仍不能改变读取，则停止用无相关性标签强迫任意历史选择，转向恢复原始多指令/路径对齐数据或更可靠的 evidence relevance 标注。

## 12. 官方路径复述监督 R1P

官方 R2R train 标注与转换后的 `tasks.jsonl` 已通过指令文本唯一匹配：当前 scene 的 75 条转换指令全部可映射到 25 个官方 `trajectory_id`，每条路径有 3 条人工指令。该映射必须由独立 manifest 固化原始文件 SHA256、匹配数、歧义数和每条路径的复述集合；dataset 仅接受状态为 `passed` 的 schema-v1 manifest，缺失、重复或复述集合不完整时立即停止。

R1P 使用同一视觉与历史输入构造三元组：anchor 为转换样本的原指令，positive 为同一官方 `trajectory_id` 的另一条人工指令，negative 为不同 `trajectory_id` 的指令。监督继续施加在 memory 实际使用的 `task_projection` query 空间。它只约束路线级任务身份，不提供当前任务阶段或哪一帧历史最相关的真值。

公平对照固定为 R0P 与 R1P：两组使用相同 8 条官方路径、seed、初始化、80 steps、instruction-only task-state 与 task-state `0.1x` 学习率；R1P 唯一增加路径复述对比 loss。训练后各自在 32 个真实因果样本上同时测量同路径复述变化和异路径指令变化。进入闭环必须同时满足：梯度有限且冻结参数无梯度；R1P 的 S2/trajectory 最终损失不超过 R0P 的 `105%/110%`；异路径 task cosine `<0.995`、relative L2 `>0.05`，且 read L1 `>=0.01` 或 stage 翻转 `>=0.1`；同路径复述必须在 task-state 距离和 read L1 上均比异路径更接近。任一条件失败即停止继续叠加 task-state 辅助损失，转向 evidence relevance 或同地点不同阶段监督。
