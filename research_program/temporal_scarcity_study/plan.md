# Temporal scarcity communication study

本轮实验把研究问题收窄为：在没有语言先验的两个学习体之间，哪些非语言能力已经足以使离散共同符号产生因果作用；持续的资源稀缺、私有信息和记忆又会如何改变符号系统的形成与使用。

## 任务

每个 episode 有两个资源地点和五个回合。每个地点有一种二值资源类型，两个地点独立采样。A 只能直接观察地点 0，B 只能直接观察地点 1；每个 agent 在每个回合有一个私有需求类型。动作是等待、取地点 0 或取地点 1。地点库存贯穿整个 episode，容量为 1（`scarce`）或 3（`abundant`）。同一地点的同时请求共享库存；只剩一个单位时，低索引 agent 获得该单位，另一方承担失败成本。匹配需求且库存可用得 +1，错误或耗尽仍请求得 −0.25，等待得 0。训练目标是两名 agent 五回合奖励的平均值。

通信通道是 8 个无意义离散 token。token 只能在第 0、1 回合发送；接收方在下一回合读到该 token，随后进入 blackout。`silent` 条件完全丢弃 token；`live` 条件传递 token。`permuted` 评估把同一批次内的 token 重新配对，用来区分 token 本身与发送者—接收者的因果对应关系。

## Agent 与因素

Agent 是从随机参数开始的 NumPy tanh policy，不使用语言模型、词表、教师信号或预训练语言知识。每个 agent 有独立的消息头（8 类）和动作头（3 类）。`stateless` 条件把 recurrent edge 置零；`recurrent` 条件保留 32 维 hidden state。因而记忆条件只改变跨回合状态保持，不改变其它初始权重。`PI` 只暴露本人的当前需求、本人所在地点的类型和库存、上一动作结果及时间；`FI` 另外暴露伙伴当前需求、远端地点类型和库存。

冻结的全因子设计为：

* memory：`stateless` / `recurrent`；
* scarcity：`scarce`（1）/ `abundant`（3）；
* information：`PI` / `FI`；
* channel：`silent` / `live`；
* 8 个 seed（68101–68108）。

训练世界在不同库存容量之间完全配对：容量不进入 episode RNG key，因此 scarcity 的差异只来自资源后果。stateless/recurrent 共享除 `W_h` 是否启用之外的初始化；live/silent 共享世界、需求序列、动作和消息抽样 uniform。需求序列使用四个训练转移模式，评估使用另外四个留出模式；资源地点边际分布保持不变。

## 主要指标

1. `team_return_mean`、标准差和正收益 episode 比率；
2. 由完全可见信息的有限时域中央 oracle 计算的 `oracle_team_return_mean`，以及在 oracle 可达 episode 上的归一化收益和 regret；
3. live 的自然通信、关闭通道和 token 置换之间的性能差：只有自然通信同时保留 token 与发送者—接收者配对；
4. token 与发送者私有需求、发送者本地资源类型的互信息；token 熵、token 使用频率和 action histogram；
5. 训练轨迹的收敛速度、跨 seed 方差、参数/episode stream hash 和 checkpoint 一致性。

共同符号的最小操作性判据是：在相同最终参数和相同评估世界上，`live/natural` 的收益显著高于 `closed`，且 `permuted` 的收益下降；与此同时 token 对私有需求或资源状态具有稳定的互信息。只有 token 熵下降或性能上升而没有这两个因果证据时，不称为通信形成。

## 分阶段执行

1. **烟雾测试**：1 个 seed、4 个代表性条件、20 updates，检查消息时序、PI 遮蔽、容量后果、梯度有限性和 oracle 上界。
2. **先导网格**：2 个 seed、16 条件、200 updates，确定奖励是否可学习、live 是否产生可检验的通信效应，并调整正式实验的训练长度。
3. **正式网格**：8 个 seed、16 条件、2000 updates，保存 0/500/1000/2000 checkpoint；独立审计重放配对世界、哈希、消息控制和 oracle 计算。
4. **条件复杂化**：仅在正式网格出现稳定的 live–closed/permuted 差异后，再增加伙伴轮换、三人竞争或更长 blackout；每次只引入一个新压力并保留本轮固定基线。

## 可证伪预测与边界

* 如果私有信息和稀缺库存是共同符号的压力来源，`PI + scarce + live` 应比对应 silent/closed 条件更依赖 token；`FI` 或 `abundant` 会削弱这一差异。
* 如果记忆是形成跨回合约定的必要能力，`recurrent` 的 token 互信息、留出模式收益和置换敏感度应高于 `stateless`；若差异不存在，记忆不是该任务中的必要条件。
* 若 live 的优势在关闭和置换控制中都保持，优势更可能来自动作探索或初始化偶然性，而不是共同通信。

本实验研究的是可控的人工学习系统中的“共同符号的操作性起点”，不把离散 token 直接等同于人类语言，也不声称重建人类语言史。后续若要连接视觉模型或开源多模态模型，必须先在本任务上复现实验的因果指标，再替换感知模块。
