# 新接收者适应与协议传递：结果报告

## 研究目的

前几轮实验已经表明，live 消息通道可以提高严格互选任务中的物理执行率，但固定策略上的后验探针还不能说明新主体是否能获得既有协议。本轮把“传递”作为实验单位：从已经训练完成的三主体群体中冻结 A、B，删除 C 的全部参数，放入同架构、随机初始化的新 C，比较它在任务反馈下的适应过程。

核心问题是：

> 当既有主体保持不变时，一个没有协议参数的新主体能否通过任务反馈恢复协调行为？如果能，跨主体消息通道是否是必要条件？

这仍是任务协议传递实验，不是人类语言起源实验。实验中的源策略和新策略都是本地 NumPy float64 tanh MLP，没有调用 LLM、预训练语言模型、视觉模型、外部 API 或网络服务。

## 冻结设计

- 源实验为 research_program/triadic_factorized_neutral_altpartner_study/results/altpair_001。
- 源策略使用 PL 观察：每个主体看到自己的需求、公共布局和地点所有者，不看到其他主体的私有需求。
- 每个世界有两个 full-success 方案，且两方案使用不同搭档对；行动由 neutral/engage 意图头和 16 类 proposal 头组成。
- 选择源种子 66701–66708，共 8 个独立初始参数块。每个种子分别使用 static 和 rematched 源检查点。
- 冻结源主体 A、B 的六个模块；删除源主体 C 的三个模块并重新初始化。
- 新 C 的三个模块为 54→64→64→32、153→64→64→32 和 252→64→64→18 的本地 tanh MLP。
- live 让三个主体看到三方消息；silent 只保留每个主体自己的消息。两条件共享源检查点、新 C 初始化、世界均匀数、批索引、消息均匀数和 rematching 分配。
- 每个运行 6,000 个更新，批量大小 256；A/B 每步得到零梯度，并在每个检查点做参数哈希核验。
- 固定监控集为 1,560 个需求 × 2 个训练布局 × 6 个地点所有者 = 18,720 个世界，在 0、100、500、1,500、3,000、6,000 更新评估。最终集为 1,560 个需求 × 6 个新布局 × 6 个所有者 = 56,160 个世界。

主要读数预先固定为：

1. 最终新布局上的 live − silent 团队 Q 率；
2. 六个检查点上 live − silent 监控 Q 率相对于初始差异的中心化时间 AUC。

统计单位是源种子。static 与 rematched 先在每个种子内平均，再作为一个总体估计；不能把不同 schedule 当成额外独立社会。

## 结果

最终新布局的团队 Q 率如下：

| 源 schedule | 新 C live | 新 C silent | live − silent |
|---|---:|---:|---:|
| static | 33.42% | 11.30% | +22.12 pp，95% CI [+16.82,+27.42] |
| rematched | 29.39% | 10.31% | +19.08 pp，95% CI [+11.05,+27.10] |
| 两 schedule 在种子内平均 | — | — | +20.60 pp，95% CI [+14.55,+26.65] |

监控 Q 差的总体中心化时间 AUC 为 **+20.07 pp**，95% t(7) 区间为 **[+18.03,+22.12]**，8/8 个种子为正。总体 live − silent 差从更新 0 的 **+1.85 pp**（CI [−2.14,+5.84]）上升到：

| 更新 | Q 差（pp） | 95% CI（pp） |
|---:|---:|---:|
| 100 | +9.31 | [+0.33,+18.28] |
| 500 | +21.15 | [+15.33,+26.98] |
| 1,500 | +21.95 | [+16.91,+27.00] |
| 3,000 | +22.94 | [+18.29,+27.60] |
| 6,000 | +23.51 | [+18.86,+28.15] |

Q 差主要由“是否形成可执行的物理互选”造成。最终 live − silent 的物理执行率差为 static **+57.58 pp**、rematched **+51.39 pp**；而已经发生物理执行后，Q 的条件差为 static **−7.12 pp**、rematched **−7.55 pp**。目标搭档合法率差为 static **−14.21 pp**、rematched **−12.92 pp**，proposal 合法率差为 **+12.99/+12.28 pp**。engagement 率的差异为 0。条件比例的下降不能解释为“错误词义”，因为其分母只包含已经执行的世界，且 live 与 silent 的执行分母变化很大。

为了判断新主体是否恢复源协议，另行保留了同一源检查点、同一路由下的旧 C 评估。旧 C 的新布局 live Q 均值为 static **45.14%**、rematched **44.81%**，而适应后的新 C 分别为 **33.42%** 和 **29.39%**。因此本轮得到的是稳定的路由适应优势，而不是完整恢复源主体协议的证据。

## 结论边界

本轮支持三个窄结论：

1. 在冻结的搭档和任务生态中，新 C 的适应明显依赖跨主体消息路由；同样的任务反馈在 silent 条件下只能得到约 10–11% 的最终 Q，而 live 条件约 29–33%。
2. 这个优势主要体现在 A/B 与新 C 更容易形成可执行的互选和合法 proposal，尚未转化为执行后的正确搭档选择。
3. 新 C 仍明显落后于源 C，说明“有通道”不等于“已经传递了源协议”。可能的瓶颈包括随机初始化、信用分配、源 A/B 对旧 C 消息的依赖，以及新 C 没有显式的协议对齐阶段。

因此不能把这组结果写成新语言、词义、组合语法或代际语言形成。消息仍是固定长度的 8 类离散 token，任务接口和奖励也由研究者预先给定。

## 审计与可复现文件

- 冻结方案：[plan.json](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_new_receiver_transmission_study/results/transmission_002/plan.json)、[prepared.json](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_new_receiver_transmission_study/results/transmission_002/prepared.json)。
- 运行结果：[execution/results.json](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_new_receiver_transmission_study/results/transmission_002/execution/results.json)，SHA-256 为 e3af7012f2b38006be46b2796e0368a1c8b6103ea5174208641b88237c6cfef2。
- 独立检查点审计：[verification.json](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_new_receiver_transmission_study/results/audit_transmission_002/verification.json)，SHA-256 为 301c019a7a8770c71f4d8c5c3ec48da2caba06ec49a983cc118211ab9673ce45。审计重放 192 个监控检查点和 32 个最终评估，共 48,522,240 次模块前向，最大绝对误差 0；192 个检查点的 A/B 参数均未改变；96,000 行配对训练日志的随机流通过核验。
- JSON 汇总：[summary/results.json](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_new_receiver_transmission_study/results/summary_transmission_002/results.json)。
- 图表：[figures_transmission_002](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_new_receiver_transmission_study/results/figures_transmission_002)。三张 PNG 已逐幅查看，收据标记为 visual_review=passed。

## 下一步

下一轮不应直接把更多主体放进同一训练，而应先把“新 C 学到行动”与“新 C 学到旧协议”拆开：

1. **协议恢复对照。** 增加 C-only action、C-only sender、固定旧 C 消息、可学习消息映射四个适应臂，记录 Q、物理执行、搭档、地点、目的地和长度选择，定位新 C 需要恢复的具体模块。
2. **新需求组合。** 在适应后评估未见过的对象 × 属性 × 目的地组合；只在消息改变且行为按新组合改变时，才算接收内容的候选证据。
3. **链式传递。** 让适应后的 C 与一个新的 D 形成下一代，重复冻结两主体、替换一主体的流程，报告每一代的协议保真度和任务性能衰减。需要同时保留 silent、随机 token 和消息重编码对照。
4. **逐步增加社会压力。** 先增加资源稀缺和互补分工，再增加多种可行分工、动态任务和新主体加入；每次只改变一个环境或任务因素。
5. **模型升级作为独立因素。** 当前证据来自可审计的本地 NumPy MLP，不涉及用户之前设想的无语言视觉模型或更大开源模型。若小模型路线出现稳定的跨代、跨组合传递，再把感知模型和规模作为独立实验因素，而不是与任务设计同时更换。
