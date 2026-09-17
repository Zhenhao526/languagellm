# 定向消息移植的错配 placebo 对照

2026-09-17。上一项固定角色消息移植在 destination 轴观察到 +22.8866 个百分点的 source_pull，但这仍可能只是“任意跨人消息被替换后会改变动作”的一般扰动。为区分这一点，本批次保持每个案例的发送者、听者、需求轴和方向不变，只把源消息在同层案例之间循环错配，再与正确源端消息的移植做配对比较。

结果没有支持“正确源消息优于错配消息”。轴均衡的 PI-live aligned source_pull 为 **+5.9288 pp**，错配 placebo 为 **+7.1641 pp**，aligned−placebo 为 **−1.2353 pp**；四种子描述性 t(3) 95% 区间分别为 [−4.2952,+16.1528]、[−2.5754,+16.9036] 和 **[−2.4676,−0.0029]**。错配消息反而平均改变更多贪心动作（26.4468 pp 对 17.3322 pp）。这不是证明“错配一定更好”，而是说明当前主探针的正 source_pull 不能被解释为源端需求与消息之间已经建立了可识别对齐。

## 对照怎样构造

使用与[固定角色定向移植批次](semantic_transfer_结果与下一步.md)完全相同的 240 个有向案例和 36 个背景。案例按 `axis × changed_person × listener × direction` 分层；kind 和 length 层各有4个案例，destination层有12个案例。每层内按固定排序循环一格，placebo donor 永远不是自身案例，但保持发送者、听者、轴和方向的边际分布。目标端自然消息、目标自视角和其他消息不变，仍只替换改变者的跨视角首窗消息。

因此 aligned 与 placebo 使用相同的目标状态和自然行动分布，差别只有供给听者的源消息是否来自正确的源端需求。PI-live 两种干预都重新计算W2与行动；PI-silent 两者都应是自然路由别名。共8个策略块、1920行、69120个世界；aligned和placebo各有960个live重算，合计414720次模块前向，无训练或优化器更新。

## 结果

| 条件 | aligned source_pull | placebo source_pull | aligned−placebo | aligned动作改变率 | placebo动作改变率 |
|---|---:|---:|---:|---:|---:|
| PI-live | +5.9288 [−4.2952,+16.1528] | +7.1641 [−2.5754,+16.9036] | **−1.2353 [−2.4676,−0.0029]** | +17.3322 [+8.7634,+25.9009] | +26.4468 [+14.1220,+38.7716] |
| PI-silent | 0.0000 [0.0000,0.0000] | 0.0000 [0.0000,0.0000] | 0.0000 [0.0000,0.0000] | 0.0000 [0.0000,0.0000] | 0.0000 [0.0000,0.0000] |

PI-live 分轴的 aligned−placebo source_pull 为：kind **−1.6584 pp**（[−3.6493,+0.3325]）、length **−1.2742 pp**（[−6.5506,+4.0023]）、destination **−0.7732 pp**（[−3.6717,+2.1253]）。destination 上 aligned 和 placebo 都约 +23 pp，说明该轴的定向效应对“消息来自哪个源端案例”并不敏感。

图表：[aligned 与 placebo](results/figures_semantic_transfer_placebo_unique_001/aligned_vs_placebo_source_pull.png)、[aligned−placebo 分轴](results/figures_semantic_transfer_placebo_unique_001/aligned_minus_placebo_by_axis.png)。

## 解释边界

这个阴性对照收窄了上一批结果的解释。上一批的 source_pull 仍然是一个可复现的**跨人消息扰动指标**：live 与 silent 分离，且某些替换会把听者的行动集合质量推向源集合。但由于同层错配消息没有降低该指标，不能说听者读取了源端对象、属性或目的地，更不能说消息中已有稳定词义。

至少有三种机制仍然可能：网络把整包符号当作任意状态键；消息替换触发了身份、位置或第二窗口路径的通用变化；或者错配消息偶然更接近目标策略的动作边界。当前对照没有分离这些机制，也没有测试单槽、组合或跨新需求复用。

因此，论文主张应改写为：**冻结策略对跨人首窗消息具有任务内行动敏感性，但在本批错配 placebo 下没有显示源需求—消息对齐的增量。** 后续独立训练必须把 aligned、同层错配、随机符号、单槽替换和组合替换一起预注册，并直接比较 aligned−placebo，而不是只报告 aligned source_pull。

## 审计与文件

独立审计不导入本生产脚本，重新过滤案例、重建循环错配、重算两种live W2/行动概率并逐行比较：1920行、960 aligned live、960 placebo live、960 silent alias，最大绝对误差0；检查8个checkpoint、8份源结果、8份自然NPZ和240项错配映射。完整文件：

- [执行结果](results/semantic_transfer_placebo_unique_001/execution/results.json)
- [独立审计](results/audit_semantic_transfer_placebo_unique_001/verification.json)
- [统计汇总](results/semantic_transfer_placebo_unique_001/summary_001/summary.json)
- [图形收据](results/figures_semantic_transfer_placebo_unique_001/receipt.json)

所有输入和输出SHA-256见同目录 `provenance.json`。本批次只是在已冻结的`partner_001`策略上做机制控制，不能替代新的从头训练或代际形成实验。
