# 同架构 A 全组合控制：联合组合留出语义转移

## 这轮实验为什么必要

上一轮比较了只见部分联合组合训练的 A 与冻结父代 B/C。A 的 aligned−placebo 转移接近零，而 B/C 的点估计约为 9–10 个百分点；但 B/C 与 A 不仅训练支持不同，主体角色和训练历史也不同。一个更严格的解释是：A 的失败只是因为它没有见过全部联合组合。

本轮使用已有的 generation-2 `replace_A` 全组合策略作为匹配控制。它与只见联合组合的 A 共享同一父代 B/C、同一新 A 初始化、同一世界／消息／重配随机流和同一 6,000 步预算；区别只有 A 的训练组合支持。两臂都在完全相同的留出案例上做后验 aligned/placebo 消息干预，没有新的优化更新。

## 冻结设计

- **seen A。** 来自 `triadic_new_receiver_compositional_holdout_study/results/holdout_001`，A 训练只使用 1,248 个联合组合。
- **all A。** 来自 `triadic_protocol_chain_study/results/chain_001` 的 generation-2 `replace_A`，A 训练使用完整联合组合支持。
- **配对。** 8 个种子、static/rematched × live/silent 四格；每策略 3,456 条有向案例、11,232 个自然世界。共有 64 个冻结策略块。
- **干预。** 供体消息来自同一背景中沿 `kind` 或 `length` 的留出近邻；placebo 在发送者×轴×背景层内循环错配，保留消息边际而破坏需求—消息对应。
- **审计。** 独立 replay 不导入生产探针实现，重新构造自然消息库、留出合法方案、aligned/placebo 路由、计划质量、搭档转移和严格结算。

逐条比较 32 对训练日志后，32/32 对的 `world_uniforms`、`sample_uniforms` 和 `permutation_indices` 哈希完全相同；由于训练支持不同，采样到的训练索引本来就不同。这验证了“只有 A 的组合支持改变”的随机流条件。

## 结果

| 条件 | seen A aligned−placebo | all A aligned−placebo | all−seen |
|---|---:|---:|---:|
| static/live | +0.3595 pp [−0.5050, +1.2240] | +0.2904 pp [−0.2402, +0.8210] | −0.0691 pp [−0.5299, +0.3917] |
| rematched/live | −0.4221 pp [−1.4202, +0.5760] | −0.5308 pp [−2.0471, +0.9855] | −0.1087 pp [−0.6552, +0.4377] |
| static/silent | 0.0000 pp | 0.0000 pp | 0.0000 pp |
| rematched/silent | 0.0000 pp | 0.0000 pp | 0.0000 pp |

同架构的全组合 A 仍接近零，且与只见组合 A 的差距也接近零。all−seen 的 static/live 与 rematched/live 分别为 −0.0691 pp 和 −0.1087 pp，区间都跨零；预先定义的 schedule 交互为 −0.0396 pp，t(7) 区间 [−0.1334, +0.0541]。物理执行、Q、执行后条件 Q 和行动改变率的交互同样很小，搭档转移理论上为零且逐项为零。

这与前一轮 B/C 的正点估计形成了清楚的三层边界：

1. 只见联合组合 A：没有稳定的留出内容转移。
2. 同架构、同初始化、全组合训练 A：仍没有稳定的留出内容转移。
3. 全组合训练的冻结父代 B/C：点估计约 +9–10 pp，但区间较宽。

因此，当前负结果不能归因于“全组合支持不足”这一单一因素。更可能的候选包括主体角色／模块历史差异、A 的接收路径没有学会使用跨主体包、训练目标只推动可执行动作而没有推动内容选择，或者整包消息在当前任务中只是父代策略的固定接口。这里的“更可能”是机制解释，不是由本轮直接识别的因果结论。

## 审计与记录

- [冻结准备与匹配输入](results/matched_control_005/prepared.json)
- [权威执行结果](results/matched_control_005/execution/results.json)：64 个策略块、8,091,648 次模块前向、0 次优化更新
- [随机流匹配核验](results/matched_control_005/stream_matching.json)：32 对训练日志逐项通过
- [独立审计](audit_matched_control_005/verification.json)：221,184 行重放、110,592 条 live 干预、64 个 checkpoint、`max_abs_error=0`
- [机器可读汇总](results/summary_matched_control_005/summary.json)与[汇总表](results/summary_matched_control_005/汇总表.md)
- [图表收据](results/figures_matched_control_005/receipt.json)
- [完整哈希链](results/matched_control_005/provenance.json)

`matched_control_002` 至 `matched_control_004` 保留了同一生产结果在冻结／审计修复过程中的开发批次；本报告只使用最后的修正版 `matched_control_005`。

## 解释边界与下一步

本轮把“未见联合组合上的零转移”从训练支持混淆中分离出来，但仍没有证明 A 的内部表示不存在组合性。A、B/C 的角色位置不同，当前观察、奖励和两窗消息接口也允许整包查表与行动捷径。因此这仍是行为识别边界，不是词义、组合语法、公共词典或人类语言起源证据。

下一轮应把接收者角色也做成完全匹配的交叉设计：同一初始化和父代下，让 A、B、C 分别承担接收者训练；同时加入单槽替换、保持等式模式的重编码和新的独立种子。只有当某一接收者在未见联合组合上稳定通过 aligned−placebo，且效果不随角色和整包重编码消失，才值得进入代际传递或更开放的分工生态。
