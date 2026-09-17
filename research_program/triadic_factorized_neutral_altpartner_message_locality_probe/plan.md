# FI 消息局部性与重编码探针计划

## 问题

在三人的需求全部可见时，已经训练好的实时消息策略是否仍依赖跨主体消息的 token 身份、槽位顺序或某一发送者的包？如果实时策略对这些干预敏感而静默策略不敏感，说明通道有固定的任务协调作用；它仍不能说明消息具有公共词义。

## 固定来源与模式

- 来源是 `triadic_factorized_neutral_altpartner_information_control_study/results/fi_001` 的 64 个最终 checkpoint，种子 66701–66716、`static/rematched × FI_live/FI_silent`。
- 每个策略在 56,160 个新布局世界（1,560 个需求 × 6 个布局 × 6 个站点所有者）上评估。
- `natural`、`cross_closed`、固定符号双射、四槽循环、四槽反转、A/B/C 发送者遮蔽，共 8 个模式。
- 变换只作用于跨主体 payload；自消息仍可见。第一窗变换后的路由进入第二窗，第二窗也按同一模式路由到行动头。
- 使用贪心 token 和贪心因子化行动；无优化器更新、无新模型调用。

## 读数与边界

每个模式记录团队 Q、物理执行率、`Q|physical`、目标搭档率、proposal 合法率、engage/neutral 和第三人 neutral。主要报告自然减干预的差异，以及同 seed 下 live 与 own/silent 差异的配对描述；不把后验读数当作预注册因果结论。

