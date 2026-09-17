# Generation transmission study

先训练一个 rotating-hidden 的四 worker parent population，再从同一终点替换一个主体。`worker_*` 条件冻结 sender 与 worker 1–3，只让 worker 0 从随机初始化恢复；`sender_*` 条件冻结四个 worker，只让 sender 从随机初始化恢复。三种 child channel 使用相同的 world/goal/partner/action/message 随机流：`live` 传递自然 token，`silent` 传递 NULL 但保留行动奖励，`scrambled` 对每个 episode 的 token 加独立随机循环偏移。

主读数是新主体 held-out natural return、`live−silent`、`live−scrambled`、训练曲线 AUC 和与 parent 协议的 greedy fidelity。`permuted` 与 `closed` 是终点评估干预，不能单独替代从头 silent 对照。worker 替换只在 partner 0 的 active episodes 上更新，sender 替换更新所有 rotating episodes；互补组件始终冻结。

正式矩阵为 8 个 seed × 6 个 child conditions。每个 seed 共享一次 parent 训练，child arms 共享同一替换初始化和 phase-specific paired streams。结果只检验离散协议的跨主体传递，不把成功恢复称为自然语言、词义或语法。
