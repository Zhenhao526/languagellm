# 多合法行动定向首窗探针

这是对 `triadic_multi_legal_coordination_study/results/confirm_001` 最终策略的后验机制探针。它在新布局上只替换一个发送者发给另外两人的第一窗口，并重新计算第二窗口和动作；silent 使用真正的自见路由作为通道关闭参照。

运行方案、指标和解释边界见 [`plan.md`](plan.md)，完整结果见 [`结果与下一步.md`](结果与下一步.md)。`results/probe_001` 保存修正版运行、汇总、审计和图表；`results/invalid_001` 保留首版 silent 路由错误的完整输出，不进入统计。
