# 群体感知异质性实验同步与清理

本轮完成了 `heterogeneous_grounding_study` 的正式九种子矩阵，并将源代码、紧凑结果、冻结方案和审计收据纳入 `git@github.com:Zhenhao526/languagellm.git`。正式矩阵包含 36 个 parent run、432 个 child run（48 个 child 条件，每个 3,000 次更新）。三片独立 replay audit 均通过，合计检查 108,000 条 parent 与 1,296,000 条 child 训练记录、144/1,728 个 parent/child checkpoint，以及各 648,000 行 live/silent 和 identity/swap 配对流，最大绝对回放误差为 0。

结果位于：

- `research_program/heterogeneous_grounding_study/results/formal_20260918/`：正式终点、三片审计、分析表、执行计划和 exploratory `pilot_curves.json`；
- `research_program/heterogeneous_grounding_study/results/diagnostic_20260918/`：一枚种子的 3,000-update 设计诊断，不作为正式群体效应估计。

正式结果首先是可识别性诊断。当前 child 只训练和评估被替换的 worker 0，`child_mapping` 又覆盖该 worker 的 surface mapping，因此异质群体中 worker 1、3 的约定不会进入目标 child 流。正式矩阵没有把这一设置误报为“异质性无效应”；后续应让 child 实际接触多个异质伙伴，或在跨伙伴映射下迁移 incumbent worker，再估计群体效应。

三片正式分片的原始日志和 checkpoint 在审计后删除，仅保留紧凑 payload。临时目录清理记录在 `LOCAL_CLEANUP_RECEIPT_2026-09-18_heterogeneous_formal.json`；本批实际删除约 0.87 MB 的中间聚合和备份文件，之前自动清理的正式分片临时树均核验为不存在。
远端核验提交：`fc026e6504d3d78fb213604899c229dafa3a4e00`（本地 `HEAD` 与 `origin/main` 已一致）。
随后发现旧错误 clone 在进程结束时完成了写入，已再次删除；storage sweep 两次合计释放约 2.88 GB，当前 `/private/tmp` 约 20 MB。
