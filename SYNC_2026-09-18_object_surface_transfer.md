# incumbent worker 表面迁移实验同步与清理

本轮在冻结 sender 后直接比较两类 child：`incumbent` 复制 parent worker 0，`fresh` 使用独立初始化。child 在 identity 或稳定 0↔1 object-surface swap 下运行，并保留 live/silent 配对。正式矩阵为 9 个 parent、72 个 child、每个 3,000 次更新。

三片独立 replay audit 均通过，最大回放误差为 0；审计检查了 9,000 条 parent 和 72,000 条 child 训练记录、12/96 个 checkpoint、初始 incumbent 拷贝、sender 冻结以及 live/silent 与 identity/swap 配对流。紧凑结果、学习曲线、冻结方案、审计和哈希清单位于 `research_program/object_surface_transfer_study/results/formal_20260918/`。

目标仓库是 `https://github.com/Zhenhao526/languagellm`。原始训练日志与 checkpoint 在审计后删除，本地清理收据为 `LOCAL_CLEANUP_RECEIPT_2026-09-18_object_surface_transfer.json`。
