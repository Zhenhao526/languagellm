# 二元对象表面重映射实验同步与清理

修正后的二元对象表面重映射矩阵已完成并通过三片独立 replay audit：9 个 seed、18 个 parent、216 个 child；重放最大误差为 0。`research_program/object_surface_remap_study/results/formal_20260918_corrected/` 保留冻结配置、聚合指标、parent identity/swap shift、紧凑 per-run 结果、审计摘要和文件哈希；首轮无效 permutation 对照单独保留在 `formal_20260918/`。

目标仓库为 `https://github.com/Zhenhao526/languagellm`，已核验 `origin/main` 提交 `798d5663722d4c3fd1b41d052e02e50ca5c76f90`。完整逐更新日志和 checkpoint 在哈希核对、审计通过后删除，本地只保留可复核的紧凑记录。
