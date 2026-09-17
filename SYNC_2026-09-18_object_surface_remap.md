# 二元对象表面重映射实验同步与清理

修正后的二元对象表面重映射矩阵已完成并通过三片独立 replay audit：9 个 seed、18 个 parent、216 个 child；重放最大误差为 0。`formal_20260918_corrected/` 保留冻结配置、聚合指标、parent identity/swap shift、紧凑 per-run 结果、审计摘要和文件哈希；首轮无效 permutation 对照单独保留在 `formal_20260918/`。

目标仓库为 `https://github.com/Zhenhao526/languagellm`。源码与诊断归档的已核验提交为 `9a8a3bdda609773a884cd0729dda0d5874e12a08`；修正版紧凑归档将在本次元数据提交后一并推送。完整逐更新日志和 checkpoint 在哈希核对、审计通过后删除，本地只保留可复核的紧凑记录。
