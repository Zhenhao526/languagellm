# 二元对象表面重映射实验同步与清理

本轮把修正后的二元对象表面重映射实验同步到 `https://github.com/Zhenhao526/language`。正式矩阵包含 9 个 seed、18 个 parent 和 216 个 child；三片独立 replay audit 均通过，最大重放误差为 0。首个内容提交为 `56f9a1b7bb8fef19c20d84608dbe50256da9b28f`，清理元数据提交为 `1f4b63d14593b4f7c4d9c33092b2b3ecb3e148b6`，均已用 `git ls-remote` 核验远端 `main`。`results/formal_20260918_corrected/` 保留冻结配置、聚合指标、parent identity/swap shift、紧凑 per-run 结果、审计摘要和文件哈希；合并的完整 `results.json` 仅作为远端过程记录上传。

首轮 all-goal permutation 无效的 diagnostic archive 仍单独保留在 `results/formal_20260918/`，其失效原因见 `validity.md`，不与修正版统计合并。

远端提交完成并核验后，已删除本地 `/private/tmp/osr_remap_v4_*` 原始训练树、合并过程 JSON、临时审计输出、旧诊断临时树和临时 clone，共释放 1,544,335,853 bytes；本地只保留实验源代码、冻结方案、紧凑结果、审计收据、研究报告和运行环境。清理明细记录在 `LOCAL_CLEANUP_RECEIPT_2026-09-18_object_surface_remap.json`。
