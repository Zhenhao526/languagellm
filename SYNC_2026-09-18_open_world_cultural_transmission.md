# 开放世界文化传递实验同步与清理说明（2026-09-18）

本轮把“冻结 single-new 扩展后替换 worker”的正式矩阵整理到 `research_program/open_world_cultural_transmission_study/results/formal_20260918/`。归档保留 expansion 与 transfer 的紧凑端点、seed 分析、六片 replay audit、冻结配置、源码快照和实现修正说明；不保留原始训练日志或 checkpoint。

正式矩阵包含 54 个 parent、27 个 expansion child 和 108 个 transfer run。transfer 条件分别替换 sender、receiver、双方，或使用 0-update 随机替换控制。六片审计最大回放误差均为 0，并验证 incumbent 参数哈希在传递训练中保持不变。factorized 的 sender-only 和 receiver-only double-new return 均为 1.000（9/9），0-update 控制为 0/9；holistic 的替换条件均为 0/9；tied_routed 对方向和 seed 敏感。

早期诊断曾错误地让 `expanded_single_none` 执行标准训练循环，该批次已删除，未进入正式统计。修正后的正式控制在 0 update 下运行，详见归档中的 `implementation_correction.md`。

清理前 `/private/tmp` 中与本轮匹配的 41 个临时路径已删除；完整文件数和字节数见 [LOCAL_CLEANUP_RECEIPT_2026-09-18_open_world_cultural_transmission.json](LOCAL_CLEANUP_RECEIPT_2026-09-18_open_world_cultural_transmission.json)。
