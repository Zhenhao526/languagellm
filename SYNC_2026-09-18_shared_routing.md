# 共享 routing 角色对称正式批次同步与清理

本轮在 `routed_factorization_study` 的同一角色对称 referential game 中加入 `tied_routed`：sender 与 receiver 共享一枚可学习的 slot—attribute routing matrix，但保留各自独立的 atomic factor table。`holistic`、fixed `factorized`、independent `routed` 为匹配对照。

正式矩阵冻结为 9 个 seed、4 种 architecture、2 个 community population、2 种 partner visibility、3 种 support，包含 144 个 parent 和 432 个 child，每个 run 3,000 updates。三片独立执行树均完成；逐片 replay audit 的最大绝对误差为 0.0，合并 audit `passed`，重放 432,000 条 parent 与 1,296,000 条 child 训练日志、576/1,728 个 checkpoint，architecture、visibility、population、support 配对流分别为 1,944,000、648,000、648,000、864,000 行。

aligned/hidden/held-out-combination 的 seed 均值为：`factorized` 0.948（8/9 达到 0.60）、`tied_routed` 0.552（4/9）、`routed` 0.050（0/9）、`holistic` 0.028（0/9）。配对差 `tied_routed−routed` 为 +0.502（95% CI [0.241, 0.762]）。独立 routed 的 sender/receiver route 平均绝对差为 0.369；tied route 差为 0。aligned/hidden/held-out-value 中 tied_routed 为 −0.096，0/9 达到功能阈值。

紧凑归档位于 `research_program/shared_routing_study/results/formal_20260918/`，包含结果、aggregate、seed-level analysis、合并和分片 audit、冻结方案、源快照及 manifest。源代码提交为 `5e4fd39dcc27ad74758d02d38e24ebe943c9e413`。

原始 execution、checkpoint、pilot 和 prepared 临时树已在归档 manifest 与审计核验后删除。本轮删除 13 个临时路径、4204 个文件，共 1,870,388,916 字节；清理收据为 `LOCAL_CLEANUP_RECEIPT_2026-09-18_shared_routing.json`。
