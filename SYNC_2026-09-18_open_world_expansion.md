# 开放世界扩展实验同步与清理说明（2026-09-18）

本轮把开放世界扩展正式结果整理到 `research_program/open_world_expansion_study/results/formal_20260918/`，并纳入 GitHub 仓库。归档保留逐 run 紧凑端点、seed 汇总、fresh-pair 对照、冻结配置、源码快照、三片 replay audit 和失效设计记录；不保留训练日志、checkpoint 或临时执行树。

正式矩阵为 54 个 parent、135 个 child、9 个 seed。v3 执行树的训练重放与 v4 role=2 重组评估修正后的 combined audit 通过，最大回放误差为 0。严格的 `expanded_single_pair` 排除了双新值 `(3,3)` 的训练，因此 `new-double` 是零样本组合端点；factorized 为 1.000，tied_routed 为 0.988，holistic 为 −0.250。这个结果支持“结构化组合坐标 + fresh–fresh 后果窗口”的受限机制解释，不能单独宣称 holistic 主体自主产生了人类语言。

v1 和 v2 只作为失效设计说明保留：v1 的意义索引和 distractor 场景有设计缺陷；v2 虽修正了这些问题，但训练支持包含双新值，不能用于严格零样本组合结论。详情见归档中的 `invalid_design_notes.md`。

清理前 `/private/tmp` 中与本批次匹配的 34 个临时路径已删除，共释放 3,763 个文件、1,494,698,343 bytes；清理后没有残留 `open_world_expansion*` 临时路径。完整收据见 [LOCAL_CLEANUP_RECEIPT_2026-09-18_open_world_expansion.json](LOCAL_CLEANUP_RECEIPT_2026-09-18_open_world_expansion.json)。
