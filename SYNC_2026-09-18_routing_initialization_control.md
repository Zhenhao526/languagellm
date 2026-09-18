# Routing 初始化控制实验同步与清理说明（2026-09-18）

本批次已完成并同步到 `origin/main`：比较 `routed`、同初值独立更新的 `sync_routed` 与训练期间共享 routing 的 `tied_routed`。正式矩阵为 9 个 seed、54 个 parent、81 个 child，冻结为 aligned/hidden 与 full、held-out-combination、held-out-value 三种 support。

合并 replay audit 通过，最大回放误差为 0；重放 162,000 条 parent 与 243,000 条 child 日志、216/324 个 checkpoint，并完成 architecture/support 配对流校验。aligned/hidden 的 held-out-combination 回报为 routed 0.050、sync_routed 0.325、tied_routed 0.552；`sync_routed−routed` 为 +0.274（95% CI [0.070, 0.478]），`tied_routed−sync_routed` 为 +0.227（[−0.036, 0.491]）。因此同一初始化的收益已被识别，但共享更新的额外收益在当前 9 个 seed 下仍不确定。

紧凑结果、冻结 JSON、源码快照和审计收据位于 `research_program/routing_initialization_control_study/results/formal_20260918/`。原始训练日志、checkpoint、pilot 和合并临时树已按 [LOCAL_CLEANUP_RECEIPT_2026-09-18_routing_initialization_control.json](LOCAL_CLEANUP_RECEIPT_2026-09-18_routing_initialization_control.json) 删除。
