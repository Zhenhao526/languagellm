# Open-world cultural transmission formal archive (2026-09-18)

本归档检验：已经形成的单新值扩展，能否由一个新 worker 通过与 incumbent 的单新值互动学会，并在没有见过 `(3,3)` 的情况下恢复双新值组合。父代只见值 `0,1,2`；expansion child 与 fresh sender/receiver 一起学习六个 single-new 意义；随后冻结该 child checkpoint，分别替换 sender、receiver、双方，或完全不训练新 worker。

正式矩阵包含 54 个 parent、27 个 expansion child 和 108 个 transfer run，共 9 个 seed、3 种架构。六片独立 replay audit 全部完成，最大回放误差为 0；transfer audit 还逐 update 检查 incumbent 参数哈希在训练中保持不变。`expanded_single_none` 是真正的 0-update 对照；此前曾误把该控制训练 3,000 步的诊断批次已删除，未进入本归档。

主要 transfer 结果（`new-double` natural return）：

- factorized：sender replacement 1.000（9/9）、receiver replacement 1.000（9/9）、fresh–fresh 0.989（9/9）、0-update 受控替换 −0.164（0/9）；
- tied_routed：sender replacement 0.549（6/9）、receiver replacement 0.028（2/9）、fresh–fresh 0.252（4/9）、0-update −0.088（1/9）；
- holistic：sender/receiver/fresh–fresh 均 0/9，分别为 −0.250、−0.250、−0.250。

Factorized 的 sender-only 和 receiver-only 传递在自然与 donor-recombined double-new endpoint 上均一致，说明新 worker 可以从单新值互动中恢复两个可重组的原子坐标；holistic 没有相同的跨意义坐标。tied_routed 的方向和 seed 敏感性表明共享 routing 仍是脆弱的中间机制，不能当作稳定文化传递。

`expansion_*` 保存扩展阶段结果，`transfer_*` 保存传递阶段结果；`part_audits/` 保存六片审计，`combined_audit.json` 保存合并收据；`prepared.json`、`plan.json`、`freeze.json` 和 `source_snapshot/` 保存冻结设计与源码。原始日志和 checkpoint 在归档校验后删除。
