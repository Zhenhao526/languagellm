# Invalid design notes

- v1 不是正式证据：旧意义曾用连续 `0..8` 索引，而环境使用 base-4 的双 slot 编码；随机 distractor 还允许接收者只靠一个属性完成选择。
- v2 修正了意义索引和严格 distractor 场景，但 `expanded_pair` 在训练中包含双新值 `(3,3)`，因此其 double-new 结果可能是直接记忆，不可用于零样本组合结论。
- v3 保留了修正后的训练树，并加入 `expanded_single_alternating`/`expanded_single_pair`，把 `(3,3)` 从训练支持中排除。v4 只重算 role=2 fresh sender 的 recombination evaluator；所有 v3 训练 checkpoint 的 replay audit 仍通过，最终使用 v4 重算 endpoint。

这些批次的原始执行树没有进入正式归档。
