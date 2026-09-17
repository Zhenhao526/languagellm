# 同架构 A 全组合控制

本包把 `holdout_001` 的只见联合组合 A 与 `chain_001` 的 generation-2 `replace_A` 全组合 A 放到同一组联合组合留出 aligned/placebo 干预上。两者共享 B/C 父代端点、A 初始参数和随机流；区别只有 A 的训练组合支持。该批次不重新训练，只重放冻结 checkpoint。

权威结果在 `results/matched_control_002/`，独立审计在 `audit_matched_control_002/`，完整说明见 [matched_control_结果与下一步.md](matched_control_结果与下一步.md)。
