# 策略增益研究：完整结果分析

64 个社会运行完整，4 个新独立训练种子。各格先平均三个划分，完整训练参照单列。
J 是实际同一世界两个查询都正确；U 是存在同一完整码可同时正确；N 是固定同照片自然消息双目标正确。

| 范围 | 数据 | 指标 | λ效应 α1 | λ效应 α3 | 交互 |
|---|---|---|---:|---:|---:|
| full | train | J | 18.16% | 34.04% | 15.88% |
| split | heldout | J | 4.58% | 6.52% | 1.94% |
| split | train | J | 16.10% | 30.95% | 14.84% |

完整 JSON 保存全部四种子的效应、所有检查点及五种评价模式。交互为 (λ1−λ0)α3−(λ1−λ0)α1。

- Policy gain scales REINFORCE only; entropy/value/native R are unscaled, but actual gradients and clipping may change.
- J samples random pairs from the old test pool of 8 food and 8 water photographs (64 possible pairs); N uses 16 fixed old validation pairs; do not merge denominators.
- Checkpoint J reads the saved 1200-world score summaries; those checkpoint evaluation raw worlds were not stored.
- No blocked training arm in this study; evaluation blank/shuffle effects are inference-time interventions.
- Four paired seed values describe this fixed experiment; no inference from photos/splits as independent training repetitions.

实现更正：原汇总将 JSON logits 读成 float64，导致与 float32 保存的前两名差值不能逐位相等。此次在确认 logits 精确回转、动作及并列完全不变后，按原 float32 重算差值；所有原检查继续执行。原失败、冻结源码和原探针均保留。见 analysis_recovery_001/plan.json。
