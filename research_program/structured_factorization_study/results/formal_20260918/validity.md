# 有效性收据

- 独立 replay audit：`passed`。
- 覆盖 `72` parent、`216` child；训练日志 `216,000` / `648,000`；checkpoint `288` / `864`；最大回放误差 `0.0`。
- 配对重放：architecture `324,000`、visibility `324,000`、population `324,000`、support `432,000` 行。
- 同一 seed 和 paired stream 比较 holistic/factorized；conflict 只改变 community 1 的外部 token permutation。

## 解释边界

factorized 策略的 held-out combination 成功来自显式 slot/attribute 参数共享，因此这是结构性归纳偏置的因果对照，不是无先验语言涌现的证明。held-out value 仍失败，说明该机制复用了已学原子值，不能从未见过的原子值凭空建立词义。
