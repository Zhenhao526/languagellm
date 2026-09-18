# Structured factor-sharing referential study：正式归档

源代码提交：`6480b69c`。任务保持四对象、两个三值属性、双槽位四符号 referential game 不变，只改变策略的参数共享结构：`holistic` 为完整意义/消息表，`factorized` 按属性值共享发送参数，并让接收者按 slot 对两个属性的证据相加。

- 正式矩阵：9 个 seed、72 个 parent、216 个 child，每个 run 3,000 updates。
- 审计：`passed`；最大绝对回放误差 `0.0`；训练日志 `216,000` + `648,000` 行；checkpoint `288` + `864` 个。
- factorized/aligned/hidden/heldout-combination return `0.974`，9/9 达到 `0.60`；holistic 对照 `-0.220`，0/9 达到。配对差 `1.194`，95% CI [1.127, 1.260]。
- factorized/aligned/hidden/heldout-value return `0.122`，0/9 达到，说明组合复用不等于未见原子值传递。
- factorized/conflict/hidden/heldout-combination 为 `0.434`；伙伴身份 visible 后为 `0.974`，显示社区冲突与伙伴专用适应之间的可识别差异。

该结构化策略是明确的架构干预，不能被表述为无先验主体自发发明了组合语法；它提供的是“环境压力本身不足以产生组合复用，结构化共享可以使其出现”的可检验边界。
