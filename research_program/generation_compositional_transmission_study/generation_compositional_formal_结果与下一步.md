# staged dual2 协议的 parent→child 传递结果

## 设计

父代来自 `action_dependent_signaling_study` 的 staged、dual2、factorized、rotating-hidden 条件。每个 seed 的 parent checkpoint 冻结后，分别替换 worker 0 或 hidden sender；互补组件保持不变。child 训练 3000 updates，比较：

- `live`：自然 staged 双 token；
- `silent`：NULL token，但保留同一奖励流；
- `permuted`：按 partner 分组置换消息，破坏稳定目标对应关系。

结构判据沿用行动依赖实验：held-out natural ≥ 0.60 且 `|recombined−natural|≤0.02`。parent 是否可组合在 child 训练前由 parent held-out checkpoint 预先确定。

## 结果

32 个 parent 中只有 9 个进入可组合平衡。对这 9 个 parent：

| 替换方向 | child live 可组合 | live natural | live 重组−natural | live−silent | live−permuted |
|---|---:|---:|---:|---:|---:|
| 新 worker | 9/9 | 0.667 | 0.000 | +0.499 [+.441,+.556] | +0.419 [+.350,+.488] |
| 新 sender | 9/9 | 0.667 | 0.000 | +0.393 [+.300,+.486] | +0.613 [+.539,+.688] |

对另外 23 个 parent，两个替换方向的 child 可组合平衡均为 0/23；worker live natural 为 0.468，sender 为 0.458，重组损失分别为 −0.199 和 −0.188。所有 192 个 child run 通过独立审计：576,000 条训练日志、192 个终点 checkpoint、384,000 对配对通道轨迹，最大回放误差为 0。

新 worker 和新 sender 在可组合 parent 条件下都恢复了功能性槽位协议，而不是只恢复一个整体码本。这个结果支持“组合协议可以跨主体传递”，但传递成功严格依赖 parent 已经进入组合平衡；它没有证明组合性会在新主体中从零自发产生。

## 解释边界与下一步

这是一项条件化传递实验，parent stratum 是由先前独立训练得到的终点决定的，因此不能把 9/9 写成总体形成概率。当前四种目标组合仍都在 parent 训练中出现，child 传递也还没有严格的零样本新组合。下一步要把 held-out goal combination 从 parent 训练中真正移除，再检验新主体是否能从两个已知槽位恢复未见组合；随后将同一 child 继续作为下一代 parent，测量两到三代的重组保真度、漂移和噪声鲁棒性。

完整压缩结果、parent checkpoint 哈希、冻结配置、独立审计和分层分析位于 `results/formal_20260917/`；原始 parent/child execution 未保留。
