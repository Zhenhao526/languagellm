# 行动依赖与组合性信号：32-seed 完整结果

## 设计

发送者观察两个私有二元目标，worker 依次完成两个资源子任务，消息固定为两个二值 token。两种协议只改变消息到达时间：

- `simultaneous`：两个 token 都在第一个行动前到达，允许任意整体码本；
- `staged`：slot 0 在第一个子任务前到达，slot 1 在第二个子任务前到达，使每个槽位承担不同的行动后果。

每个协议包含 factorized 与 entangled 任务，以及 live/silent 对照。8 个探索 seed 之后，预先固定 seed-level 判据：held-out natural ≥ 0.60 且 `|recombined−natural|≤0.02`，称为“可组合平衡”。随后增加 16 个新 seed，最终共 32 seeds、256 runs、3000 updates。

## 结果

| 任务 | 协议 | 可组合平衡 | natural | 重组−natural |
|---|---|---:|---:|---:|
| factorized | simultaneous | 3/32 = 9.4% | 0.494 | −0.150 |
| factorized | staged | 9/32 = 28.1% | 0.517 | −0.135 |
| entangled | simultaneous | 0/32 = 0% | 0.471 | −0.137 |
| entangled | staged | 10/32 = 31.2% | 0.523 | −0.111 |

在同一 seed 配对比较中，staged−simultaneous 的 natural 差值为 factorized `+0.022`（95% t 区间 [−0.016,+0.060]），entangled `+0.052`（[+0.016,+0.088]）。可组合平衡的 McNemar 配对计数为 factorized `7 vs 1`（exact p=0.0703），entangled `10 vs 0`（p=0.0020）。这说明行动依赖主要改变的是进入高回报、可重组吸引域的概率，而不是所有 run 的平均回报。

所有 256 runs 通过独立审计：768,000 条训练日志、256 个终点 checkpoint、384,000 对 live/silent 轨迹，最大回放误差为 0。结果不支持“分阶段消息必然产生组合性语言”；它支持一个更窄、可检验的机制命题：当槽位承担不同的行动后果时，组合性平衡更容易出现，尤其在第二个目标由两个因素的 XOR 决定时。

## 边界与下一步

当前 worker 和 sender 仍是 tabular 策略，所有四种目标组合在训练中都出现过；因此这里测量的是组合平衡的形成概率，而不是严格的零样本新组合泛化。下一步需要在同一 staged/simultaneous 对照下加入三种传递测试：训练完成后替换新 worker、只给新 worker 成功/失败反馈、以及只允许它观察消息。若 staged 形成的槽位语义能被新 worker 恢复，并在未见组合上保持重组收益，才可把它提升为文化传递或组合语法的证据。

完整压缩结果、冻结方案、独立审计和 seed-level 分析位于 `results/formal_20260917/`；原始 execution、日志和 checkpoint 未保留。
