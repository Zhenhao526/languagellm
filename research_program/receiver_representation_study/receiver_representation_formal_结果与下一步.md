# 接收者表示与未见组合正式结果

## 设计

本实验固定 staged、factorized、dual2 parent sender、世界、伙伴轮换、消息长度、奖励和 leave-one-out 支持，只替换新 worker 的状态表示。`joint_history` 使用完整消息历史作为 worker 的离散状态；`slot_local` 在每个子任务只读取当前 staged slot。两种表示使用完全相同形状的参数张量、初始化种子、训练更新数和 live/silent 配对流。

每个 seed 留出一个目标组合。`full` child 看到四种组合，`leave_one_out` child 只看到另外三种。留出组合的 held-out natural team return ≥ 0.60 被预先定义为有限协议中的零样本恢复。

## 结果

| parent 分层 | 表示 | full live 留出组合 | leave-one-out live 留出组合 | 达到 0.60 |
|---|---|---:|---:|---:|
| parent composable | joint-history | 0.667 | 0.255 | 0/9 |
| parent composable | slot-local | 0.667 | 0.667 | 9/9 |
| parent non-composable | joint-history | 0.499 | 0.250 | 0/23 |
| parent non-composable | slot-local | 0.440 | 0.196 | 0/23 |

在 composable parent 分层内，slot-local 把 leave-one-out 的留出表现从 0.255 提升到 0.667，9/9 个 seed 通过；joint-history 仍为 0/9。slot-local−joint-history 的总体留出差为 +0.077 [−0.012,+0.165]，总体区间被非 composable parent 拉宽；因此论文主分析应报告 parent 分层交互，而不是只报告总体平均。

这个结果把前一轮的负结果解释为表示结构边界：同一个稳定的双槽协议，在完整消息历史 lookup 中不能对未见联合状态泛化，而在 slot-local lookup 中可以。slot-local 并没有改变 sender 或任务 payoff，它直接把每个子任务的行动后果绑定到对应 slot，因此支持“行动依赖提供了组合结构、接收者表示决定能否利用该结构”的机制解释。

256 个 child run 通过独立回放审计：768,000 条训练日志、1,024 个终点 checkpoint、384,000 行 live/silent 配对轨迹，最大回放误差为 0。parent checkpoint 哈希与 child 记录一致；原始 execution 未保留。

## 解释边界与下一步

slot-local 是受控的结构化表示，不应被写成自然语言自发产生的证据。它表明当前实验可以把“协议是否具有组合结构”和“新主体的表示是否允许组合泛化”分离。下一步应把 slot-local 表示放回可学习的 recurrent/attention 接收者中，比较显式结构、参数共享和训练压力三种来源；随后把通过的 slot-local child 接入两到三代传递链，测量组合保真度、漂移和噪声下的重新协商。

压缩结果、冻结配置和独立审计收据位于 `results/formal_20260917/`。
