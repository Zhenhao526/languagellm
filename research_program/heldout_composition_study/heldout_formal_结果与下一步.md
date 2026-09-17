# 未见目标组合正式结果

## 设计

每个 seed 先取一个 staged、factorized、dual2 parent checkpoint。随后只替换 worker 0，冻结 sender 和其他 worker。`full` child 在训练中看到四种目标组合；`leave_one_out` child 看到其中三种，第四种组合只在评估时出现。留出组合在 32 个 seed 间均衡，并配有同随机流的 live/silent 通道对照。parent 是否满足此前的可组合平衡在 child 训练前由独立的 action-dependent 结果冻结。

零样本判据是留出组合的 held-out natural team return ≥ 0.60；四步任务的功能上限约为 0.667。这个判据只涉及有限协议中的行为恢复，不等同于开放词汇或人类语言的组合语法。

## 结果

| parent 分层 | n | full live 留出组合 | leave-one-out live 留出组合 | 达到 0.60 |
|---|---:|---:|---:|---:|
| parent composable | 9 | 0.667 | 0.263 | 0/9 |
| parent non-composable | 23 | 0.499 | 0.250 | 0/23 |

总体 full live 的留出表现为 0.546，leave-one-out 为 0.254；配对差值为 −0.293 [−0.344, −0.241]。在 leave-one-out 支持内，live−silent 的留出差值为 +0.319 [+0.268, +0.371]。因此新 worker 仍然使用稳定消息来完成已见组合，但未见的联合消息状态没有被零样本恢复。

128 个 child run 通过独立审计：384,000 条训练日志、512 个终点 checkpoint、192,000 行 live/silent 配对轨迹，最大回放误差为 0。parent checkpoint 在重建后与 child 结果中记录的 32 个 SHA-256 全部一致。

## 解释边界

即使 parent 已经满足“整体表现高且槽位重组无损”的平衡，新的 tabular worker 仍不能处理训练中缺失的联合消息状态。这说明整体码本传递、已见槽位重组和未见组合泛化是三个不同层次的能力。当前 worker 把完整消息历史作为离散状态，没有足够的参数共享压力去把两个槽位的动作映射分解到新组合；因此，下一步应比较 joint-history 与 slot-local 的接收者表示，并保持同一 leave-one-out 干预。

压缩结果、冻结配置、parent 分层、重建哈希核验和审计收据位于 `results/formal_20260917/`。原始 child 日志、checkpoint 和合并结果未保留。
