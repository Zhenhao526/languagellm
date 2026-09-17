# 生态因子化与新组合传递：正式结果（2026-09-18）

## 设计

本批实验把“新对象”和“新组合”拆成一个可审计的传递干预。每个 episode 有两个行动阶段；每个阶段显示三个对象的一次随机排列，worker 必须结合可见对象排列和消息选择正确对象。sender 观察两个私有三值属性，因此共有九个等概率目标组合。

- `factorized`：两个阶段分别对应两个原始属性；
- `holistic`：对同样九个目标组合施加冻结的双射置换，再得到两个阶段目标；
- `tri3`：两个分阶段到达的三值 slot；
- `mono9`：一个九值原子 token；两者都有九种完整消息状态；
- parent 在九个组合上训练；随后冻结 sender、替换 worker 0；child 使用完整支持或留出一个目标组合的支持；
- tri3 child 比较 `joint_history` 与 `slot_local` 接收者表示，所有 child 都有 live/silent 对照。

正式矩阵覆盖 9 个 seed、36 个 parent 和 216 个 child。零样本判据预先固定为 leave-one-out、live、held-out natural return ≥ 0.60；该任务的功能上限为 0.6667。

## 审计

三片独立执行均通过 replay audit：合计 36,000 条 parent 训练日志、216,000 条 child 训练日志、324,000 条配对 live/silent 轨迹和 1,008 个终点 checkpoint，最大 replay error 为 0。每条训练流、参数更新、checkpoint 和 source snapshot 都有 SHA-256 收据。

## 结果

parent 的 all-goal natural return 为：

| form | task | mean |
|---|---|---:|
| `mono9` | `factorized` | 0.143 |
| `mono9` | `holistic` | 0.131 |
| `tri3` | `factorized` | 0.195 |
| `tri3` | `holistic` | 0.167 |

tri3 在 parent 阶段有一定通信增益，但没有接近功能上限。child 的 leave-one-out live held-out return 为：

| representation | form | task | mean | 达到 0.60 |
|---|---|---|---:|---:|
| `joint_history` | `mono9` | `factorized` | −0.044 | 0/9 |
| `joint_history` | `tri3` | `factorized` | −0.023 | 0/9 |
| `slot_local` | `tri3` | `factorized` | −0.036 | 0/9 |
| `joint_history` | `mono9` | `holistic` | −0.074 | 0/9 |
| `joint_history` | `tri3` | `holistic` | −0.065 | 0/9 |
| `slot_local` | `tri3` | `holistic` | −0.063 | 0/9 |

完整支持下，`joint_history/tri3/factorized` 的 held-out control 为 0.303，`slot_local/tri3/factorized` 为 0.293；移除一个组合后，slot-local 的 paired leave-one-out minus full 差值为 −0.329，95% CI [−0.468, −0.190]。这确认留出干预确实删除了训练支持，而不是单纯的末点波动。

预注册的结构对比没有得到稳定正效应：

- `slot_local − joint_history` 在 factorized tri3、leave-one-out、live、held-out 上为 −0.013，95% CI [−0.057, 0.031]；
- factorized − holistic 在 slot-local tri3 上为 +0.027，95% CI [−0.109, 0.163]；
- tri3 − mono9 在 joint-history factorized leave-one-out 上为 +0.021，95% CI [−0.045, 0.086]。

因此，分阶段 slot、等容量组合形式和因子化行动目标都没有单独带来新组合的零样本恢复。已见组合上的 raw-factor slot recombination 在 factorized slot-local child 中接近无损，但这没有转化为新 worker 对未见联合状态的行为泛化。

## 解释边界

这批结果把三个能力区分开：

1. parent 能否获得消息带来的行为收益；
2. 已形成的 slot 码能否在已见组合上被新 worker 使用；
3. 新 worker 能否把两个已见 slot 语义组合成未见联合状态。

当前三值 tabular 主体在第一层有弱到中等通信收益，在第三层完全没有达到预设阈值。不能把这个负结果解释为“组合语法不可能”：它也可能反映三值场景、状态表大小和当前优化预算的可学习性边界。正式结果支持的较窄命题是：**把目标写成因子、把消息拆成等容量 slot，并不能自动产生可传递的组合协议；新组合泛化需要额外的参数共享或接收者归纳偏置。**

完整紧凑结果、冻结方案和审计收据位于 `results/formal_20260918/`；原始日志和 checkpoint 在归档完成后作为临时执行树清理。

## 下一步

下一批先使用已经验证可学习的二值 `dual2` 生态，加入 parent 与 child 间的对象表面重映射（固定的 object-label permutation），并保持 leave-one-goal-out、slot-local/joint-history、live/silent 和 mono4/dual2 容量对照。这样可以把“旧码本传递失败”拆成两部分：符号协议是否仍被新主体恢复，以及对象落地关系改变后是否需要局部修复。只有二值能力匹配控制重现通信和留出效应，才把三值任务扩展为更大的视觉对象空间。
