# 固定分工：显式需求能力门槛

## 目的

前面的 fixed-role signaling 批次没有得到稳定的 natural−closed 或
natural−permuted 差异。仅凭这个结果，不能判断是消息约定没有形成，还是
worker 连在需求已公开时也没有学会执行。因此先加入一个不训练、不发消息的
直接规则控制：worker 直接读取 scout 的目标序列、两个站点类型和库存，按当前
目标选择可用站点。这个控制只检验任务接口和结算，不被当作语言结果。

## 设置与复核

- 4 个种子：68101–68104；scarce 容量；persistent 与 switching 两种任务；
  training-support 与 held-out 两个分割，共 16 个评价块。
- 直接控制不初始化策略网络、不采样符号、不更新参数。
- 每个时刻从可用且类型匹配的站点中取一个；回报按同一环境的
  `HORIZON=6` 归一化。
- 同一批 episode 同时调用环境内的有限期 oracle，逐块比较直接规则与 oracle。

原始紧凑结果见
[`fixed_ability_gate.json`](fixed_ability_gate.json)，实现见
[`ability_gate.py`](ability_gate.py)。

## 结果

| task | split | direct return | oracle return | direct−oracle | 非负回合率 |
|---|---:|---:|---:|---:|---:|
| persistent | training-support | 0.33616 | 0.33616 | 0 | 1.000 |
| persistent | held-out | 0.33657 | 0.33657 | 0 | 1.000 |
| switching | training-support | 0.54051 | 0.54051 | 0 | 1.000 |
| switching | held-out | 0.54053 | 0.54053 | 0 | 1.000 |

16 个块逐块的 `direct_minus_oracle_mean` 都为 0，直接规则与 oracle 完全一致。
因此环境中存在一个简单、可执行、跨 held-out pattern 仍成立的解决方案；此前
fixed-role 学习批次的低通信差异不能归因于任务不可解或奖励上界过低。

## 解释边界

这个门槛只排除了“环境本身不可解”。它没有证明当前 NumPy policy-gradient
实现能够学到显式需求，也没有证明符号 channel 能形成词义。当前更合理的下一
步是做同架构的 **FI 显式目标可见学习控制**：worker 获得目标，但仍用相同网络、
动作头、奖励和更新规则；先确认 action learner 能达到直接规则的可比水平，再
恢复 PI 消息条件。若 FI 仍失败，应先修复优化/信用分配或降低动作学习难度，
不再把 PI 的零结果写成语言起源证据。

本控制无新增 Qwen 调用；它是固定分工路线的能力审计，不是投稿主结果。
