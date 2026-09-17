# 表格策略：固定分工通信正对照

## 目的

固定分工神经策略的 FI 能力对照没有通过。为了区分“任务要求的约定无法通过
自博弈形成”和“神经策略梯度/动作信用分配没有学会一个本来可行的协议”，本
批使用同一任务接口，但把 sender 与 worker 换成低方差的表格 softmax 策略。
它仍然只用环境回报更新，没有教师标签、预置码表或语言先验。

## 冻结设置

- `recurrent`、PI、persistent；abundant 与 scarce 两种容量；live 与 silent
  配对，共 4 个条件。
- seed 68101–68104，2,000 次更新，批量 512；共 16 个 run、32,000 条训练
  日志、每个 run 一个最终 checkpoint。
- held-out 评价沿用 fixed-role 协议：natural、closed 和同 sender 组内 token
  置换 `permuted`。预先要求的因果信号是 natural 高于 closed，且置换消除
  natural 增益。
- 冻结计划在
  `results/tabular_002_confirmatory`；原始运行保留在本机的 ignored results
  目录，紧凑审计结果见
  [`tabular_002_confirmatory_audit.json`](tabular_002_confirmatory_audit.json)。

## 结果

| 条件 | natural | closed | permuted | natural−closed | natural−permuted | 消息 MI |
|---|---:|---:|---:|---:|---:|---:|
| abundant PI live persistent | 0.5363 | 0.2534 | 0.3177 | **+28.284 pp** | **+21.854 pp** | 0.750 bit |
| abundant PI silent persistent | 0.3180 | 0.3180 | 0.3180 | 0 | 0 | 0 |
| scarce PI live persistent | 0.2124 | 0.1011 | 0.2124 | **+11.132 pp** | 0 | 0.705 bit |
| scarce PI silent persistent | 0.2124 | 0.2124 | 0.2124 | 0 | 0 | 0 |

abundant live 的四个 seed 都有正的 natural−closed（均值 **+28.284 pp**）和
natural−permuted（均值 **+21.854 pp**），给出一个可重复的低方差“消息有用、
自然配对有用”的正对照。scarce live 的 natural−closed 四个 seed 也都为正，
但 natural 与 permuted 完全相同；在该容量下不能把增益解释成稳定的伙伴约定，
因此不应与 abundant 结果合并为单一语言证据。

此前 1 个 seed 的 switching 先导中，held-out natural−closed 约 **+0.02 pp**，
natural−permuted 约 **−5.59 pp**；它只用于选择后续方向，不能作为确认结果。

## 独立审计

[`audit.py`](audit.py) 重新读取冻结计划、16 个最终 checkpoint 和训练日志，并
用 checkpoint 从头重放两种 split 的三种 channel 控制。16 个 run、32,000 条
日志均通过，所有数值有限，回放与保存的 team-return 最大绝对误差为 **0**。

## 解释边界与下一步

这批结果证明当前任务至少存在一个容易学习的离散通信解，也说明此前 fixed-role
神经零结果不能直接写成“环境不支持语言”。但表格策略的 token 仍可能只是单步
目标编码；它没有证明词义、组合语法、代际传承或人类语言起源。

下一步先把这个 abundant persistent 表格解作为神经优化的正对照：使用相同的
世界、奖励和控制，比较神经 worker 的 FI 显式目标学习、表格策略初始化迁移和
PI 消息学习。只有神经动作基线达到可比的 oracle 归一化回报，才继续增加
switching、身份重配或新成员迁移；否则应把论文主问题收窄为“通信机会、策略
可学性和可组合泛化之间的边界”。
