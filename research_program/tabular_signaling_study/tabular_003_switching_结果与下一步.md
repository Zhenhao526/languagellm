# 表格策略：switching 与未见模式迁移

## 目的

persistent 正对照只要求发送一个稳定的二元目标。这里保留同一 fixed-role
tabular policy 和通信协议，改为六轮 switching 请求：训练只出现 4 个目标
序列，held-out 评价只出现另外 4 个序列。目标是观察任务结构改变后，已形成的
token—行动关系能否迁移，而不是把训练内的任意 token 映射直接称作语言。

## 设置与审计

- recurrent、PI；abundant/scarce；live/silent；4 个 seed（68101–68104）。
- 2,000 更新，batch 512，共 16 个 run、32,000 条训练日志。
- held-out 仍比较 natural、closed、同 sender 组内 token 置换的 `permuted`。
- 冻结计划和原始运行位于本机 ignored results 目录；
  [`tabular_003_switching_audit.json`](tabular_003_switching_audit.json) 通过了
  16 个 checkpoint 的独立重放，所有保存的 team return 最大绝对误差为 0。

## 结果

| 条件与分割 | natural−closed | natural−permuted | 正差 seed 数 | token MI |
|---|---:|---:|---:|---:|
| abundant live，training-support | **+22.878 pp**（95% CI [+15.340,+30.416]） | **+13.620 pp**（[+11.290,+15.950]） | 4/4，4/4 | 0.846 bit |
| abundant live，held-out | **+11.701 pp**（[+2.837,+20.565]） | +2.191 pp（[−1.550,+5.932]） | 4/4，3/4 | 0.875 bit |
| scarce live，training-support | **+9.257 pp**（[+3.488,+15.026]） | **+2.576 pp**（[+0.126,+5.027]） | 4/4，3/4 | 0.909 bit |
| scarce live，held-out | **+5.548 pp**（[+0.302,+10.794]） | −1.267 pp（[−4.352,+1.819]） | 4/4，1/4 | 0.875 bit |

silent 条件的 natural、closed、permuted 逐行相同。训练模式内，两个容量条件
都显示稳定的自然配对和置换损失；切换到未见模式后，natural−closed 仍为正，
但 permutation loss 大幅衰减并跨零，scarce 条件最明显。token MI 在 held-out
仍约 0.875 bit，却没有相应的稳定置换损失。

## 解释边界

这批结果支持一个窄结论：离散通信约定可以在固定任务中形成，并部分影响未见
switching 模式的行动；但当前 token 是任务级编码，MI 不能替代因果内容泛化，
也不能称词义、组合语法或人类语言起源。scarce 资源进一步放大了“有信号但无
稳健迁移”的差异。

下一步应把 message space 因子化或增加跨模式重组检验，例如把对象、属性、
搭档、目的地拆成可独立变化的需求轴，并在训练中留出轴的组合；只有当单轴和
组合留出同时保留 aligned−placebo 或 natural−permuted 作用，才有资格讨论
组合性。新 agent/代际传递应放在这个内容泛化门槛之后。
