# 固定分工：神经 FI 显式动作控制

## 目的

显式规则能力门槛证明环境可解，表格策略正对照证明任务中的离散通信信号可以
被低方差策略学到；但 fixed-role 神经策略的 FI 自博弈仍没有达到 oracle。这里
把消息和自博弈信用分配完全拿掉，让同一个 32 单元 recurrent worker 直接用
oracle 产生的动作标签做监督学习。该实验只回答“网络能否表示并执行显式目标”，
不属于语言形成实验。

## 设置

- 4 个 seed：68101–68104；scarce、persistent、FI；worker 仍看到目标、两个站点
  类型和库存。
- 同一 fixed-role 网络结构（32 单元 tanh recurrent core、3-way action head），
  batch 256，1,000 次更新，学习率 0.0005。
- 标签是当前库存下与目标类型匹配的可用站点；每个样本的库存按标签动作顺序
  更新。没有 token、消息头或 policy-gradient reward credit。
- training-support 与 held-out 评价都用 silent environment 的 greedy worker，
  并与同一 episode 的有限期 oracle 对照。

结果保存在 [`fi_supervised_control.json`](fi_supervised_control.json)，实现见
[`fi_supervised_control.py`](fi_supervised_control.py)。

## 结果

| seed | held-out team return | held-out oracle | oracle regret | 最后 100 步平均交叉熵 |
|---:|---:|---:|---:|---:|
| 68101 | 0.33390 | 0.33390 | 0 | 0.01239 |
| 68102 | 0.33879 | 0.33879 | 0 | 0.01410 |
| 68103 | 0.33618 | 0.33618 | 0 | 0.01574 |
| 68104 | 0.33740 | 0.33740 | 0 | 0.01425 |

4 个 seed 的 held-out `team_return` 都与 oracle 完全一致，逐样本平均 regret 为
0。也就是说，当前神经网络能够表达并执行这个显式行动规则；此前 FI/PI 自博弈
的低回报主要来自探索、动作信用分配或训练程序，而不是网络表达能力或任务的
物理上界。

## 解释边界与后续

该结果不能说明网络已经形成词义。监督标签直接提供了请求—行动对应关系，因而
只能作为优化能力控制。后续应先用它检查两条不泄漏主实验的路线：

1. 从监督 FI checkpoint 出发，逐步隐藏目标并恢复 PI 消息自博弈，观察消息是否
   能替代被移除的目标输入；
2. 使用表格正对照的 abundant persistent 条件，比较从头训练、FI warm-start
   和冻结动作头三种方式的 natural/closed/permuted 差异。

正式的语言形成主张仍必须来自从头训练、自然配对增益、置换损失和未见内容迁移
的联合证据；本控制不计入这些主量。
