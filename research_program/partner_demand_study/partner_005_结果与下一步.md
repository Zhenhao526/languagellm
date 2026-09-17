# Private-demand 先导网格：结果与下一步

## 设计

这轮把信息不对称放进任务收益：agent `a` 私下知道自己要的材料，但真正被评分的是伙伴 `1-a` 是否收集到该材料。每个 episode 有两个地点、六轮、持续库存；消息只在前两轮发送，之后 blackout。策略不接收伙伴需求或远端状态（PI）时，伙伴动作不能从本地观察直接推出目标。评价同时保存自然通信、关闭通道 `closed` 和批内循环配对 `permuted`。

先导使用当前冻结的 `partner_005` 源快照：2 个 seed（68101、68102）× 32 条件、每条件 200 次更新，共 64 个 run；每个 run 在 training-support 和 held-out switching 分区各评价 4,096 个 episode。错误动作代价为 −0.25，策略不接收结果正确性反馈；消息字母表为 8 个 token。原始输出由 `.gitignore` 排除，冻结计划和源哈希保留在本地 `results/partner_005/`。

## 结果

在 32 个 live held-out 条件上，自然通信相对关闭通道的平均差为 **−0.667 个百分点**，相对 permuted 的平均差为 **−0.175 个百分点**。PI 且 switching 的 8 个 cell 上，这两个差分别为 **−0.824** 和 **−0.039 个百分点**；在更稀缺的 PI＋scarce＋switching 子集（4 个 cell）上为 **+0.241** 和 **+0.231 个百分点**。自然与 permuted 几乎相同，不能把这个小正差解释为消息—episode 的因果对应。

消息和私有状态之间的平均互信息并不低：所有 live held-out cell 为 **0.360 bit**，PI＋switching 为 **0.442 bit**，PI＋scarce＋switching 为 **0.352 bit**。这组结果给出一个重要的负控制：**token 可以携带可预测的本地状态信息，但没有稳定提高自然配对下的伙伴收益；互信息本身不能称作语言形成。**

当前先导只有两个 seed、200 次更新，不能作为最终零结果或投稿主统计。它说明训练预算和信用分配仍是可辨识性的瓶颈：消息头能学到状态相关性，却没有把这种相关性转成被伙伴正确使用的行动。

## 审计

`audit.py` 按 `partner_005` 的 source snapshot 审计了 64 个 run、12,800 条训练日志和 384 个 final evaluation block：结果副本、日志哈希、action/message 计数、oracle 一致性和有限性检查均通过；非有限诊断为 0，oracle 一致性最大误差为 0。当前工作树随后出现了新的消息 credit baseline 改动，因此本报告明确绑定 `partner_005` 源快照，不把后续代码的结果混入本批。

## 下一步

先冻结当前改进后的 message credit baseline，再做同一网格的 2,000 更新确认批次，并将主量固定为 `natural−closed` 与 `natural−permuted` 的差中差：PI 相对 FI、scarce 相对 abundant、switching 相对 persistent。只有当自然通信增益在两个控制下方向一致、且置换能消除增益时，才进入 8 seed 正式网格；如果互信息继续存在而自然／置换差仍接近零，应停止扩大训练，把结论收窄为“私有信息与离散消息相关，但当前任务未产生可操作的共享约定”。

这条路线比继续改符号字母表更接近语言产生的必要条件：消息必须改变伙伴对私有需求的行动，而不是只改变发送者自己的策略统计。
