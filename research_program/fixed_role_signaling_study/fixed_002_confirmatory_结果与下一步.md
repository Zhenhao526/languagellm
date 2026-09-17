# 固定分工确认批次：结果与下一步

## 设计

这一版把同时动作的 credit noise 去掉：agent 0 固定为 scout，私下知道需求并在前两轮发 8-token 消息；agent 1 固定为唯一 worker，只有 worker 能采集地点。PI 条件下 worker 不看到 scout 需求或远端地点；消息在第 0、1 轮后进入 blackout。4 个 seed × `recurrent_scarce_PI_{persistent,switching}_{live,silent}`，共 16 个 run、2,000 次更新；held-out 评价同时保存 natural、closed 和 permuted。

## 结果

8 个 live held-out cell 的平均 natural−closed 为 **+0.387 个百分点**，但 natural−permuted 只有 **+0.0006 个百分点**。效应几乎全部来自一个 seed 的 persistent cell：该 cell natural−closed 为 **+3.10 pp**，natural−permuted 仍约 **0.01 pp**；其余 live cell 的自然与 closed 大多完全相同。PI＋switching 的 4 个 live cell 中，两个差值均为 **0**。消息互信息均值约 **0.332 bit**，仍没有对应的置换损失。

因此固定分工解决了同时结算的混淆，却没有得到“自然配对有效、置换失效”的共同符号证据。一个 token 统计与回报相关的 cell 不足以称为通信，因为打乱 token 与 episode 的对应关系没有降低回报。

## 审计

独立 [audit.py](audit.py) 按冻结源快照检查 16 个 run、32,000 条训练日志和 96 个 final evaluation block；silent alias、计数、有限性、训练日志哈希和 oracle 一致性均通过，非有限诊断为 0，oracle 最大误差为 0。

## 下一步

现在最需要的是能力对照，而不是再扩大 seed：对同一个固定分工环境，给 worker 一个显式的 scout-request 标签，确认它能在相同资源条件下获得正的 held-out gain；同时记录 worker 是否真正读入接收 token。若显式标签有效而 natural/permuted 仍为零，瓶颈在离散消息的信用分配或时序；若显式标签也无效，应先修正任务和环境。只有通过这个门槛，才继续公共／私有码本、随机身份和新成员接入。
