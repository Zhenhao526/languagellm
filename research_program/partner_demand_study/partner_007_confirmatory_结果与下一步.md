# Private-demand 确认批次：PI＋scarce

## 设计

`partner_007` 固定当前的消息信用分配、错误代价 −0.25 和结果反馈屏蔽，只保留最有信息量的四格：`recurrent_scarce_PI_{persistent,switching}_{live,silent}`。4 个 seed（68101–68104）× 4 条件，每条件 2,000 次更新；每个最终策略在 4,096 个 held-out episode 上同时评价 natural、closed 和 permuted。

## 结果

16 个 run 的 held-out 结果中，**每个 live 条件的 natural、closed、permuted 团队回报逐行相同**，所以 `natural−closed=0`、`natural−permuted=0`。persistent live 的 token—状态互信息在四个 seed 为约 **0.077、0.125、0.125、0.079 bit**；switching live 的互信息为 0。消息统计没有转化为伙伴行动差异。

这不是“通信已经形成但置换不敏感”的证据：同一最终参数、同一 held-out episode 和同一 greedy policy 下，关闭或置换消息完全不改变行动回报。它给当前任务一个清楚的负边界：在两轮消息、局部库存和当前 score-function 学习下，agent 可以产生少量状态相关 token，却没有把 token 当作伙伴需求的操作性约定。

## 审计

独立 `audit.py` 按 `partner_007` 源快照检查 16 个 run、32,000 条训练日志和 96 个 final evaluation block；结果副本、训练日志哈希、action/message 计数、oracle 一致性和有限性均通过，非有限诊断为 0，oracle 最大误差为 0。

## 下一步

继续增加更新次数不会解决当前的可辨识性问题。下一版应先做一个能力对照：给一个固定的共同计划或显式需求标签，确认 agent 能在相同环境中执行伙伴动作；随后把消息目标改成只对伙伴未来动作负责，并加入 message-on/off 的在线学习曲线。如果显式标签能提高伙伴回报而离散消息仍逐行零，结论就是信用分配和消息时序没有把私有需求传给行动者；只有在这个能力界通过后，才值得加入公共／私有码本、随机身份和新成员接入。
