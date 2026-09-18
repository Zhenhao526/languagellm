# 可学习 routing 因子化正式批次同步与清理

本轮在已经完成的社区冲突 referential game 上加入 `routed` policy：原子 factor table 保留，但 sender 与 receiver 的 slot—attribute routing 由 payoff 学习。正式冻结设定为 routing-logit 初始标准差 0.5、softmax 温度 5；低梯度近均匀 routing 另做敏感性基线。源代码提交为 `4577b32`，正式归档位于：

- `research_program/routed_factorization_study/results/formal_20260918_scaled/`
- `research_program/routed_factorization_study/results/formal_20260918_low_gradient/`

scaled 与 low-gradient 两套矩阵均为 9 个 seed、108 个 parent、324 个 child、每个 run 3,000 updates。两套独立 replay audit 均通过，分别重放 324,000 条 parent 和 972,000 条 child 日志，432/1,296 个 checkpoint，最大 replay error 均为 0.0。

scaled 的 aligned/hidden held-out-combination 结果为：fixed `factorized` 1.000（9/9 functional），`holistic` −0.156（0/9），`routed` 0.245（1/9）。routed 的 sender/receiver routing alignment 均值约 0.717/0.728，但 seed 间不稳定；部分 seed 获得 near-permutation，部分 seed 把两个 slot 指向同一属性或让 sender/receiver 路由错位。held-out-value return 为 −0.020。low-gradient 基线的 routed held-out-combination 约 −0.016、route alignment 约 0.50，记录了优化盆地敏感性。

这轮支持的机制边界是：显式 factor sharing 可以稳定传递已见原子值的组合，而可学习 routing 即使形成部分槽位对齐，也不能仅凭当前 payoff 稳定得到组合泛化；槽位对齐不是组合语义的充分条件。该结果不把 tabular code 解释为人类语言，也不把 routing 失败直接归因于生态压力不足。

归档 manifest、冻结方案、紧凑 endpoint、种子级分析和 audit 收据均已写入 Git。原始 training logs、checkpoints、pilot、diagnostic 和 prepared 临时树在审计通过后删除；本地清理收据为 `LOCAL_CLEANUP_RECEIPT_2026-09-18_routed_factorization.json`。远端提交哈希将在本轮归档提交后补写。
