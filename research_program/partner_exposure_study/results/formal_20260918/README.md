# 多伙伴发送者对齐正式归档

本轮修正了上一轮 target-worker child 不可识别的问题：child 替换为 fresh sender，每个训练 batch 实际混合四个轮换伙伴。parent 先在均质或异质表面约定下形成协议；child 交叉 hidden/visible 伙伴身份、sender-only/coadapt 适应方式和 full/leave-one-out 支持。

正式矩阵为 9 个 seed、18 个 parent、144 个 child。每个 run 训练 3000 次更新；合计 54,000 条 parent 和 432,000 条 child 训练日志，72/576 个 checkpoint。独立 replay 审计重放全部更新，visibility、adaptation、population、support 配对流均通过，最大绝对误差为 0。

远端保留完整 `results.json`；本地只保留冻结方案、聚合统计和审计收据。原始训练日志、checkpoint、临时执行树和本地完整 payload 均在核验后删除。这里的语言仍指有限离散任务协议，不表示出现人类语法或开放词汇。

source commit: `11582a826ce829d7fc30c8c0d7856340a0f0f971`
