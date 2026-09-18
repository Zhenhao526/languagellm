# 群体感知异质性实验同步与清理

本轮正式九种子矩阵已同步到 `git@github.com:Zhenhao526/language.git`。实验包含 36 个 parent、432 个 child（48 个 child 条件，每个 run 3,000 次更新）。三片独立 replay audit 均通过，合计重放 36,000 条 parent 与 432,000 条 child 训练记录、48/576 个 parent/child checkpoint，以及各 216,000 行 live/silent 和 identity/swap 配对流，最大绝对回放误差为 0。

远端保留：

- `research_program/heterogeneous_grounding_study/` 的源代码、冻结方案、紧凑聚合、片级审计和诊断归档；
- 正式 `results.json` 的完整终点 payload，SHA-256 为 `18a60dfd1b7625c0c68560e5f68d2c24f0b62d74f8124b288b9d2c81bd37cbc7`。

本地保留：正式 `aggregate.json`/`aggregate.md`、三片 audit、freeze/plan/prepared、pilot 曲线、validity、源代码和本报告。正式完整 `results.json` 已在远端可复核后从本地删除；训练日志、checkpoint 和临时执行树此前已删除。`manifest.json` 明确记录了 remote-only payload 及其哈希，清理收据见 `LOCAL_CLEANUP_RECEIPT_2026-09-18_heterogeneous_formal.json`。

群体异质性结论按可识别性边界表述：parent 仍形成有因果作用的消息协议，但单目标 worker 替换不会把非目标伙伴的异质约定暴露给 child，因此下一版需加入跨伙伴 incumbent 迁移或非目标伙伴服务任务。
