# Open-world expansion formal archive (2026-09-18)

本归档是开放世界扩展实验的正式紧凑结果。父代只训练值 `0,1,2` 的 3×3 意义；子代首次接触值 `3`。消息为两个四值 slot，完整 4×4 意义空间与消息容量相等。`expanded_single_alternating` 和 `expanded_single_pair` 都把双新值 `(3,3)` 完全留出，因而其 `new-double` 端点是严格的零样本组合检验。

正式矩阵包含 54 个 parent、135 个 child、9 个 seed。`combined_audit.json` 的三片重放合计 162,000 条 parent 和 405,000 条 child 训练日志、216/540 个 checkpoint、405,000/162,000 条 architecture/support 配对行，最大回放误差为 0，状态为 `passed`。训练来自 v3 执行树；v4 只修正了 role=2 fresh sender 的重组评估，并用相同 checkpoint 重算自然与重组终点。

在排除 `(3,3)` 训练的 `expanded_single_pair` 中：

- `factorized`：new-single 0.996（9/9），new-double 1.000（9/9）；
- `tied_routed`：new-single 0.513（4/9），new-double 0.988（9/9）；
- `holistic`：new-single 0.616（4/9），new-double −0.250（0/9）。

对于 `factorized` 和 `tied_routed`，double-new 的自然消息与由两个 single-new donor 重组的消息逐 seed 完全一致；这排除了直接记忆 `(3,3)` 作为主要解释。fresh–fresh 相对 alternating 的 double-new 差值分别为 +0.887（factorized，95% CI [0.708, 1.066]）、+1.068（tied_routed，[0.924, 1.212]）和 −0.230（holistic，[−0.374, −0.085]）。结果支持一个受限机制命题：结构化组合坐标需要新原子值的学习信号和双方共享的后果窗口，才能恢复未见的新联合意义。它不证明 holistic 主体自主发明了人类语言。

目录内容：`results.json` 是逐 run 的紧凑 endpoint；`aggregate.*`、`seed_analysis.*`、`support_contrasts.*` 是汇总；`combined_audit.json` 和 `part_audits/` 是 replay 收据；`prepared.json`、`plan.json`、`freeze.json` 与 `source_snapshot/` 保存冻结设计和源码；`invalid_design_notes.md` 记录 v1/v2 诊断批次为何不能作正式证据。

原始训练日志、checkpoint、临时执行树和 pilot 输出在归档校验后删除；`manifest.json` 列出保留载荷的 SHA-256。
