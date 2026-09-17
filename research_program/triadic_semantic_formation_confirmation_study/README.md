# 独立语义转移确认

本目录冻结四个新种子（49301–49304）的 unique PI-live/PI-silent 从头训练，主机制读数为留出布局上的 `aligned−placebo source_pull`。计划见 [plan.md](plan.md)。训练结果、独立审计、消息移植、错配 placebo 和图表将在同一目录的运行树中保存。

这项确认只回答首窗消息是否对任务内容产生可识别的定向响应，不把成功率或字符串频数称作语言。只有跨新种子、跨留出背景且优于错配 placebo 的响应稳定后，才进入组合和代际实验。

本批次已经完成：8次独立训练、48个checkpoint和16个自然终点的训练审计通过；aligned 与 placebo 各完成1920行重放，placebo审计重放414720次live模块前向，均为 `max_abs_error=0`。PI-live 的 aligned `source_pull` 为 +10.2970 pp，placebo 为 +10.1656 pp，预注册的 aligned−placebo 为 **+0.1314 pp**，t(3) 区间 [−1.5910,+1.8538]，因此没有源端需求—消息对齐增量。完整结果见[确认实验报告](确认实验_结果与下一步.md)，哈希链见[`provenance_confirmation_001.json`](results/provenance_confirmation_001.json)。

同一批冻结策略的[槽位子集探针](slot_probe_结果与下一步.md)把四槽首窗消息按16个子集移植并独立重放30720行。PI-live单槽 `source_pull` 为+0.9169–+1.7383 pp，两槽为+3.5911–+6.8080 pp，三槽为+8.5802–+9.4617 pp，完整四槽为+10.2970 pp；这支持分布式整包／槽位交互的窄机制解释，仍不能称词义或组合语法。
