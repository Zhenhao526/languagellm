# 语言形成预实验：河谷补给与协作搬运

这是对 `qwen_collect_pilot` 两字符采集任务的重设，旧代码和结果保留在原目录。实现依据[任务与互动方案](../qwen_collect_pilot/语言形成实验_v0.2_任务与互动协议.md)。本目录使用同一份已部署的 Qwen3.5-9B-8bit 权重，不重复下载。

2026-09-14本轮已全部结束：两个符号条件各连续2段，共48步；两条件第1段均完成1/2，第2段均完成2/3，木材交付均未完成。完整规则下的两项自然语言能力控制也只完成1/2。先看[结果与下一步](results/final_summary_20260914/结果与下一步.md)、[符号实验回放](review_outputs/symbolic_final/replay.html)和[完整数据报告](results/final_summary_20260914/report.md)。当前结果用于任务与模型诊断，尚未验证稳定共同语义或组合规则。

三个持续存在的主体只有各自的局部观察和互动记忆。每个物理动作步有四个同步广播窗口，随后独立提交动作、同时结算。广播由 `@#%&*+=~` 组成，单条最多32符号，每人每步最多64符号。即时条件逐窗口公开消息；延迟条件在四个窗口结束后一次公开。模型参数固定，每次调用重新创建推理缓存。

环境需要区分带属性的物品、寻找工具、加工木材、配合搬运和交付目标。物品操作句柄对每人独立；共同搬运需要双方分别提交匹配对象、搭档和去向。语言文字仅用来说明世界、记录个人观察和私有分析；符号主条件不共享自然语言分析、词典或计划。

## 运行

在项目根目录执行：

```sh
qwen_collect_pilot/.venv/bin/python -m pytest qwen_language_v2/tests -q
qwen_collect_pilot/.venv/bin/python -m qwen_language_v2.run --phase calibration --conditions natural --groups 17 --episodes 1
```

本轮已在符号组启动前锁定：每条件1个三主体群体、各连续2段，场景variant为0与31。参数与缩小首轮规模的理由见[锁定预算](pilot_budget_locked.json)及[执行记录](执行记录.md)。启动一次独立复跑可执行：

```sh
/usr/bin/caffeinate -i qwen_collect_pilot/.venv/bin/python -m qwen_language_v2.run --phase pilot --conditions immediate,delayed --groups 17 --episodes 2 --variants 0,31
```

默认生成新时间戳目录；指定`--run-dir`时要求该目录尚不存在。`run.py --help`查看参数。已有结果不覆盖。

输出目录保存执行源码、场景与可行见证、模型每次调用、每步局部观察与广播、私有历史检查点、完成度及描述性报告。研究日志中的全局快照不作为符号组的额外输入；完整信息能力控制通过单独接口观察当前完整状态。私有分析在同一次决定内用于生成正式输出，原文不作为跨决策可读记忆，详见[记忆说明](私有记忆与语言形成_解释边界.md)。

可用`python3 -m qwen_language_v2.replay_snapshot RUN_DIR --out OUTPUT/replay.html`生成离线回放。页面区分研究者全局状态、主体实际观察、正式广播和环境反馈，不读取私有分析。[完整能力控制回放](review_outputs/complete_rules_final/replay.html)及[最终符号回放](review_outputs/symbolic_final/replay.html)均已通过320与736像素宽度的全步骤浏览器检查。本轮程序验证为56项测试及206项子检查通过。

## 解释范围

即时和延迟条件比较行动前相互回应的机会；两者在行动前都能读取全部正式消息。自然语言和完整信息控制检查任务能力，独立于符号群体，也不视为通信带宽匹配的对照。

任务完成度、符号长度和字符串重复是描述指标。它们不能独自证明共同语义、组合性、关系表达或误解修复。后续语言证据须来自保留情境、冻结历史副本及针对候选片段的干预。Qwen 的既有语言经验与动作接口提供的概念结构均属于本实验前提。
