# 三主体语言形成工程先导：运行总报告

汇总时间：2026-09-14T22:42:37.407975+08:00

本报告汇总实际记录的能力检查与符号实验。它描述合作任务、动作和通信使用，不宣布语言已经形成。

## 运行与版本范围

| 运行 | phase | 条件 | 群体 | 当前状态 |
|---|---|---|---|---|
| calibration_natural_20260914 | calibration | natural | 17 | completed |
| calibration_clarified_20260914 | calibration | natural, full_information | 17 | stopped_for_rule_clarity |
| calibration_complete_rules_20260914 | calibration | natural, full_information | 17 | completed |
| pilot_immediate_delayed_20260914 | pilot | immediate, delayed | 17 | completed |

每次运行的配置和源码快照各自保存；不同校准版本不能合并成同一训练过程。表中群体编号在不同条件下代表各自独立的私有历史。

natural为局部观察下的自然语言能力控制；full_information为完整当前物理信息下的独立自然语言能力控制，均不与符号条件匹配带宽。

| 运行 | 核心源码快照摘要 | 相对上一列出运行的源码差异 | 模型revision记录 |
|---|---|---|---|
| calibration_natural_20260914 | 7a701acaf57a | 首个列出运行 | 16daa4818c54ce5f5436f929d52542eb65bbed9d |
| calibration_clarified_20260914 | e734d800b08f | agents.py, backend.py, environment.py, run.py | 16daa4818c54ce5f5436f929d52542eb65bbed9d |
| calibration_complete_rules_20260914 | 1ee0b80bb4a8 | backend.py, environment.py | 16daa4818c54ce5f5436f929d52542eb65bbed9d |
| pilot_immediate_delayed_20260914 | 1ee0b80bb4a8 | 已能比较的文件相同 | 16daa4818c54ce5f5436f929d52542eb65bbed9d |

摘要只用于识别已保存的核心源码版本；文本差异不自动意味着物理引擎行为不同，也不是对模型权重文件的再次校验。完整逐文件SHA256保存在summary.json。

**calibration_clarified_20260914已为补齐规则说明而手动中止。** 仅报告停止前已记录动作；未完成任务段不作为已完成的失败结果，未启动条件不计实验结果。

## 能力检查（calibration）

| 运行 | 条件 | 群体 | 段 | variant | 最终完成度 | 最终步数 | 已记录步 | 记录状态 |
|---|---|---|---|---|---:|---:|---:|---|
| calibration_natural_20260914 | natural | 17 | 1 | 0 | 0.0% | 12 | 12 | 已有任务段完成记录 |
| calibration_clarified_20260914 | full_information | 17 | 1 | 0 | — | — | — | 尚无完整调用或动作记录 |
| calibration_clarified_20260914 | natural | 17 | 1 | 0 | — | — | 5 | 已中止，任务段未完成 |
| calibration_complete_rules_20260914 | full_information | 17 | 1 | 0 | 50.0% | 12 | 12 | 已有任务段完成记录 |
| calibration_complete_rules_20260914 | natural | 17 | 1 | 0 | 50.0% | 12 | 12 | 已有任务段完成记录 |

没有任务段完成记录时，最终完成度和最终步数留空；已记录步数只表示执行进度。任务超时但正常写入完成记录时，其实际完成度仍计入。

## 符号工程先导

| 运行 | 条件 | 群体 | 段 | variant | 最终完成度 | 最终步数 | 已记录步 | 记录状态 |
|---|---|---|---|---|---:|---:|---:|---|
| pilot_immediate_delayed_20260914 | delayed | 17 | 1 | 0 | 50.0% | 12 | 12 | 已有任务段完成记录 |
| pilot_immediate_delayed_20260914 | delayed | 17 | 2 | 31 | 66.7% | 12 | 12 | 已有任务段完成记录 |
| pilot_immediate_delayed_20260914 | immediate | 17 | 1 | 0 | 50.0% | 12 | 12 | 已有任务段完成记录 |
| pilot_immediate_delayed_20260914 | immediate | 17 | 2 | 31 | 66.7% | 12 | 12 | 已有任务段完成记录 |

没有任务段完成记录时，最终完成度和最终步数留空；已记录步数只表示执行进度。任务超时但正常写入完成记录时，其实际完成度仍计入。

## 动作与实际发生的过程

| 运行 | 条件/群体/段 | 等待 | 失败的非等待动作 | 曾携带物品 | 拿取成功/尝试 | 加工成功/尝试 | 交付成功动作/尝试 | 新交付单位 |
|---|---|---:|---:|---|---:|---:|---:|---:|
| calibration_natural_20260914 | natural/17/1 | 22 | 0 | 未观察到 | 0/0 | 0/0 | 0/0 | 0 |
| calibration_clarified_20260914 | full_information/17/1 | 0 | 0 | 未知 | — | — | — | — |
| calibration_clarified_20260914 | natural/17/1 | 8 | 1 | 观察到 | 2/2 | 0/0 | 0/1 | 0 |
| calibration_complete_rules_20260914 | full_information/17/1 | 5 | 12 | 观察到 | 4/4 | 1/1 | 1/8 | 1 |
| calibration_complete_rules_20260914 | natural/17/1 | 12 | 7 | 观察到 | 4/8 | 1/1 | 1/4 | 1 |
| pilot_immediate_delayed_20260914 | delayed/17/1 | 12 | 1 | 观察到 | 3/3 | 1/1 | 1/2 | 1 |
| pilot_immediate_delayed_20260914 | delayed/17/2 | 5 | 3 | 观察到 | 8/10 | 0/0 | 2/3 | 2 |
| pilot_immediate_delayed_20260914 | immediate/17/1 | 9 | 6 | 观察到 | 3/3 | 1/1 | 1/5 | 1 |
| pilot_immediate_delayed_20260914 | immediate/17/2 | 6 | 5 | 观察到 | 4/4 | 0/0 | 2/4 | 2 |

拿取包括工具，不能把拿到斧具直接记为完成资源交付。成功与尝试分别列出；仅发出拿取、加工或交付动作不等于已完成该操作。曾携带来自实际状态或观察，也可能由交接或共同搬运产生。

等待单列，不因未交付而把等待记成执行失败。失败只计明确反馈action_succeeded=false的非等待动作；缺失反馈不算失败。共同交付可产生两人的成功动作记录，但交付单位从状态变化计算，不重复算两份。未完成段的动作也按已有记录展示。

- **calibration_natural_20260914**：move=14，wait=22。缺少布尔执行反馈的动作0次。
- **calibration_clarified_20260914**：deliver=1，move=4，pickup=2，wait=8。缺少布尔执行反馈的动作0次。
- **calibration_complete_rules_20260914**：carry_together=5，cut=2，deliver=9，deliver_together=3，drop=2，move=22，pickup=12，wait=17。缺少布尔执行反馈的动作0次。
- **pilot_immediate_delayed_20260914**：carry_together=2，cut=2，deliver=9，deliver_together=5，drop=5，move=69，pickup=20，wait=32。缺少布尔执行反馈的动作0次。

## 符号长度与额度检查

下列统计直接引用各运行现有analysis.json，不把自然语言控制按符号字母表判为非法。统计以该分析器已纳入的任务段为准，运行中可能尚不包含未结束任务段。

| 运行 | 阶段/条件/群体 | 消息数 | 平均字符数 | 非空比例 | 超32字符 | 每人每步超64 | 非法字符 |
|---|---|---:|---:|---:|---:|---:|---|
| pilot_immediate_delayed_20260914 | pilot/delayed/17 | 288 | 5.25 | 100.0% | 0 | 0 | {} |
| pilot_immediate_delayed_20260914 | pilot/immediate/17 | 288 | 5.50 | 100.0% | 0 | 0 | {} |

触及上限不等于存在被截去的语义内容；长度、重复和相邻窗口变化不作语言结构判据。

## 调用、内存与耗时

| 运行 | 已记录完整调用 | 其中输出校验失败 | MLX峰值GB | 未受已知中断影响的调用 | 这些调用耗时合计秒 | 中断调用 |
|---|---:|---:|---:|---:|---:|---|
| calibration_natural_20260914 | 360 | 0 | 10.82 | 359 | 1456.23 | 249 |
| calibration_clarified_20260914 | 161 | 0 | 10.48 | 161 | 537.02 | 无已知标记 |
| calibration_complete_rules_20260914 | 720 | 0 | 11.11 | 720 | 4069.61 | 无已知标记 |
| pilot_immediate_delayed_20260914 | 1440 | 0 | 10.93 | 1440 | 6344.54 | 无已知标记 |

- **calibration_natural_20260914**：模型加载记录1.41秒；call 249受SIGSTOP/供电影响，已从上表其余调用耗时合计中排除。中断调用的有效推理时间无法准确还原；epoch暂停时长与perf_counter时长不可直接相减。原始backend推理耗时记录为2889.83秒，含暂停影响，不标作有效推理总时长。 已完成任务段原始耗时合计2890.11秒（含暂停影响）。
- **calibration_clarified_20260914**：模型加载记录1.86秒；原始backend推理耗时记录537.02秒。 已完成任务段原始耗时合计—秒。 该运行中止时仍有未完成任务段；尚未完成并写入日志的调用耗时未包含在完整调用合计内。
- **calibration_complete_rules_20260914**：模型加载记录1.25秒；原始backend推理耗时记录4069.61秒。 已完成任务段原始耗时合计4070.36秒。
- **pilot_immediate_delayed_20260914**：模型加载记录1.97秒；原始backend推理耗时记录6344.54秒。 已完成任务段原始耗时合计6345.73秒。

调用耗时合计不是整个实验的墙钟总耗时；未完成调用尚未写入日志时不计入完整调用数。token分模式统计和原始计时字段保存在summary.json。

## 冻结状态检查与语言证据

- pilot_immediate_delayed_20260914：[冻结状态探针报告](</Users/xia/Documents/ChatGPT/语言/qwen_language_v2/results/pilot_immediate_delayed_20260914/frozen_probe_ep1_step4/report.md>)。
  记录2个冻结案例；原动作未完全重现0例，需结合该报告的版本与复现检查解释。

若已有本步消息文字置空检查，它最多支持该文本对给定状态下一步行动的影响；过去消息仍保留，长度也变化，不能当作整段无通信基线或语义、组合性确认。

下列证据本汇总未确认；只有另行实施并报告相应检查后才能作更强结论：

- 跨情境的共同意义及稳定理解，且区分发送者编码与伙伴理解。
- 预先固定留出组合中的首次理解，排除测试中重新协商。
- 在独立确认情境中对熟悉片段替换/重组所得的选择性行为改变。
- 成对情境中的关系与参与者敏感性，排除整场景编号。
- 对特定可观察误解的回应干预及选择性修复。
- 新成员学习、文化传递和代际变化。

同一条件只有一个群体时，没有群体层面的独立重复。多任务段提供过程记录，不能补成多个独立群体。即使任务完成度不同，也可能来自任务理解、动作执行、规划或通信使用差异。

## 原始资料与数据质量

- **calibration_natural_20260914**：[运行配置](</Users/xia/Documents/ChatGPT/语言/qwen_language_v2/results/calibration_natural_20260914/manifest.json>)；[单运行报告](</Users/xia/Documents/ChatGPT/语言/qwen_language_v2/results/calibration_natural_20260914/report.md>)；[字符串统计JSON](</Users/xia/Documents/ChatGPT/语言/qwen_language_v2/results/calibration_natural_20260914/analysis.json>)。
- **calibration_clarified_20260914**：[运行配置](</Users/xia/Documents/ChatGPT/语言/qwen_language_v2/results/calibration_clarified_20260914/manifest.json>)；[单运行报告](</Users/xia/Documents/ChatGPT/语言/qwen_language_v2/results/calibration_clarified_20260914/report.md>)；[字符串统计JSON](</Users/xia/Documents/ChatGPT/语言/qwen_language_v2/results/calibration_clarified_20260914/analysis.json>)。
  读取提示：原分析器：episodes.jsonl不存在；尚无可分析的任务段记录。
- **calibration_complete_rules_20260914**：[运行配置](</Users/xia/Documents/ChatGPT/语言/qwen_language_v2/results/calibration_complete_rules_20260914/manifest.json>)；[单运行报告](</Users/xia/Documents/ChatGPT/语言/qwen_language_v2/results/calibration_complete_rules_20260914/report.md>)；[字符串统计JSON](</Users/xia/Documents/ChatGPT/语言/qwen_language_v2/results/calibration_complete_rules_20260914/analysis.json>)。
- **pilot_immediate_delayed_20260914**：[运行配置](</Users/xia/Documents/ChatGPT/语言/qwen_language_v2/results/pilot_immediate_delayed_20260914/manifest.json>)；[单运行报告](</Users/xia/Documents/ChatGPT/语言/qwen_language_v2/results/pilot_immediate_delayed_20260914/report.md>)；[字符串统计JSON](</Users/xia/Documents/ChatGPT/语言/qwen_language_v2/results/pilot_immediate_delayed_20260914/analysis.json>)。
