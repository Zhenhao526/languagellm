# 部分成功收益与伙伴选择

4个新配对初始化×2收益设置×3条件的24组训练已完成，独立实现复核通过。降低部分成功效用0.5→0.1后，PL开放双留出满分率23.0025%→27.9915%、内容两端适切16.9211%→24.4861%，但全部政策仍固定伙伴；唯一主要角色差中差四个均为0。

- [结果与下一步](结果与下一步.md)：主量、辅助结果、全部初始化、形成曲线及解释边界。
- [训练前固定计划](plan.md)、[正式冻结](results/payoff_001/plan.json)、[主结果](results/payoff_001/execution/results.json)。
- [独立实现复核](results/payoff_001/audit_execution_001/verification.json)、[审计预检](audit_preflight_001.json)。
- [全量描述汇总](results/payoff_001/summary_001/summary.json)、[角色转移](results/payoff_001/summary_001/role_transitions.json)、[全部终点](results/payoff_001/summary_001/all_endpoint_cells.json)。
- [三图及绘图数据](results/payoff_001/figures_001/receipt.json)、[PNG实际检查](results/payoff_001/figures_001/visual_review_001.json)。
- [下一检验的取舍](next_step_review.md)：冻结转录角色解码与互动机制审查，尚未执行。

只读验证正式冻结：

```sh
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_partial_payoff_study.runner verify --out research_program/triadic_partial_payoff_study/results/payoff_001
```

`utility.py`改变部分成功效用，原物理结算与旧冻结源保持不变。`runner.py`执行一次固定批次，`audit_execution.py`另实现结算/统计并独立重放完整末点，`summarize_results.py`只读全部实际NPZ，`plot_results.py`只读完成的JSON。输出目录拒绝覆盖；重做必须选新目录，已有结果不能被当作新运行。无新增Qwen推理，不涉及另一任务的视觉实验。

审计与效用模块由同一代理分别实现，非作者独立；不重放优化器或中间检查点前向。4初始化是独立单位，世界、检查点和静默引用不是独立重复。下一步不能从任务成功率直接跳到词义或组合语言的主张。
