# v0.46 交叉适应日程与陌生主体恢复

本轮把 resident 的形成文化与 newcomer 的适应伙伴日程完全交叉，区分“协议在形成阶段是否对角色变化稳定”和“新人适应时看到了多少伙伴”这两个因素。实验使用本地 PyTorch 受控 agent，未调用 LLM 或外部 API；resident 与视觉前端冻结，只重置 identity 0 的通信模块。

## 设计

- 4 个 seed × 3 个视觉 partition × 6 个资源列排列 = 72 个视觉组；
- 6 类 resident 形成文化：静态/随机角色顺序 × 固定 A、轮换 A/B、随机 A/B/C；
- 3 种 newcomer 适应日程：固定 A、轮换 A/B、随机 A/B/C，完整交叉，共 1,296 条链；
- 每条链训练 300 次更新，检查点为 0、100、300；
- 每个检查点保存 3 种评估拓扑、4 个 team 和 6 种角色排列；
- 主要指标为 identity-012 等变 target-60 joint J，辅以 role spread、token pair agreement 和 position NMI。

正式设计见 [cross_schedule_design.json](cross_schedule_design.json)，结果见 [交叉适应日程与陌生主体恢复研究报告.md](results/cross_schedule_001/交叉适应日程与陌生主体恢复研究报告.md)。

## 已封存输出

- [training_complete.json](results/cross_schedule_001/training_complete.json)：1,296 条链、10,368 条轨迹、46,656 个协议文件；
- [cross_schedule_analysis.json](results/cross_schedule_001/cross_schedule_analysis.json)：独立 NumPy 重算及完整摘要；
- [cross_schedule_statistics.json](results/cross_schedule_001/cross_schedule_statistics.json)：72 个视觉组配对单位和 20,000 次 bootstrap；
- [cross_schedule_audit.json](results/cross_schedule_001/cross_schedule_audit.json)：独立动作、采样、分数和重放审计；
- [01_cross_schedule.png](results/cross_schedule_001/figures/01_cross_schedule.png)：六面板结果图；
- [结果审查.md](结果审查.md)：科学审查、限制和下一步。

## 复现实验

以下命令应按顺序执行。正式训练会产生大量 NPZ 文件，运行时间取决于本机。

```bash
../.venv/bin/python cross_schedule_train.py --out results/cross_schedule_001
../.venv/bin/python cross_schedule_analysis.py --out results/cross_schedule_001
../.venv/bin/python cross_schedule_statistics.py --out results/cross_schedule_001
../.venv/bin/python cross_schedule_audit.py --out results/cross_schedule_001
../.venv/bin/python plot_cross_schedule.py --out results/cross_schedule_001
../.venv/bin/python visual_qa.py --out results/cross_schedule_001
../.venv/bin/python build_cross_schedule_report.py --out results/cross_schedule_001
../.venv/bin/python build_review.py --out .
../.venv/bin/python package_results.py --out .
```

`results/cross_schedule_smoke` 只用于流程冒烟，不属于正式批次；正式封存目录是 `results/cross_schedule_001`。
