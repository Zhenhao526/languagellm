# v0.47 通信模块重置粒度与陌生主体恢复

本轮检验通信恢复是否主要依赖发送端、接收端，还是两端共同可塑。每个 newcomer 从同一 resident endpoint 复制 identity 0，只重置 sender 模块、receiver 模块或两者；适应伙伴日程固定为 v0.46 中表现最好的随机 A/B/C。视觉前端和 resident 参数冻结，未重置的通信模块保持 resident endpoint 参数。

## 设计

- 4 个 seed × 3 个视觉 partition × 6 个资源列排列 = 72 个视觉组；
- 6 类 resident 形成文化：静态/随机角色顺序 × 固定 A、轮换 A/B、随机 A/B/C；
- 3 种重置模式：`sender_only`、`receiver_only`、`both`；
- 完整交叉共 1,296 条链，每条 300 次更新，检查点 0、100、300；
- 每个检查点保存 3 种评估拓扑、4 个 team 和 6 种角色排列。

正式设计见 [reset_granularity_design.json](reset_granularity_design.json)，研究结果见 [通信模块重置粒度与陌生主体恢复研究报告.md](results/reset_granularity_001/通信模块重置粒度与陌生主体恢复研究报告.md)。

## 复现

```bash
../.venv/bin/python reset_granularity_train.py --out results/reset_granularity_001
../.venv/bin/python reset_granularity_analysis.py --out results/reset_granularity_001
../.venv/bin/python reset_granularity_statistics.py --out results/reset_granularity_001
../.venv/bin/python reset_granularity_audit.py --out results/reset_granularity_001
../.venv/bin/python plot_reset_granularity.py --out results/reset_granularity_001
../.venv/bin/python visual_qa.py --out results/reset_granularity_001
../.venv/bin/python build_reset_granularity_report.py --out results/reset_granularity_001
../.venv/bin/python build_review.py --out .
../.venv/bin/python package_results.py --out .
```

`results/reset_granularity_smoke` 是 5-update 冒烟运行，不属于正式批次。
