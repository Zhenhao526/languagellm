# 概率覆盖补充分析

先读[结果与下一步](结果与下一步.md)和[固定方案](plan.md)。本轮只重算前轮保存数值，没有新训练。

- [主准备清单](results/probability_001/plan.json)、[完整结果](results/probability_001/execution/results.json)及逐条记录目录。
- [独立数值审计](results/probability_001/audit_probability_001/独立核验.md)与[解释审查](概率结果_解释审查.md)。
- [轨迹图](results/probability_001/figures/probability_trajectory.png)及[数据／SHA收据](results/probability_001/figures/receipt.json)。
- [后续候选](独立确认候选方案.md)与[直接近邻核查](角色更新次序_近邻核查.md)：作为有限机制控制保留，未升级为正式确认，未冻结、未执行。
- [新增文献索引](neighbor_papers/下载索引.md)：两份成功下载、两份既有文件复用核验，另保存Lu2020两次DNS失败。
- [需求时点与既有实验核对](需求时点_既有实验核对.md)：v0.5已有已知／隐藏需求配对，单纯延后独立抽样不构成新的信息操纵。

实际完成的主流程（工作目录为项目根目录）：

```sh
redesign_v0.3/deployment/.venv/bin/python research_program/probability_coverage_study/analyze.py prepare --out research_program/probability_coverage_study/results/probability_001
redesign_v0.3/deployment/.venv/bin/python research_program/probability_coverage_study/analyze.py execute --out research_program/probability_coverage_study/results/probability_001
research_program/.plotting_venv/bin/python research_program/probability_coverage_study/plot_results.py
```

这些输出目录已经存在，脚本拒绝覆盖；以上是执行记录，不是要求重跑。来源与新分析源码均绑定 SHA，原训练／探针／修正历史保留。连续概率为温度1的策略参照，不是经校准的人类行为概率，也不是完整随机发送评价。
