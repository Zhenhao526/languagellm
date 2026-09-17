# v0.19 冻结视觉接口的新私人行动学习

完成96个新私人行动头的固定预算训练，来自四个既有来源。仅行动成败奖励下，保留接口与重置后幅度匹配接口的未训练组合贪心双目标成功率为97.624%和96.441%，主差+1.183个百分点，四来源均正。第100步差17.437个百分点，过程AUC差2.580个百分点；保留接口的读出学得更快，但重置端也能达到较高私人泛化。原生随机双成功Q反而为87.903%和90.118%，两种决策指标不能混用。

- [完整研究报告](results/readout_001/冻结视觉接口的新私人行动学习研究报告.md)、[结构化分析](results/readout_001/analysis.json)、[结果解释](结果解释.md)
- [学习曲线 PNG](results/readout_001/figures/01_private_learning.png) / [PDF](results/readout_001/figures/01_private_learning.pdf)、[配对终点 PNG](results/readout_001/figures/02_paired_sealed_readout.png) / [PDF](results/readout_001/figures/02_paired_sealed_readout.pdf)
- [固定方案](固定执行方案.md)、[前置审查](前置审查.md)、[科学解释与未执行候选](科学解释界限.md)
- [运行器](run_readout.py)、[新头与损失](readout_interface.py)、[冻结源码与输入](results/readout_001/invocation.json)、[训练完成凭证](results/readout_001/training_complete.json)
- [完整开发预检](preflight_qa.json)、[正式独立执行审计](results/readout_001/audit_execution.json)、[独立原始复算](results/readout_001/independent_recount.json)、[分析比较](results/readout_001/analysis_comparison.json)
- [图表核验](results/readout_001/figure_qa.json)、[报告读审](results/readout_001/audit_report_review.json)、[同期近邻文献](../paper_program/literature_capability_20260916/短近邻定位.md)

本轮冻结官方DINOv2 ViT-L/14及所有视觉接口，只训练10,476参数的私人行动头；没有新视觉主干推理、接口训练或共同通信。两接口 × 奖励/正确地点监督（CE）两信号，共四条件，同一来源使用相同全新头初值、世界与预算。训练仅old18，added6与sealed6本轮都不进入训练。每头2400次更新，正式训练约304秒，CPU单线程；审计、分析和文档另计。

J表示同一个私人头分别应对两种需求均选对地点，不是通信成功率。每次评价使用原test池的8×8照片对，共1920世界；不能直接减去旧私人头每图16照片对的分数。四来源内先平均三分区和两人，独立统计外层仍为四个开发来源，没有等效或显著性结论。新头和CE标签均不回灌既有社会主体。

查看结果无需重跑。复跑需要既有权重、原特征和相同本地依赖，输出目录必须尚不存在：

```sh
redesign_v0.3/deployment/.venv/bin/python redesign_v0.19/run_readout.py --out redesign_v0.19/results/readout_reproduction
```

新头恢复较高行动表现，限制了把重置接口解释为缺乏基本资源定位能力的说法；它没有证明语言形成机制。重训读出、视觉准备与通信落差已有直接文献先例。下一候选回到可独立验证的事件保持与更新能力，目前序列环境及新社会训练尚未执行。旧负结果、短开发原方案与历史索引均保留，完整论文目标仍在进行。
