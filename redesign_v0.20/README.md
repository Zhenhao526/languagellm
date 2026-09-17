# v0.20 资源移动后的私人记忆与更新

48次私人主体训练已完成，来自四个新初始化来源。环境先显示完整资源场景，再移动一个资源并只显示其新位置。完整跨步反传与截断梯度组的共同30种终态组合双目标成功率为92.391%和91.481%，主差+0.911个百分点，四来源均正。两组都已利用历史；清零历史后分别为20.000%和19.983%，同末帧不同历史的配对成功均降为0。

预定整批能力验收未通过：old延迟成功率差仅+0.047个百分点，未达10个百分点，四来源中两项为0。截断梯度保留前向记忆，不能作为“无记忆”组。本轮不启动以此代表强弱能力的社会对照，也没有新共同符号结果。

- [完整报告](results/temporal_001/资源移动后的私人记忆与更新研究报告.md)、[结构化分析](results/temporal_001/analysis.json)、[结果解释](结果解释.md)
- [任务示意 PNG](task_figure/00_temporal_task.png) / [PDF](task_figure/00_temporal_task.pdf)、[学习曲线 PNG](results/temporal_001/figures/01_temporal_learning.png) / [PDF](results/temporal_001/figures/01_temporal_learning.pdf)、[历史配对 PNG](results/temporal_001/figures/02_paired_history_capability.png) / [PDF](results/temporal_001/figures/02_paired_history_capability.pdf)
- [固定执行方案](固定执行方案.md)、[环境与能力审查](环境与能力验收_独立审查.md)、[近邻与因果边界](近邻与因果边界.md)
- [运行器](run_temporal.py)、[时间接口与损失](temporal_model.py)、[世界与指标](temporal_world.py)、[输入及源码绑定](results/temporal_001/invocation.json)、[训练完成凭证](results/temporal_001/training_complete.json)
- [完整开发预检](preflight_qa.json)、[执行审计](results/temporal_001/audit_execution.json)、[投影参数实际指纹](results/temporal_001/prepared_project_fingerprints.json)、[原始结果遍历复算](results/temporal_001/independent_recount.json)、[独立公式分析源码](analyze_temporal.py)、[统计比较](results/temporal_001/analysis_comparison.json)
- [图表核验](results/temporal_001/figure_qa.json)、[报告读审](results/temporal_001/audit_report_review.json)、[旧封存完整性](prior_integrity_qa.json)

保留官方冻结DINOv2 ViT-L/14的原60行缓存特征，未使用LLM或新增视觉主干推理。四来源各两人先完成资源后果准备，然后冻结project。每人每分区复制为完整/截断两组，训练memory、slot_phi及全新私人行动头，共155,598个参数。三分区嵌套在四来源内，不作为12个独立来源。

训练初态和终态只用old18；评价初态仍用old18，终态覆盖共同30。评价行按终态平衡，每模型每模式5760行；事件重复只实现权重。共同30仅末帧策略的双目标上界为20%；单独added或sealed可达100%，所以单组高分不能证明历史利用。J是同一私人主体两需求都行动正确，Q是其原生随机行动双成功期望；均非通信成功率。

正式完成115,200次更新、58,982,400次单目标行动，前置资源准备另计1,600次更新、102,400次行动。训练约472.76秒，CPU单线程。执行审计独立重放每个主体首步及第2101步的参数与Adam更新，检查全部评价世界表并抽样重放每文件首256/末128行前向；没有重跑全部训练。原始遍历复用冻结环境指标函数，独立分析另写公式，两者54,425项比较最大差0。

查看结果不需要重跑。以下命令使用既有缓存与本地环境，输出目录必须不存在：

```sh
redesign_v0.3/deployment/.venv/bin/python redesign_v0.20/run_temporal.py --out redesign_v0.20/results/temporal_reproduction
```

下一项可另行固定环境信息条件比较：让具备行动能力的共同起点在“直接看见”或“需要历史”的任务中学习新协议，检验自然消息和正确行动是否随合法历史变化。这是尚未执行的新问题，不能绕过本轮失败的能力对照，也不能仅凭任务更复杂宣布新颖性。当前仍缺独立视觉确认和足以支撑论文中心贡献的证据，整体研究目标继续。
