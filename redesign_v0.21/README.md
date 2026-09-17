# v0.21 资源可见性与新共同通信

已完成24次双主体社会训练，来自四个v0.20 full私人准备来源。完整终态可见与局部终态＋历史两条件的共同30自然双目标J为65.230%和63.073%；唯一主差为局部减完整−2.157个百分点，四来源三负一正。熟悉终态约98%/97%，未训练12图只有16.081%/11.936%。

局部条件已经能用离散消息传递历史信息：同末帧、不同静止资源位置的历史配对，两次静止目标均正确为54.499%；局部条件内置换整条消息后，静止资源准确率由74.681%降至19.950%。但这不证明组合语法或一般语言能力。

- [完整研究报告](results/communication_001/资源可见性与共同通信形成研究报告.md)、[结构化独立分析](results/communication_001/analysis.json)、[结果解释](结果解释.md)
- [形成曲线 PNG](results/communication_001/figures/01_social_learning.png) / [PDF](results/communication_001/figures/01_social_learning.pdf)、[来源配对终点 PNG](results/communication_001/figures/02_paired_social_endpoints.png) / [PDF](results/communication_001/figures/02_paired_social_endpoints.pdf)
- [固定执行方案](固定执行方案.md)、[独立前置审查](前置审查.md)、[设计取舍](设计取舍.md)、[近邻文献](近邻与主要比较.md)
- [运行器](run_social.py)、[通信接口与损失](social_model.py)、[接口说明](通信接口说明.md)、[世界表](social_world.py)、[生产指标](social_metrics.py)
- [输入与源码绑定](results/communication_001/invocation.json)、[完整训练凭证](results/communication_001/training_complete.json)、[开发预检](preflight_qa.json)、[正式执行审计](results/communication_001/audit_execution.json)
- [独立公式源码](analyze_social.py)、[数值比较](results/communication_001/comparison.json)、[图表核验](results/communication_001/figure_qa.json)、[报告读审](results/communication_001/audit_report_review.json)
- [前轮封存完整性](prior_integrity_qa.json)、[历史索引原件](index_history/receipt.json)

沿用官方冻结DINOv2 ViT-L/14原60行特征与v0.20两帧私人接口。全部24个full私人终点均保留，v0.20的full–detach能力差验收失败不变。新发送和接收参数每人合计49,683；价值模块虽然重新初始化，但冻结且不调用。词表7、长度2，接收者仅见完整离散消息，分别回应两需求。没有新私人训练或DINO推理。

共同30总体比较包含60%熟悉布局。old18→old18的72种移动事件进入训练，added/sealed终态均未训练；评价初态仍为old18，5760行是终态平衡的权重表，不是5760个独立样本。三分区和两方向嵌套在四来源内。训练共57,600对主体更新、29,491,200消息、58,982,400次行动，约222.46秒，CPU单线程。

资源分项比总J揭示更多差异：局部条件的移动资源准确率高9.820个百分点，静止资源低12.899个百分点，四来源同向；该方向在v0.20私人行动中已出现。固定同一协议、只换评价观察，也能重现它，因此不能归因于社会训练才创造偏向。同终态不同路线的码一致率为79.469%/73.532%，与正确率共同解释，不能把码一致或条件互信息直接当组合结构。

执行审计检查全部世界/来源/外生随机流，按原批大小抽样重放缓存与评价前向，并独立重建第1/2101次更新；它没有重做全部训练。独立分析从480份协议原始数据重写公式，42,240项比较最大差约8.44e−15。绘图曾因缺包、NumPy路径优先级失败；保留失败记录，使用现有绘图依赖并保持训练NumPy1.26后完成，无生产代码或统计公式修改。

查看结果无需重跑。若复跑，输出目录必须尚不存在：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 redesign_v0.3/deployment/.venv/bin/python redesign_v0.21/run_social.py --out redesign_v0.21/results/communication_reproduction
```

统计和绘图同时运行时，应先导入训练环境的NumPy，再添加既有绘图依赖，避免旧绘图库目录内NumPy覆盖版本：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 redesign_v0.3/deployment/.venv/bin/python -c 'import numpy,sys,runpy; sys.path.insert(0,"redesign_v0.9/.analysis_deps"); sys.path.insert(0,"redesign_v0.21"); sys.argv=["analyze_social.py","--out","redesign_v0.21/results/communication_reproduction"]; runpy.run_path("redesign_v0.21/analyze_social.py",run_name="__main__")'
```

下一候选是无移动、完整初态后分别仅再次看见食物或水的配对探针，用来分开当前可见性和移动角色。该新事件支持尚未实施，需要先检查全部私人主体，保留分布外失败。当前仍缺独立视觉确认与足够清晰的论文中心贡献，整体研究目标继续。
