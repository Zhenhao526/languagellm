# 三人约定形成轨迹

本轮追踪四个已有配对初始化、16项三人政策的六个训练检查点，区分发送响应、单位置功能及跨时间两轮消息兼容性。主计算和独立复核均完成；没有新增训练或Qwen调用。

- [结果与下一步](结果与下一步.md)：四个配对主要量、所有时点、反向结果和解释边界。
- [冻结设计](plan.md)：观测时间网格、发现/验证分离及唯一主要比较。
- [独立复核002](results/trajectory_001/audit_execution_002/verification.json)、[001失败与修复记录](results/trajectory_001/audit_source_snapshot_002/manifest.json)。
- [正式准备与冻结](results/trajectory_001/plan.json)、[主计算结果](results/trajectory_001/execution/results.json)。
- [全量描述汇总](results/trajectory_001/descriptive_summary.json)、[绘图收据](results/trajectory_001/figures_001/figures.json)、[实际PNG检查](results/trajectory_001/figures_001/visual_review_001.json)。
- [文献近邻](literature/形成过程研究的近邻与边界.md)、[下载及复用记录](literature/download_index.json)。

`dataset.py`只准备静态索引和保存权重清单；`channel.py`执行两轮通信，`runner.py`负责完整固定批次；`metrics.py`计算时间曲线与兼容矩阵。正式通过的独立实现位于`audit_execution_v2.py`，初版与失败记录保留在`audit_execution.py`及`audit_execution_001`，`summarize_results.py`与`plot_results.py`只读取完成后的JSON，不调用模型。

正式主运行目录为`results/trajectory_001`；已存在目录拒绝覆盖。6000步直接复用此前单位置实验的已核输出，不能把本轮当作新社会或独立确认。跨时间矩阵的行是接收者时点，列是供体消息时点；这些时点不是世代。

使用仓库根目录的既有Python环境，可只读核验冻结文件：

```sh
qwen_collect_pilot/.venv/bin/python -m research_program.triadic_formation_trajectory_study.runner verify --out research_program/triadic_formation_trajectory_study/results/trajectory_001
```

真正重做主批须选新的输出目录，先`prepare`再`execute`；参数与预算须由新计划绑定。完整一次新增203,853,600模块样本、88次参数加载、0训练更新，独立重放另计同量计算；文件复制和静默别名均计0前向。不要覆盖本轮记录。
