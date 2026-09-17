# v25 资源角色分别读入与共同通信学习

从v24 mean完全相同的初始发信函数开始，将其无任务梯度的bilinear矩阵换为同数参数、零初始化的资源对比读出。两个资源的融合权重初始相同，此后可分别学习。新增joint12次双主体训练，旧mean/attention24次训练仅作已封存参考。

- [完整报告](results/joint_001/资源角色分别读入与共同通信学习研究报告.md)
- [固定方案](固定执行方案.md)、[科学前审](前置科学审查.md)、[有限代码读审](code_review.json)
- [函数关系](联合读入的函数关系.md)、[初态与信息流自检](self_test_qa.json)、[正式前门禁](preflight_qa.json)
- [独立分析](results/joint_001/analysis.json)、[执行审计](results/joint_001/audit_execution.json)、[指标比较](results/joint_001/comparison.json)
- [结构评估方案](结构评估方案.md)、[结构结果](results/joint_001/结构重组评估.md)、[有界朴素核查](results/joint_001/structural_assay_audit.json)
- [过程描述](results/joint_001/process_summary.json)、[训练凭证](results/joint_001/training_complete.json)、[封存清单](results/joint_001/completion_manifest.json)

本轮主量为old18留出测试照片上的联合成功率joint−mean，针对v24熟悉任务学习不足进行诊断；new12、Q、AUC、两遮挡及人工供体重组是辅助，未改v24原主量。四个继承来源是外层单位，方向/分区/照片行不增加独立重复。两组初始函数相同并不意味着有效容量、梯度或后续更新相同。

## 复现

从项目根目录运行。沿用既有Python环境、v23准备结构与v24完整封存输出，原目录不覆盖，另立输出目录：

```bash
redesign_v0.3/deployment/.venv/bin/python redesign_v0.25/run_joint.py --out redesign_v0.25/results/reproduction_001
redesign_v0.3/deployment/.venv/bin/python redesign_v0.25/analyze_joint.py --out redesign_v0.25/results/reproduction_001
redesign_v0.3/deployment/.venv/bin/python redesign_v0.25/analyze_structure.py --out redesign_v0.25/results/reproduction_001 --step 2400
redesign_v0.3/deployment/.venv/bin/python redesign_v0.25/plot_results.py --out redesign_v0.25/results/reproduction_001
```

模型源码按固定门禁hash验证，环境为Torch2.14.0、NumPy1.26.4、CPU单线程；不同运行库不承诺逐位相同。绘图先加载虚拟环境NumPy再加入v0.9现成matplotlib。自检、报告及过程描述脚本绑定本轮证据目录，不列为默认重写命令。

本轮没有新DINO推理、私人训练或私人缓存前向；直接复用v24私人概率缓存并核查字节。独立审计重算全部新终点和初态消息分布、全部新评价指标、全部外部随机流；只在固定33101/p1重放第1与2101成对参数/Adam更新。旧基线指标复用v24已审结果，不能把它们算作本轮新训练或重新独立确认。

该任务已给定需求角色、地点与模型预测分支，两枚符号的含义并未设定。旧图库多轮开发、独立材料不足及中心贡献问题仍在，不能用熟悉任务高分代替语言结构或论文完成。
