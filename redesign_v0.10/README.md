# v0.10：新增经验与未训练组合迁移

已完成：4个新种子、12个18图来源、48个继续学习条件，共60个社会训练阶段；4组独立个人能力诊断另列。新增组合学习提高，始终未训练组合表达仍低。

- [完整研究报告](results/generalization_001/新组合经验与未训练组合迁移研究报告.md)
- [事前固定执行方案](固定执行方案.md)与[协议探针方案](协议探针方案.md)
- [功能结果](results/generalization_001/generalization_analysis.json)、[逐种子功能分析](results/generalization_001/泛化分析草稿.md)
- [协议结果](results/generalization_001/protocol_analysis.json)、[协议报告](results/generalization_001/协议探针报告.md)
- [完整执行审计](results/generalization_001/audit_execution.json)、[协议与实际轨迹审计](results/generalization_001/audit_protocol.json)
- [图表QA](results/generalization_001/figure_qa.json)、[独立探针审查](../paper_program/v10协议审查.md)
- [固定实例的300条实际消息与动作](results/generalization_001/消息实例.md)
- [结果后追加的旧用途约束诊断](results/generalization_001/compatibility_exploratory_002/兼容性诊断.md)与[独立穷举核验](results/generalization_001/compatibility_exploratory_002/independent_audit.json)：单独新增图解码上界92.49%，保持旧正确总数后15.06%；不能作为主体自然成绩或本批事前预测。

## 已执行矩阵

种子29101—29104 × 三个平衡划分。每个来源旧18图训练2,400×512，再克隆四臂，各600×512：expand_both、stay_old_both、expand_sender、expand_receiver。新增6图只在三扩展臂参与训练，另外6图在所有社会训练阶段均未参与。共57,600社会更新，29,491,200世界。三个划分是同构重复，不构成12个独立种子。

终点自然双目标成功率：扩展两端学习的新增图45.11%、始终未训练图4.71%，只练旧图控制的后者1.82%。主差+2.89个百分点，四种子为−0.68、+6.91、+2.76、+2.57；旧图平均约94%。新增图已学习，不能把45.11%叫零样本泛化。

## 复算

从项目根目录执行，使用已有隔离环境；以下分析和审计读取正式运行记录，不重新训练主体：

```sh
redesign_v0.3/deployment/.venv/bin/python redesign_v0.10/analyze_generalization.py
redesign_v0.3/deployment/.venv/bin/python redesign_v0.10/audit_execution.py
redesign_v0.3/deployment/.venv/bin/python redesign_v0.10/audit_protocol.py
redesign_v0.3/deployment/.venv/bin/python redesign_v0.10/audit_compatibility.py
redesign_v0.3/deployment/.venv/bin/python redesign_v0.10/export_examples.py
```

绘图依赖复用v0.9的隔离分析目录，不修改训练环境。首次训练命令为 `redesign_v0.3/deployment/.venv/bin/python redesign_v0.10/run_generalization.py`；首次端点探针为相同解释器运行 `redesign_v0.10/probe_protocol.py`。正式源及方案已按SHA归档，参见 `results/generalization_001/sources_81ffa3102043/`、`protocol_sources/` 和运行invocation JSON。探针用独占新目录避免覆盖，因此不能在已有 `protocol/` 时直接再运行；完整重现实验应在独立项目副本中使用相同输入与空输出目录，保留当前结果。

`s<seed>_p<partition>_<arm>/` 内有来源配置、初末权重和Adam、八个检查点、每步训练日志、五种终点模式NPZ。`protocol/` 内有120份方向NPZ，包含逐图照片ID、自然消息和完整发收logits。个人诊断权重在 `individual_controls/`，不转入社会学习。

## 限制与后续

本轮沿用旧照片、DINOv2-L特征和六地点规则，属于开发域。执行审计通过不等于证明语言起源、组合语法或论文新颖性。后续按[论文推进索引](../paper_program/README.md)检验具体机制与直接基线，再做独立视觉来源确认；所有后加分析需说明提出时间。

旧用途诊断脚本为 `analyze_compatibility.py`，使用独占输出目录，重新计算时显式指定新的 `--out`。正式阅读采用 `compatibility_exploratory_002`；001保留用于追溯措辞/记账完善，主要数值一致。下一轮[旧伙伴约定压力候选](../paper_program/兼容性机制候选方案.md)尚未冻结或执行。
