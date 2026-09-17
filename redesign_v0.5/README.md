# v0.5 从简单到复杂的视觉通信探索

状态：`results/progression_001`的54个运行已全部完成，依次完成A、B、C，三个独立种子及全部预定条件均保留。执行审计54/54通过；没有按结果替换种子。

- [阶段实验报告](results/progression_001/analysis/阶段实验报告.md)：方法、形成过程、全部条件结果和六张图。
- [原始协议与成分干预](results/progression_001/protocol_analysis.md)与[探索性重编码扩展](results/progression_001/protocol_analysis_extended_null.md)：区分任务成功、有限地图中的符号片段复用和未见地图组合。
- [固定当前情境的消息干预](results/progression_001/fixed_context_probe.md)：保持世界与接收者私人状态不变，检查消息对当前行动的作用。
- [完整执行审计](results/progression_001/audit_execution.json)：训练批次、预算、起点、冻结参数、课程／混排一致性及延迟回放等价性。

本轮隐藏目标任务中，课程训练平均成功率76.11%，直接训练89.27%，相同批次混排91.03%。持续三轮采集中，有通信83.79%，始终无通信39.51%，后者接近40.90%的精确无通信上界。混排条件中一个通信方向出现了可逐个替换的双符号资源位置表示；其他方向及留出组合仍不稳定。以上是旧照片上的三种子探索，不能称为普遍语法或论文级确认。

- [研究方向与渐进设计](研究方向与渐进实验设计_2026-09-15.md)
- [运行前固定的执行方案](渐进探索_固定执行方案.md)
- [独立可解性审查](design_review.md)
- [数据与能力复用审查](data_capability_review.md)
- [信息路由与动作校准](route_review.md)
- [结构评价补充](评价补充_整体消息重编码.md)
- [看到结果后的评价扩展说明](评价补充_整体消息重编码_扩展.md)

主任务A是侦察者看四地点资源照片、采集者按私人需求选地点；先逐步开放2、3、4个地点，再比较直接完整任务与相同批次混排。B加入观察与通信之间的短延迟，C加入三轮库存、个人采集历史与资源补充。B、C分别从各自指定的A检查点开始，C没有挑选B中成绩最好的模型。共54个运行、77,400次训练更新，三个独立种子。工具链、自由分工、代际传播尚未实现。

视觉骨干是本地冻结的官方DINOv2 ViT-L/14。旧照片只用于探索，不称新的视觉确认集。新通信接口不接收文本、未观察地图的资源类别元数据或伙伴隐藏状态；自身需求、公开地点坐标、库存和已经实际采得的资源数量是明确提供的非语言状态。接收者使用需求条件行动分支，这是给定的能力结构，结论不能外推为完全没有任务先验的认知系统。

## 复现

在工作区根目录执行。已有完整运行会跳过，部分写入的失败运行不会被覆盖。若要完整独立复跑，指定一个尚不存在的新输出目录。

```bash
redesign_v0.3/deployment/.venv/bin/python -m unittest discover -s redesign_v0.5 -p 'test_*.py' -v
redesign_v0.3/deployment/.venv/bin/python redesign_v0.5/bounds_audit.py
redesign_v0.3/deployment/.venv/bin/python redesign_v0.5/run_stages.py --out redesign_v0.5/results/progression_001 --stages ABC --seeds 24001 24002 24003 --updates-a 1800 --updates-b 900 --updates-c 1200 --batch 512 --eval-n 8192
redesign_v0.3/deployment/.venv/bin/python redesign_v0.5/audit_execution.py
redesign_v0.3/deployment/.venv/bin/python redesign_v0.5/analyze_protocols.py
redesign_v0.3/deployment/.venv/bin/python redesign_v0.5/analyze_protocols.py --extended-null
redesign_v0.3/deployment/.venv/bin/python redesign_v0.5/fixed_context_probe.py
.venv/bin/python redesign_v0.5/analyze_stages.py
```

训练用既有PyTorch环境，图表分析用已有绘图库环境。无需下载或重新部署大模型。运行记录包含源代码哈希、准备权重、阶段起点、配置、完整终点与检查点、逐步训练指标及每种评估模式的逐例轨迹。模型与原始数据目录中的旧实验均保留。改用新输出目录时，审计使用`--root 新目录`，协议分析使用位置参数`新目录`，固定情境干预与汇总分析使用`--input 新目录`。

开发用`smoke_001`至`smoke_006`、三套`receiver_control_99004*`阳性诊断不纳入54个主运行。诊断中的外部固定代码不进入社会学习，诊断权重也不回灌。source_snapshot/development_sources用于溯源，独立重跑应在原工作区布局中使用对应源码。
