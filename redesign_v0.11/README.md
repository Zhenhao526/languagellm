# v0.11：旧伙伴约定压力与中点解除

本轮在DINOv2-L冻结视觉特征、私人投影、两位七符号通信及六地点资源任务上，比较旧伙伴版本对约定扩展的影响。

36个正式运行、功能分析、预定协议探针及独立审计均已完成。[完整研究报告](results/anchoring_001/旧伙伴约定压力与新表达研究报告.md)记录全部结果：解除相对持续锚定的新增图后半程平均成功率提高1.261个百分点，四个来源种子为+0.804、+1.852、+2.390、−0.003个百分点；终点则两正两负。三臂新增图终点58%—60%，始终未训练图约4%—5%。没有支持旧约定是共同学习主要瓶颈的强解释，也未证明效应为零。

- [功能汇总与四种子差值](results/anchoring_001/anchoring_analysis.json)、[协议诊断](results/anchoring_001/协议诊断.md)
- [执行审计](results/anchoring_001/audit_execution.json)：1,830,923项检查，0失败，封存图训练暴露0。
- [独立查询重放](results/anchoring_001/audit_query_replay.json)：82,944次保存的训练通信及2,419,200个终点世界全部吻合。
- [独立协议审计](results/anchoring_001/audit_protocol_independent.json)：12,142项检查，0失败。
- [最终科学读审](results/anchoring_001/scientific_review.json)、[交付核查](results/anchoring_001/delivery_qa.json)、[本批完成清单](results/anchoring_001/completion_manifest.json)
- [下一步接收诊断实施方案](../paper_program/receiver_baseline/固定发送者接收诊断_实施方案.md)：尚未执行。

正式矩阵：v0.10全部12个18图形成终点，各克隆current、anchor、release三臂，600更新；release前300次与anchor相同，第301次开始改用当前伙伴。两端都可学习，每角色各128旧+128新增样本；每步512上下文、768通信、1536资源动作。独立单位为4个继承来源种子，三划分与两方向嵌套其内。

- [固定执行方案](固定执行方案.md)与[事前测量细则](测量与解释边界.md)
- [运行器](run_anchoring.py)、[前置独立审查](前置独立审查.md)、[2,886项预检](preflight_qa.json)
- [独立查询重放器](audit_query_replay.py)：单独重建伙伴路由、逐位发信与政策抽样，不导入训练器查询函数。
- [预定协议测量](协议测量方案.md)、[探针](probe_anchoring.py)、[冻结哈希](protocol_fixed_manifest.json)
- [功能分析器](analyze_anchoring.py)、[执行审计器](audit_anchoring.py)

## 执行记录

smoke_001为三个4步开发运行；补充首次更新权重/Adam记录及旧副本无梯度断言后，在smoke_002重复验证，未改变正式训练设计。两套开发记录均保留。smoke使用99510的历史开发来源，仅用于流程验证。正式前把探针与QA归档于protocol_sources；正式训练源副本位于results/anchoring_001/frozen_sources。

正式输出位于 `results/anchoring_001`；进度日志 `formal_run.log`，全36训练结束显示COMPLETE。任何成绩汇总必须等待全批固定预算完成。主比较为release−anchor在300—600步added自然双成功的梯形面积/300；两种旧跨伙伴互通、当前旧任务与sealed表达各自报告。

## 重算入口

从项目根目录使用已有训练环境：

```sh
redesign_v0.3/deployment/.venv/bin/python redesign_v0.11/analyze_anchoring.py
redesign_v0.3/deployment/.venv/bin/python redesign_v0.11/audit_anchoring.py
redesign_v0.3/deployment/.venv/bin/python redesign_v0.11/audit_query_replay.py
```

完整训练首次命令为相同解释器运行 `redesign_v0.11/run_anchoring.py`；首次协议测量运行 `redesign_v0.11/probe_anchoring.py`。两者均要求新的输出目录，防止覆盖。重现实验需独立项目副本或显式新输出路径，不能删改当前冻结源和结果以重跑。

本轮不是新视觉确认，不能把旧伙伴副本解释为新人的学习或代际传承。伙伴版本同时改变非平稳性与奖励反馈，干预主效应不直接识别某个数学兼容界的因果作用。独立视觉元数据准备另见[抽样框](../paper_program/visual_confirmation_v1/README.md)，正式确认图片仍未就绪。
