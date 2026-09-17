# v0.16 消息策略与价值输入幅度

本轮把v0.15的整体幅度补偿拆成两条计算支路。新增policy_only_scale和value_only_scale共24次社会学习，复用v0.14 reset（00）与v0.15 reset_scaled（11）各12次，组成2×2。正式批次已完成后汇总：00/10/01/11的未训练sealed自然双成功为4.44%/6.90%/7.97%/13.63%。唯一主要差10−01为−1.08个百分点，四来源未显示一致的单支路优势。辅助11−10为+6.73个百分点，四来源均正；交互+3.20个百分点、两正两负。

唯一主要对比是sealed终点自然N的“仅策略缩放−仅价值缩放”。四来源内先平均三分区、两个方向，n=4。四简单效应和交互、old/added及全程AUC为辅助；不按新成绩改主要比较。旧图片和继承来源不构成独立确认。

- [完整研究报告](results/branches_001/策略与价值输入幅度的通信形成研究报告.md)、[结构化分析](results/branches_001/branches_analysis.json)、[报告数值](results/branches_001/report_numbers.json)
- [正式执行审计](results/branches_001/audit_execution.json)：975,860项，0失败；[独立原始复算](results/branches_001/independent_recount.json)46,848个标量，[分析比较](results/branches_001/independent_recount_comparison.json)71,936项通过。
- [事后裁剪诊断](results/branches_001/clip_activity_exploratory.json)、[独立核验](results/branches_001/clip_activity_independent_qa.json)：460,800个人×角色更新均无非单位裁剪，890检查通过；[机制解释与候选](裁剪核查后的机制解释与候选.md)中的发送基线置换尚未训练。
- [图表检查](results/branches_001/figure_qa.json)、[展示修订](results/branches_001/display_revision.json)：补齐轴刻度，366,147个数值不变。
- [固定执行方案](固定执行方案.md)、[前置科学审查](前置科学审查.md)、[执行审查](前置审查.md)
- [运行器](run_branches.py)、[两个输入分支的固定倍率](branch_interface.py)
- [成功预检](preflight_qa.json)、[源文件绑定](source_receipt.json)
- [两步开发端点重放](results/smoke_001/root_endpoint_replay.json)、[独立分析开发检查](results/smoke_001/analysis_smoke.json)
- [文献与贡献边界](文献与贡献判断.md)：计算路径归因本身不足以声称语言起源或通用机制创新。
- [当前图片流程](../paper_program/visual_confirmation_v1/pixel_workflow_003/README.md)：累计11/24张有限流程可用，3项待核未计通过；与模型实验分开，尚无独立新图模型确认。

每人复用v0.15在old训练支持校准的同一alpha，不重新估计。原raw observe不变，策略和价值的第一层前各只缩放96维h，另4维情境保持原值。两个persistent buffers明确冻结，旧可训练参数名和对象保持。协议同时记录raw_h/policy_h/value_h，h字段明确为raw；不将旧v0.15的effective h与新raw h混为同层向量。

正式命令（已完成；查看结果不需要再次运行，复跑须指定不存在的输出目录）：

```sh
redesign_v0.3/deployment/.venv/bin/python redesign_v0.16/run_branches.py --out redesign_v0.16/results/branches_001
```

每新运行2400×512通信，合计29,491,200条通信、58,982,400次资源行动；没有新私人训练、参考重跑、视觉主干微调或云端训练。固定DINOv2 ViT-L/14缓存、资源任务、回报、信道及外生随机流；两新条件共享所有预算与非干预参数初值。

主要差的方向在任何开发训练前经审查统一；最早未执行草稿另存[讨论历史](design_history/固定执行方案_未训练草稿_interaction.md)，不是看到新成绩后更换指标。任何后续候选另立方案，不回写本批固定文件。
