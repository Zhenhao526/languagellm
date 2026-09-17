# v0.15 单一全局幅度校准

当前已固定一个新条件reset_scaled。它只在v0.14重置编码的observe输出乘每人一个常量，检验全局幅度差是否足以解释先前私人准备的迁移收益。训练支持校准、两项开发流程、12次正式社会学习及独立审计全部完成。sealed自然N：保留22.04%、重置4.44%、幅度补偿13.63%；主要剩余差+8.41个百分点，辅助修复差+9.18个百分点。新结果说明幅度可恢复部分成绩，不能据此识别特定知识机制。

- [完整研究报告](results/scaled_001/视觉编码幅度与共同符号形成研究报告.md)与[分析数据](results/scaled_001/scaled_analysis.json)
- [执行审计](results/scaled_001/audit_execution.json)：462,420检查通过；[独立原始复算](results/scaled_001/independent_recount.json)35,136标量；[分析比较](results/scaled_001/independent_recount_comparison.json)46,973项通过。
- [图表](results/scaled_001/figure_qa.json)、[展示修订留档](results/scaled_001/display_revision.json)、[下一候选审查](策略与价值缩放_候选审查.md)（未训练）。
- [固定方案](固定执行方案.md)、[科学审查](前置科学审查.md)、[执行审查](前置审查.md)
- [缩放与校准模块](scaled_interface.py)、[运行器](run_scaled.py)
- [正式24人校准](calibration_001/calibration_complete.json)：每人old18×22×22训练照片场景，倍率约4.98–5.68；test/未训练地图不参与。
- [源文件绑定](source_receipt.json)：原v13保留和v14重置参考已见过，不计新独立样本。
- [倍率1的开发重放](results/smoke_001/root_unit_replay.json)：原参数/Adam、日志和行为逐位一致；新增buffer导致state哈希与旧版不同是预期。

正式命令记录（输出目录必须不存在；必须先通过preflight_qa.json）：

```sh
redesign_v0.3/deployment/.venv/bin/python redesign_v0.15/run_scaled.py --out redesign_v0.15/results/scaled_001
```

本轮新运行仅12次reset_scaled社会训练，复用12个retained和12个reset参考。每组2400×512通信。主比较为retained−reset_scaled的sealed自然双目标终点剩余差；修复差reset_scaled−reset为辅助。四来源内三分区、两方向平均，n=4。范数校正不等于能力匹配、完整几何匹配或移除知识；沿原7×2离散消息和冻结DINO路线。

复算入口（完整运行已经完成，无需为查看结果重新训练）：

```sh
redesign_v0.3/deployment/.venv/bin/python redesign_v0.15/independent_recount.py redesign_v0.15/results/scaled_001
redesign_v0.3/deployment/.venv/bin/python redesign_v0.15/report_numbers.py
```

全部新运行均为本地CPU使用缓存特征，未重新推理或微调DINO主干。保留v13/v14参考及原图片，四来源仍是开发证据；下一候选和图片流程验收不计作已完成独立模型确认。

[本轮图片流程验收](../paper_program/visual_confirmation_v1/pixel_workflow_002/README.md)净增2张，两批累计7/24，未进入模型实验。正式报告[独立读审](results/scaled_001/audit_report_review.json)通过51项；完成清单 `results/scaled_001/completion_manifest.json`绑定本批源码、校准、检查点、统计、报告及失败记录。
