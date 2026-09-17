# v0.30 多伙伴共同目标实验

本轮将共同测试从一一匹配改为每个资源位置拥有两个伙伴，比较两种等规模、等信息量但支持路径分布不同的训练图。24组正式通信训练已经完成，目标12上的结果见[完整研究报告](<./results/support_001/多伙伴共同目标与支持结构研究报告.md>)。

- [固定执行方案](<./固定执行方案.md>)、[支持图设计](<./support_design.json>)、[正式前置记录](<./preflight_qa.json>)
- [独立统计](<./results/support_001/analysis.json>)、[原始统计复核](<./results/support_001/raw_validation.json>)、[执行审计](<./results/support_001/audit_execution.json>)
- [结果审查](<./结果审查.md>)、[结果审查 JSON](<./结果审查.json>)、[复现入口](<./run_support.py>)

四个初始化来源、三个坐标置换和两种支持臂均保留；没有按成绩筛选。正式批次不重新训练视觉主干或私人能力，输入路径和源文件由哈希绑定。
