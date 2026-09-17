# v0.32 联合收益压力下的共同符号形成

本轮把伙伴轮换与同回合联合收益结合起来。四个主体使用两种冻结私有视觉接口，发送两个离散 token；一对伙伴在同一批世界上双向通信，四个资源动作全部正确时才获得额外联合奖励。正式矩阵为4个初始化来源 × 3个坐标面板 × 固定/轮换两种伙伴制度，共24组、每组2400个群体更新。

主要结果是一个可检验的分离：轮换伙伴使同私有类型主体的完整消息一致率从1.898%提高到49.815%，但目标12终点 J 只有1.953%到2.054%，四个来源的目标差异正负各半；要求同一资源的伙伴对同时正确时，均值仍为0.116%到0.159%。这支持“形式协议趋同不等于 grounded 语义传递”的有限机制结论，不支持“已经产生语言”。

- [固定执行方案](<./固定执行方案.md>)、[支持图设计](<./support_design.json>)、[实现审查](<./implementation_review.json>)、[正式前置记录](<./preflight_qa.json>)
- [完整研究报告](<./results/joint_002/联合收益压力下的共同符号形成研究报告.md>)、[独立统计](<./results/joint_002/analysis.json>)、[原始统计复核](<./results/joint_002/raw_validation.json>)
- [执行审计](<./results/joint_002/audit_execution.json>)、[结果审查](<./结果审查.md>)、[结果审查 JSON](<./结果审查.json>)、[完成封存](<./results/joint_002/completion_manifest.json>)
- [设计图](<./results/joint_002/figures/01_partner_design.png>)、[结果图](<./results/joint_002/figures/02_partner_outcomes.png>)、[实际消息样本](<./results/joint_002/实际消息示例.md>)
- [复现入口](<./run_support.py>)、[分析入口](<./analyze_results.py>)、[报告入口](<./build_report.py>)、[打包入口](<./package_results.py>)

下一轮按一个因素递增：让同一接收者的一个行动依赖两个不可互换发送者的互补信息，并加入单发送者与随机消息对照；如果仍只有形式一致率上升，再增加延迟第三方转发。
