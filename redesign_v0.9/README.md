# v0.9 新组合与共同约定的适应

问题：已在24张地图上形成约定后，将其余6种资源组合加入生活环境，发送端、接收端各自需要怎样的调整？

执行状态：36组正式适应实验全部完成，246,496项执行检查、0失败；协议枚举与实际轨迹交叉重放通过。四个继承种子、三个划分、三个学习范围，每组600×512世界。新增6图从适应开始已入训练，不能称零样本泛化。

- [完整研究报告](results/adaptation_001/新组合通信适应研究报告.md)
- [逐种子结果与三张科学图](results/adaptation_001/适应实验分析草稿.md)、[机器可读汇总](results/adaptation_001/adaptation_analysis.json)
- [协议结构](results/adaptation_001/协议结构报告.md)、[固定场景的实际消息](results/adaptation_001/消息实例.md)
- [正式执行审计](results/adaptation_001/audit_execution.json)、[独立协议重放核查](results/adaptation_001/protocol_execution_audit.json)

| 学习范围 | 新组合终点双目标成功 | 新组合过程AUC | 旧组合成绩变化 |
| --- | ---: | ---: | ---: |
| 仅发送端学习 | 38.70% | 33.57% | −0.09个百分点 |
| 仅接收端学习 | 10.79% | 10.14% | +0.15个百分点 |
| 双方学习 | 58.86% | 47.85% | +0.15个百分点 |

三组共同起点为7.19%。双方学习在四个种子的终点与过程上均优于单端；旧成绩宏平均基本保持。仅发送端学习充分利用已有接收码，双方学习还扩展接收覆盖。自然表达与人工片段拼接的变化并不相同，尚不能把新图学习称为一般组合语言。

- [固定执行方案](固定执行方案.md)与[独立适应审查](适应实验审查.md)
- [适应前接收可达性诊断](前置接收可达性诊断.md)、[独立口径审查](reachability_review.md)
- [适应前发送可区分性与保持旧成绩的上界](发送可区分性诊断.md)
- [预先固定的协议追踪](协议追踪方案.md)与[时间及文件哈希](protocol_fixed_manifest.json)
- [协议执行独立审查](protocol_execution_review.md)：旧供体含义、分母、原v0.8起点、冻结端及9600次配对重编码核对

本轮使用v0.8 mixed全部12个双主体检查点，保持冻结DINOv2-L视觉特征、个人投影、原伙伴、同一消息服务两种资源选择。只发送端学习、只接收端学习、双方学习均从相同状态、fresh Adam及相同世界开始；发送端包括视觉关系编码，接收端包括目标动作选择，不能视为纯语言模块。

复现使用原本地环境；完整重跑应选新目录，不覆盖已有结果：

```bash
redesign_v0.3/deployment/.venv/bin/python redesign_v0.9/run_adaptation.py --out redesign_v0.9/results/adaptation_001
redesign_v0.3/deployment/.venv/bin/python redesign_v0.9/analyze_adaptation_protocol.py
.venv/bin/python redesign_v0.9/analyze_adaptation.py
redesign_v0.3/deployment/.venv/bin/python redesign_v0.9/audit_adaptation.py
redesign_v0.3/deployment/.venv/bin/python redesign_v0.9/audit_protocol_link.py
redesign_v0.3/deployment/.venv/bin/python redesign_v0.9/write_message_examples.py
redesign_v0.3/deployment/.venv/bin/python redesign_v0.9/write_final_report.py
```

分析脚本使用本轮独立目录`.analysis_deps`中的NumPy和Matplotlib，不修改原训练环境；三张科学图均有PNG/PDF版本。重新计算协议报告会恢复生成器原表头，完整研究报告与独立审查保留了“全30图片段率、新图单独C_S、当前旧图供体消息”的明确解释。

design_review.md和protocol_review.md保留最初容量/通道候选及其未执行状态，实际执行以本页链接的适应方案为准。新实验不修改v0.8归档，不启动已暂停的v0.3/VPT路线；项目中Qwen路线资料另有索引。
