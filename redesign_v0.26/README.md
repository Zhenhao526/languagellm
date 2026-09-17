# v0.26：冻结私人主体的新图片能力探针

已完成。11张新流程图片上，私人all的目标12布局双目标行动仍为100%；私人old为69.879%。四格材料替换、48冻结终点、224640个私人世界，全部数据保留。没有训练新协议，也未使用原确认48图。

- [研究报告](results/material_001/冻结私人主体的新图片能力研究报告.md)
- [固定方案](固定执行方案.md)、[生产程序](run_probe.py)、[独立核查程序](audit_probe.py)
- [结果汇总](results/material_001/summary.json)、[独立核查](results/material_001/audit_qa.json)
- [输入冻结](results/material_001/freeze.json)、[完成记录](results/material_001/completion.json)
- [11图署名和来源](results/material_001/selection.json)、[图片模型暴露记录](results/material_001/model_exposure_started.json)

复现命令见报告。必须使用新的输出目录；原v23权重和v4缓存仅作只读输入，新图沿用旧训练特征的标准化，不重估。现有11图仅为有限流程材料，已暴露于模型，不得作为后续未揭盲确认集。下一步是另立冻结协议材料稳健性检查，独立形成复现仍需新材料训练/测试划分。
