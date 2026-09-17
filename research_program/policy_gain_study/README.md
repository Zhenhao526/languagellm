# 策略学习权重对照

本轮检验策略损失的相对权重是否改变加性／联合收益的差异。沿用两个视觉主体资源定位任务，四个新种子、完整64组；不是三人Qwen实验。研究目标与语言形成的解释范围见[固定方案](plan.md)。

**已完成并独立核验。** [结果与下一步](结果与下一步.md)：三倍策略权重下，未见组合J在加性条件为4.16%→0.86%，联合为8.75%→7.38%；平均正交互来自两组降幅不同，早期U/N也未稳定改善。

- [执行核验](results/gain_001/audit_execution_001/独立核验.md)与[独立结果复算](results/gain_001/audit_results_002.md)。
- [完整数值](results/gain_001/analysis.json)、[形成图](results/gain_001/figures/formation.svg)、[终点图](results/gain_001/figures/endpoint.svg)及[绘图来源收据](results/gain_001/figures/receipt.json)。
- [一个固定情境的消息变化](results/gain_001/descriptive_example/example.md)；选择在8/64完成时确定，不是训练前登记的主要测量。

- [执行前兼容性验证](preflight_result.json)：四个微运行、八次更新；λ0/1的正常权重分支逐项复现旧代码。正式状态以 results/gain_001/status.json 为准。
- [设计审查](../policy_gain_design_review.md)：数学参照与替代解释；其可选建议由最终plan明确取舍。
- runner.py 为社会执行入口；analysis.py 为无模型结果分析；probe.py 只有 execute 子命令加载冻结小接口。
- 原v0.8与已有结果保留不变；本目录以独立输出保存正式运行、失败和后续复核。

以下为本次已执行的命令记录；输出已存在，不能原样再次执行来覆盖。新复现实验须在独立输出环境保留来源与运行记录。原analysis命令在margin精度核对处失败，显式恢复命令成功；两者均列出：

```sh
redesign_v0.3/deployment/.venv/bin/python research_program/policy_gain_study/runner.py --out research_program/policy_gain_study/results/gain_001
redesign_v0.3/deployment/.venv/bin/python research_program/policy_gain_study/probe.py prepare --batch research_program/policy_gain_study/results/gain_001 --out research_program/policy_gain_study/results/gain_001/probe
redesign_v0.3/deployment/.venv/bin/python research_program/policy_gain_study/probe.py execute --out research_program/policy_gain_study/results/gain_001/probe
redesign_v0.3/deployment/.venv/bin/python research_program/policy_gain_study/analysis.py --batch research_program/policy_gain_study/results/gain_001 --probe research_program/policy_gain_study/results/gain_001/probe --out research_program/policy_gain_study/results/gain_001/analysis
redesign_v0.3/deployment/.venv/bin/python research_program/policy_gain_study/diagnose_analysis_dtype.py
redesign_v0.3/deployment/.venv/bin/python research_program/policy_gain_study/analysis_float32_replay.py
research_program/.plotting_venv/bin/python research_program/policy_gain_study/plot_results.py
redesign_v0.3/deployment/.venv/bin/python research_program/policy_gain_study/extract_example.py
```

上列命令以项目根目录为工作目录。源码与数据来源SHA记录在正式manifest；有限预检不代替正式64组原始记录核验。四个种子是统计单位，照片和划分不增加独立样本数。

原失败与实现修正：[汇总dtype诊断](results/gain_001/analysis_dtype_diagnosis.md)、[显式恢复收据](results/gain_001/analysis_recovery_001/result.json)、[审计schema修正](results/gain_001/audit_schema_repair_plan.json)。训练、probe和原analysis的冻结文件保持不变，未重跑神经模型。

后续准备：[贪心覆盖的解释边界](贪心覆盖的解释边界.md)与[新视觉数据准备要求](视觉确认的数据准备要求.md)。其中连续概率分析与新图库确认尚未执行。
