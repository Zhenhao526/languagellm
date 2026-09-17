# v0.12：固定发送者接收诊断

本批检验既有消息保留的信息、接收目标与学习程序。仍使用官方DINOv2 ViT-L/14冻结特征及独立接口；只训练接收模块，原发送者全部冻结。监督探针得到地点标签，不能称为自然社会学习或新增语言形成。

**96个正式拟合及分析已完成。** [完整研究报告](results/receiver_001/固定发送者的接收诊断研究报告.md)显示：共同24图的原生发送联合最优为73.26%，奖励接收者达到72.04%；混合收益参照0.7928、实际0.7841。监督把NLL从3.8174降至0.6516，但随机接收Q下降4.85个百分点，四个来源种子均下降；继承与新初始化的监督接收器表现接近。单独组最优不能拼成一张共同码表，完整30图的联合最优仅60.54%。这些是对已有协议的诊断，尚未产生独立的语言形成机制贡献。

- [完整分析JSON](results/receiver_001/receiver_analysis.json)、[曲线与三类目标参照](results/receiver_001/接收诊断分析草稿.md)
- [执行审计](results/receiver_001/audit_execution.json)：1,444,323项检查，0失败。
- [独立分析审计](results/receiver_001/audit_analysis_independent.json)：728,963项检查，0失败；[纯展示版本绑定](results/receiver_001/audit_display_revision.json)确认数字未变。
- [根线程独立逐世界汇总](results/receiver_001/independent_summary.json)与[9,696值核对](results/receiver_001/independent_summary_comparison.json)
- [最终科学读审](results/receiver_001/scientific_review.json)、[交付核查](results/receiver_001/delivery_qa.json)、[本批完成清单](results/receiver_001/completion_manifest.json)
- [贡献边界与下一步候选](../paper_program/v12贡献边界与下一步.md)：非语言空间重组能力的比较尚未冻结或执行。

沿用v0.10全部12个形成终点的双向发送函数，比较奖励学习、继承接收者的监督学习、新接收者的监督学习和全30地图监督探针。预定96个单向拟合，每项2400次更新。独立来源仅4个主体对，三划分与两方向嵌套其内。训练前固定照片fit/dev/evaluation划分、随机流、预算与评价规则。

- [既定实施规格](../paper_program/receiver_baseline/固定发送者接收诊断_实施方案.md)与[本批执行细则](固定执行方案.md)
- [数学测量与解释边界](测量方案.md)、[独立前置审查](前置审查.md)
- [运行器](run_receiver.py)、[NumPy解析评价](receiver_metrics.py)、[独立审计](audit_receiver.py)

正式结果已在96个拟合全部完成后统一分析。没有用单组高分替代全批结果，或用评价照片与封存组合选择检查点。2400步为主终点，600步为短预算；监督条件最低dev支持NLL的检查点仅作补充。三种不同目标分别使用分支条件熵、联合MAP与混合收益最优参照。

运行环境为项目既有 `redesign_v0.3/deployment/.venv/bin/python`。正式运行入口为 `redesign_v0.12/run_receiver.py`，预检通过后才接受固定矩阵；默认输出新建 `results/receiver_001`，不会覆盖旧批次。重现需另建输出目录或完整项目副本。

以下重算命令请在独立项目副本运行；分析器会写回该副本的汇总和图表：

```sh
redesign_v0.3/deployment/.venv/bin/python redesign_v0.12/analyze_receiver.py
redesign_v0.3/deployment/.venv/bin/python redesign_v0.12/independent_receiver_summary.py
```

正式训练日志为`formal_run.log`，训练与独立汇总进程均已exit0；独立审计命令、源版本及图标签修订保存在各审计receipt和`analysis_versions`。原始训练、预检源、检查点、实际消息与损失记录全部保留，不覆盖旧版本重跑。

smoke_001保留开发记录；解析模块补充极端非有限数值防护后，以smoke_002核验最终来源。两批各使用99510 p1方向0、四条件各4步，不纳入正式科学结果。
