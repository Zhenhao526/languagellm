# v0.8 资源互补收益与共同消息

状态：60组社会实验、4组个人诊断全部完成。11项独立测试通过，正式执行审计1,175,469项检查、0失败；八人个人诊断正常观察均9600/9600、清空视觉表示均1600/9600，诊断权重未回灌。

- [完整实验报告与四张科学图](results/complementarity_001/analysis/资源互补收益实验报告.md)
- [完整协议与符号干预](results/complementarity_001/protocol_analysis.md)
- [固定场景中的实际消息](results/complementarity_001/消息实例.md)
- [执行审计](results/complementarity_001/audit_execution.json)与[独立汇总](results/complementarity_001/analysis/summary.json)
- [重编码随机序列来源核查](results/complementarity_001/reference_rng_audit.md)：120方向对应90个参考随机种子，主留出72方向彼此无碰撞；12,000次是条件化参考评分，不是独立主体样本。

下表先在每种子内平均三个划分和两个方向，再等权平均四种子；双目标成功要求同一条自然消息支持食物、水两次选择均正确。

| 收益条件 | 训练24图双目标 | 留出6图双目标 | 留出平均单目标 |
| --- | ---: | ---: | ---: |
| λ=0 加和收益 | 76.67% | 5.91% | 48.82% |
| λ=0.5 混合收益 | 92.80% | 7.13% | 50.97% |
| λ=1 严格联合收益 | 93.65% | 11.26% | 53.90% |

预定joint−additive比较：训练图双目标提高16.98个百分点，四种子均提高；留出图提高5.36个百分点，四种子中三个提高，一个降低0.21个百分点。互补收益对熟悉图的完整通信作用更明显，未见组合仍远未可靠。三种条件都奖励完整成功，联合收益并非本任务通信出现的必要条件；其作用也包含反馈密度和优化信号变化，不能直接归为认知能力或语法的出现。四新种子仍使用旧照片，属于机制探索。

结构干预没有随λ单调提高：加和/混合/联合三组严格片段替换为31.75%/48.22%/40.42%，训练供体人工拼接为21.96%/35.46%/19.30%。中等互补收益在这两个指标上最高，说明自然双目标完成度与片段可复用性不能视作同一结果。协议照片枚举与上表终点抽样不同，详情保留各自分母及整码重编码参照。

问题：同一条消息支持两种资源选择时，提高兼得资源的相对收益，是否促进自然消息表达未见组合？固定v0.7 identity视觉接口、两次无相互反馈的选择，仅改变回报互补强度λ=0/.5/1。四新种子27101–27104、60组社会运行与4组个人能力诊断，所有失败及种子差异保留。

- [训练前固定方案](固定执行方案.md)
- [独立设计审查](design_review.md)
- [协议分析口径](protocol_review.md)
- [文献先例与解释边界](分析口径与解释边界.md)

在项目根目录使用现有本地环境；独立重跑需新结果目录。

```bash
redesign_v0.3/deployment/.venv/bin/python -m unittest discover -s redesign_v0.8 -p 'test_*.py' -v
redesign_v0.3/deployment/.venv/bin/python redesign_v0.8/run_controls.py --out redesign_v0.8/results/complementarity_001
redesign_v0.3/deployment/.venv/bin/python redesign_v0.8/run_experiment.py --out redesign_v0.8/results/complementarity_001
redesign_v0.3/deployment/.venv/bin/python redesign_v0.8/analyze_protocols.py
redesign_v0.3/deployment/.venv/bin/python redesign_v0.8/audit_execution.py
.venv/bin/python redesign_v0.8/audit_reference_rng.py
.venv/bin/python redesign_v0.8/analyze_complementarity.py
```

三组原生收益定义不同，统一比较双目标和单目标准确率。该操纵同时改变效用与学习信号，不等同删除认知能力；同一采集者的两次需求选择不等于两名独立听者或自发分工。原始DINO照片仍为旧44/16开发划分，个人诊断权重不回灌社会实验。
