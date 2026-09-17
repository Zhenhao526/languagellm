# v24 私人预测的动态读取与共同符号结构

本轮比较Ri式mean/attention发送者的任务内改编。主体复用官方冻结DINOv2特征和v23私人all终点，两臂读取同样的私人位置概率＋角色标记；接收者只看到两枚离散符号。四继承来源×三分区×两臂，共24次新的双主体社会训练，每次2400更新。

- [完整研究报告](results/attention_001/私人预测动态读取与共同符号结构研究报告.md)
- [固定执行方案](固定执行方案.md)、[科学前审](前置科学审查.md)、[近邻方法核查](近邻基线方法核查.md)
- [独立分析](results/attention_001/analysis.json)、[执行审计](results/attention_001/audit_execution.json)、[训练凭证](results/attention_001/training_complete.json)
- [重组方法](符号重组评估方法.md)、[完整重组结果](results/attention_001/结构重组评估.md)、[正式重组核查](results/attention_001/structural_assay_audit.json)
- [正式前门禁](preflight_qa.json)、[有限代码读审](code_review.json)、[分析源码版本说明](source_revision_binding.json)
- [结果解释地图](结果解释与后续判别.md)、[封存清单](results/attention_001/completion_manifest.json)

结果分析采用来源层级的配对均值。主差为社会未见new12上的逐token贪心联合成功率attention−mean。Q、旧组合表现、曲线AUC与人工符号重组分别报告。两个条件都已在私人学习见过全部30布局；new12不是主体全新世界。只保留多个输入并动态读取相对均值压缩的整体结构对比，不能归于一个孤立的注意能力；同参数量不等于同有效容量。

## 本地复现

既有[信息流自检记录](self_test_qa.json)供只读核查，默认复现不重写该证据。

从项目根目录执行，使用既有Python环境。原始封存目录不能覆盖，新复现输出另命名。

```bash
redesign_v0.3/deployment/.venv/bin/python redesign_v0.24/run_attention.py --out redesign_v0.24/results/reproduction_001
redesign_v0.3/deployment/.venv/bin/python redesign_v0.24/analyze_attention.py --out redesign_v0.24/results/reproduction_001
redesign_v0.3/deployment/.venv/bin/python redesign_v0.24/structural_assay.py --out redesign_v0.24/results/reproduction_001 --step 2400
redesign_v0.3/deployment/.venv/bin/python redesign_v0.24/plot_results.py --out redesign_v0.24/results/reproduction_001
```

依赖v23已封存prepared、private_all终点、h缓存和世界表，以及其所依赖的既有图片特征。run_attention的正式门禁核对本轮固定源hash。新输出会记录其自身调用和文件hash；浮点轨迹依赖Torch2.14.0、NumPy1.26.4、CPU单线程等环境，跨环境不承诺逐位一致。绘图使用v0.9已有matplotlib目录，先导入当前虚拟环境NumPy再加入绘图依赖。报告与示例脚本当前绑定本次attention_001，另一次复现的图可由--out直接产生。

审计不重训所有社会更新：完整重算终点、全部评估统计与世界抽样流，只在固定来源/分区重放第1与2101更新的参数和Adam。训练缓存只复核首尾原始块；测试缓存全量复核。封存输出的hash用于确定本批材料身份，不表示四个来源以外还有独立重复。

本轮无新DINO推理、无私人训练、无新图片。旧图像池已多轮开发；独立材料确认、超出近邻的中心贡献和统一论文稿件仍待完成。
