# v0.13 私人空间训练与通信形成

正式批次 `results/spatial_001` 的24次私人主体对训练和24次社会学习、完整分析与独立执行核查均已完成。主问题是通信前的非语言空间行动训练能否改变新共同符号的形成及未训练组合表达。两臂私人未训练双目标能力约95%，社会熟悉组合约97%；主要sealed自然成功率为22.04%与19.87%，四种子2正2负，未支持额外空间约束的稳定收益。私人对照已强，能力操纵幅度很小。

- [完整研究报告](results/spatial_001/私人空间能力与共同符号形成研究报告.md)
- [分析草稿和三张科研图](results/spatial_001/空间行动准备与通信形成_分析草稿.md)、[结构化统计](results/spatial_001/spatial_analysis.json)、[固定消息形成样例](results/spatial_001/固定消息形成样例.md)
- [完整执行核查](results/spatial_001/audit_execution.json)及[绑定的独立源码](results/spatial_001/audit_execution_source.py)：1,612,171项通过，0失败；模型执行与数学诊断通过不等于新颖性成立。
- [独立NumPy重算](results/spatial_001/independent_recount.json)及[与分析器逐项比较](results/spatial_001/independent_recount_comparison.json)：前者23,424值，后者24,997值吻合。

- [固定执行方案](固定执行方案.md)：4个新来源种子×3分区×2条件，24次私人主体对训练后接24次社会训练，均2400更新。
- [前置科学审查](前置审查.md)：已知地点变换对应属于额外非语言监督，私人行动头的等变性不等于潜表示等变或语言能力。
- [正式运行器](run_spatial.py)及[私人训练模块](private_preparation.py)
- [独立预检](preflight_qa.json)：1861项通过，全部源码SHA256绑定；[随机流核查](rng_preflight.json)覆盖288096独立身份。
- [开发smoke002](results/smoke_002/training_complete.json)：使用正式批量和评价规模的两步完整流程。smoke001保留了已修复的私人/社会随机数流碰撞问题。
- [首步浮点复算差异说明](results/smoke_002/audit_roundoff_receipt.json)：独立计算图重排曾产生末位梯度差，按相同图创建次序复算后参数与Adam逐位相同，没有修改训练或放宽门槛。
- [只读完整批次分析](analyze_spatial.py)、[独立NumPy重算](independent_recount.py)、[固定过程样例规范](展示样例预设.md)
- [六篇近邻全文方法与一篇摘要核查](../paper_program/literature_spatial_20260915/近邻方法与贡献边界.md)：不宣称首次研究非语言先验或冻结DINO通信。

正式复现命令（必须使用未存在的新输出目录，以免覆盖历史结果）：

```sh
redesign_v0.3/deployment/.venv/bin/python redesign_v0.13/run_spatial.py --out redesign_v0.13/results/spatial_001
```

运行器要求源码与预检SHA256一致。实际批次已经创建；上述命令是记录，不能直接重跑覆盖。

在已完成批次上复核：

```sh
redesign_v0.3/deployment/.venv/bin/python redesign_v0.13/analyze_spatial.py
redesign_v0.3/deployment/.venv/bin/python redesign_v0.13/independent_recount.py redesign_v0.13/results/spatial_001
redesign_v0.3/deployment/.venv/bin/python redesign_v0.13/audit_v13.py
```

历史分析初版及仅显示文字变化的修订链由结果目录保留；原始训练与测量数据不修改。报告没有把同支持oracle、随机策略期望和自然贪心成功率混为一谈。现有v0.13与旧版本的得分差并非因果对照；下一轮优先补同来源的私人视觉记忆重置对照，方案仍待另行固定。

使用官方DINOv2 ViT-L/14的既有冻结特征（304,368,640主干参数）；主体间只传两个离散符号。训练的仍是各自小型行动/通信接口，没有新训练视觉主干，不是完全没有研究者提供结构的主体。旧照片仅作开发；独立照片框的像素验收是另一个阶段，不参与本批学习。
