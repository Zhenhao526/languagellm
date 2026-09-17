# v0.18 固定策略条件梯度方差

已完成72个既有策略检查点、1,296个世界、63,504条消息梯度的直接测量，没有新增训练。第0、100、600步，匹配基线相对同策略解析置换基线的方差平均降低0.63%、0.22%、62.12%；第600步四来源均为正，早期各有一个反向来源。该结果限于old18地图、一对训练照片的奖励score条件方差，不等于完整训练梯度、组合能力或v0.17终点差异的因果中介。

- [完整研究报告](results/gradient_001/固定策略下的通信学习梯度方差研究报告.md)、[结构化分析](results/gradient_001/analysis.json)、[科学图 PNG](results/gradient_001/figures/01_fixed_policy_variance.png)、[PDF](results/gradient_001/figures/01_fixed_policy_variance.pdf)
- [固定执行方案](固定执行方案.md)、[前置审查](前置审查.md)、[研究含义与下一步](研究含义与下一步.md)
- [运行器](run_gradient.py)、[矩测量模块](gradient_moments.py)、[实际输入与冻结源码](results/gradient_001/invocation.json)、[测量完成凭证](results/gradient_001/measurement_complete.json)
- [开发预检](preflight_qa.json)、[独立执行审计](results/gradient_001/audit_execution.json)：全部世界前向及矩检查；正式4个预定世界额外进行完整score重放、手写double Jacobian与有限差分核验。
- [独立原始复算](results/gradient_001/independent_recount.json)、[分析对照](results/gradient_001/analysis_comparison.json)、[图表核验](results/gradient_001/figure_qa.json)、[报告读审](results/gradient_001/audit_report_review.json)
- [同期水图元数据复核](../paper_program/visual_confirmation_v2_water_review/元数据复核报告.md)：全部79项，28/43/8仅为描述性标签，无新增可用图片、像素或模型调用。

数据来自原v0.15 reset_scaled四个开发来源31101–31104、三个分区、两个方向、0/100/600检查点；原始ScaledCampAgent严格加载。官方冻结DINOv2 ViT-L/14沿用已有视觉缓存，私人发送策略score为9张量、43,319参数。食物/水各取原训练池最小特征行0/31，所有检查点相同。每个策略基线的两个比较共享该时点发送者与伙伴。

报告主图采用原发送角色系数1/2所对应的1/4方差尺度；未加权结果、最优基线参照和数值残差同时保存。统计先平均来源内分区/方向，再平均四来源；三个时点全部报告，独立来源数仍为4。正式测量过程约35.43秒，独立核验、分析和文档另计。

复跑诊断需要本地既有权重、特征及相同依赖，输出目录必须尚不存在：

```sh
redesign_v0.3/deployment/.venv/bin/python redesign_v0.18/run_gradient.py --out redesign_v0.18/results/gradient_reproduction
```

查看结果无需重跑。旧版本、原主比较与失败开发记录均保留；本轮不新增优化器、不回灌最优基线、不把分析真值输入主体。对完整训练过程和未见组合自然表达的判断仍需专门实验。
