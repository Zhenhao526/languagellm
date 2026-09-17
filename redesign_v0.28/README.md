# v0.28：新材料上的完整形成开发复现

已完成4个新初始化来源、48次私人拟合和36次新通信训练。目标12布局的私人J由63.657%升至100%；通信训练均只覆盖18布局时，A/B的目标12自然J为6.308%/11.632%，主差+5.324个百分点，三来源提高、一来源持平。C将全部布局纳入通信训练，目标12为94.792%，属于已训练布局表现。

本轮补上新材料从私人准备到新协议形成的完整流程。它重复了有限私人经验迁移与表达缺口，没有证明组合语言或ICLR级创新性；正式独立确认和统一投稿论证仍未完成。

- [完整报告](/Users/xia/Documents/ChatGPT/语言/redesign_v0.28/results/formation_001/新材料上的私人经验与共同通信形成研究报告.md)
- [固定方案](/Users/xia/Documents/ChatGPT/语言/redesign_v0.28/固定执行方案.md)、[前置科学审查](/Users/xia/Documents/ChatGPT/语言/redesign_v0.28/前置科学审查.md)、[结果审查](/Users/xia/Documents/ChatGPT/语言/redesign_v0.28/结果审查.md)
- [独立分析](/Users/xia/Documents/ChatGPT/语言/redesign_v0.28/results/formation_001/analysis.json)、[统计复核](/Users/xia/Documents/ChatGPT/语言/redesign_v0.28/results/formation_001/raw_validation.json)、[执行复核](/Users/xia/Documents/ChatGPT/语言/redesign_v0.28/results/formation_001/audit_execution.json)
- [实际消息示例](/Users/xia/Documents/ChatGPT/语言/redesign_v0.28/results/formation_001/实际消息示例.md)与[展示选择规则](/Users/xia/Documents/ChatGPT/语言/redesign_v0.28/消息展示规则.md)
- [12图分配](/Users/xia/Documents/ChatGPT/语言/redesign_v0.28/data/selection.json)、[编码记录](/Users/xia/Documents/ChatGPT/语言/redesign_v0.28/data/encoder_receipt.json)、[训练终态](/Users/xia/Documents/ChatGPT/语言/redesign_v0.28/results/formation_001/terminal_receipt.json)

9张水果流程图与3张新水图，各类2train/1test，8训练图归一化；训练6食物×2水，测试3食物×1水，30布局×2mask得到720/180世界。9水果已开发暴露，唯一测试水重复于全部种子，属于开发复现。原确认48及另外4个水候选未使用；3张水已完成首次模型暴露。

现有官方DINOv2-L（304,368,640参数）冻结，12图MPS编码，原8锚图与缓存精确一致；每个新主体独立学习小型私人/通信接口。新来源34101–34104，各3划分、2方向，准备每人200步，私人及社会均2400步，CPU单线程正式训练与评价715.690秒，无新权重下载。

生产核心训练函数与v23 AST一致，非方形图池、来源与输入路径、初始化种子和记账作了适配。7项世界测试、40步开发批与配对初值前置通过；独立补充复算是在正式启动后完成，前置记录如实注明。正式统计复核全部720评价表、129600世界行；执行复核完整重放48私人终点、72社会方向和72个49码接收表。训练fixture只抽第1/2101/末步，未独立重训准备阶段或完整Adam轨迹。图表PNG已目视检查，PDF为同图导出，未另行渲染审查。

从项目根使用既有环境和固定数据，可选择一个不存在的输出目录重跑学习及独立复算：

```sh
redesign_v0.3/deployment/.venv/bin/python redesign_v0.28/run_formation.py --out redesign_v0.28/results/replay_001
redesign_v0.3/deployment/.venv/bin/python redesign_v0.28/analyze_results.py --out redesign_v0.28/results/replay_001
redesign_v0.3/deployment/.venv/bin/python redesign_v0.28/audit_formation.py --out redesign_v0.28/results/replay_001
redesign_v0.3/deployment/.venv/bin/python redesign_v0.28/plot_results.py --out redesign_v0.28/results/replay_001
```

`prepare_data.py`是本批已执行的编码入口，禁止覆盖已有data；本轮不需要再次编码。`build_report.py`、`message_examples.py`和`package_results.py`默认对应本批formation_001，完整来源、检查点及图表哈希见该目录的completion_manifest.json。
