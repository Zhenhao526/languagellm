# v0.23：私人状态经验与共同符号表达迁移

已完成四个新初始化来源的48次私人拟合、36次新共同通信训练。私人全30经验相对仅18种布局，使目标12图的私人双目标成功率由67.220%升到100%；随后都仅在18图学习通信时，目标12自然成功率由10.775%升到15.896%，主差+5.122个百分点，四来源均正。如果这12图也进入通信训练，则达到95.627%。

私人经验有有限迁移贡献，但当前私人行动成功不足以保证新协议自动表达其他私人熟悉状态。全私人经验组的目标12图不是全阶段零样本，通信全30组更不是通信留出。AUC有一来源反向；可见/历史资源分项表现存在取舍，未检验消息成分组合。整体论文目标仍在进行。

- [完整研究报告](results/experience_001/私人状态经验与共同符号表达迁移研究报告.md)
- [固定方案](固定执行方案.md)、[前置科学审查](前置科学审查.md)、[有限实现审查](实现前审查.json)
- [结果解释](结果解释.md)、[近邻与可证伪预测](近邻与可证伪预测.md)
- [独立分析](results/experience_001/analysis.json)、[独立执行复核](results/experience_001/audit_execution.json)、[统计比较](results/experience_001/comparison.json)
- [私人能力检查](results/experience_001/private_applicability.json)、[完整训练记录](results/experience_001/training_complete.json)
- [世界解析检查](world_selftest.json)、[正式前置记录](preflight_qa.json)、[启动环境错误记录](launch_history.json)、[启动后开发复核](development_followup_qa.json)
- [开发审计计算图修正](results/smoke_001/replay_implementation_correction.json)、[开发失败历史](results/smoke_001/analysis_history)
- [图表](results/experience_001/figures)、[生产运行](run_experience.py)、[环境](world.py)、[生产指标](metrics.py)、[独立复核程序](analyze_experience.py)、[绘图](plot_results.py)、[报告构建](build_report.py)、[封存程序](package_results.py)

## 实际范围

新初始化来源33101–33104，三次既有old18/added6/sealed6划分，每次两名独立主体。固定两帧、布局不动、末帧只再次显示食物或水；两mask精确配对。官方DINOv2-L视觉缓存不变，新基础私人资源准备共1,600更新、102,400动作；随后Camp只继承project，私人头和时序接口重新学习，通信阶段丢弃私人头、冻结视觉/时序接口并重置新协议。

私人old/all总48拟合，115,200更新、58,982,400所选需求动作；社会A私人18/通信18、B私人30/通信18、C私人30/通信30共36运行，86,400配对更新、44,236,800消息、88,473,600动作。每臂2,400步，CPU单线程，正式训练及评价约778.146秒。模型是官方自监督视觉底座与本项目的小型行动/通信接口，没有LLM、新模型下载或DINO推理。

独立分析重算720个保存的评价表，终点完整前向46,080私人世界、69,120社会发送世界以及全部72张49码接收表；训练外部流全核，test缓存46,080行全核，train缓存仅首末块17,664行。参数与Adam更新重放限于固定33101/p1：私人d0两臂第1/2101步共4次、社会三臂两时点共6次配对更新。没有独立重新训练整个轨迹；共同准备阶段只核源码、来源和实际继承project，不重训该阶段。

所有来源和失败保留。私人all的四来源×两mask共八项预定描述性门槛均通过，但并不表示无限环境中的全部能力。经验支持扩大也改变旧图复习频次和隐藏地点候选数，不能称纯知识/记忆操纵。49完整码足以记忆30布局；C的高分支持任务可学习性，不能独自证明组合语言。

## 复现

从项目根目录使用现有虚拟环境，选一个尚不存在的新输出目录，不覆盖封存数据：

```bash
redesign_v0.3/deployment/.venv/bin/python redesign_v0.23/run_experience.py --out redesign_v0.23/results/replay_001
redesign_v0.3/deployment/.venv/bin/python redesign_v0.23/analyze_experience.py --out redesign_v0.23/results/replay_001
redesign_v0.3/deployment/.venv/bin/python redesign_v0.23/plot_results.py --out redesign_v0.23/results/replay_001
```

Torch 2.14.0、NumPy 1.26.4；需要原v0.4缓存及列明的源码，来源hash见训练收据。绘图在导入环境NumPy后加载v0.9已有绘图依赖，不将整个依赖目录置于前置PYTHONPATH。第一次正式启动的辅助命令误用系统python，未创建正式目录或开始训练；改用原环境后继续，没有改生产代码。

40步开发来源99523只用于流程检查，没有据此更换预算、来源或主指标。正式启动时独立数值复核尚在完成，原前置记录明确保留pending状态；最终开发及正式审计均通过。开发审计器曾因等价反向计算图的浮点累加顺序不同失败，修正审计实现后逐位一致，无容差放宽或训练修改。

下一候选是与直接近邻对齐的结构先验基线，先核实输入与训练的可比性；尚未实施。独立图片和更多来源确认仍未完成，当前没有认证ICLR级新颖性或论文完成。
