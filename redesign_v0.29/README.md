# v0.29：训练共现结构与共同符号迁移

已完成v0.29相同曝光下的共现支持对照：同一私人all来源，24次新通信训练；两臂各18训练布局，每步每资源位置恰42曝光，整体相关信息量同为1bit。共同P6自然J为14.236%/17.130%，路径3−路径2主差−2.894个百分点，四来源两正两负，AUC也两正两负；训练布局J为98.148%/96.952%。主预测未获一致支持。两预定方向的平均人工片段重组都超出199整码重编码参照，但来源异质、架构偏置未分离，不能等同自主组合语言。下一步优先解除共同P6一一匹配的识别限制，先设计多对应的共同目标及单资源表达上界，不继续扫图追正差。见[完整报告](/Users/xia/Documents/ChatGPT/语言/redesign_v0.29/results/support_001/训练共现结构与共同符号迁移研究报告.md)、[实际消息](/Users/xia/Documents/ChatGPT/语言/redesign_v0.29/results/support_001/实际消息示例.md)和[复现索引](/Users/xia/Documents/ChatGPT/语言/redesign_v0.29/README.md)。独立材料确认与论文中心贡献仍未齐备。

两臂共用v28官方DINOv2-L冻结视觉/私人接口与固定图片缓存，全部通信按新namespace29029重新初始化。4来源×3坐标重复×2训练支持=24次双主体训练，各2400更新；57,600双主体更新、29,030,400消息、58,060,800资源动作。训练/评价记账209.086秒，含清单封存的终端209.635秒，CPU单线程。本轮无新图像、权重下载、DINO前向或私人训练。

原始片段重组FW/WF：路径3为7.485%/20.293%，路径2为15.201%/8.565%；整码随机双射均值约0.67%–0.77%。平均差异说明当前符号坐标组织存在可由分析者利用的有限对应。来源34101路径3 FW为0，34102同格参照上尾0.31；不能泛化为每个主体、方向或自发语法。整码重编码保持完整消息行为，不保持神经网络参数化，也不是另一批训练。

8支持/fixture测试、3人工协议指标测试、40步独立来源联调和独立前置审计通过。完整批次独立核对288协议表/51,840世界行和687,744个重编码target分数；执行审计重放全部8,640终点发送世界及48个49码接收表，抽144方向fixture/36,288消息轨迹。未重放完整Adam轨迹或再次验证旧DINO缓存；原输入按旧完成记录和封存哈希继承。初次联调审计依赖收集失败及修复记录完整保留，未改模型/指标。

- [固定方案](/Users/xia/Documents/ChatGPT/语言/redesign_v0.29/固定执行方案.md)、[两支持模块](/Users/xia/Documents/ChatGPT/语言/redesign_v0.29/support.py)、[事前检查](/Users/xia/Documents/ChatGPT/语言/redesign_v0.29/preflight_qa.json)
- [独立分析](/Users/xia/Documents/ChatGPT/语言/redesign_v0.29/results/support_001/analysis.json)、[原始统计复核](/Users/xia/Documents/ChatGPT/语言/redesign_v0.29/results/support_001/raw_validation.json)、[执行复核](/Users/xia/Documents/ChatGPT/语言/redesign_v0.29/results/support_001/audit_execution.json)
- [科学审查](/Users/xia/Documents/ChatGPT/语言/redesign_v0.29/结果审查.md)、[近邻原文核查](/Users/xia/Documents/ChatGPT/语言/paper_program/mechanism_pressure_20260916/近邻与可证伪预测.md)
- [原始消息展示及行号](/Users/xia/Documents/ChatGPT/语言/redesign_v0.29/results/support_001/message_examples.json)、[图表](/Users/xia/Documents/ChatGPT/语言/redesign_v0.29/results/support_001/figures/02_support_outcomes.png)
- [训练完成](/Users/xia/Documents/ChatGPT/语言/redesign_v0.29/results/support_001/training_complete.json)、[终端状态](/Users/xia/Documents/ChatGPT/语言/redesign_v0.29/results/support_001/terminal_receipt.json)、[封存清单](/Users/xia/Documents/ChatGPT/语言/redesign_v0.29/results/support_001/completion_manifest.json)

从项目根，使用一个不存在的输出目录重跑核心训练与复核：

```sh
redesign_v0.3/deployment/.venv/bin/python redesign_v0.29/run_support.py --out redesign_v0.29/results/replay_001
redesign_v0.3/deployment/.venv/bin/python redesign_v0.29/analyze_results.py --out redesign_v0.29/results/replay_001
redesign_v0.3/deployment/.venv/bin/python redesign_v0.29/audit_results.py --out redesign_v0.29/results/replay_001
redesign_v0.3/deployment/.venv/bin/python redesign_v0.29/plot_results.py --out redesign_v0.29/results/replay_001
redesign_v0.3/deployment/.venv/bin/python redesign_v0.29/message_examples.py --out redesign_v0.29/results/replay_001
```

图像依赖与旧私人终点的绝对路径列于invocation.json。复现上述核心不需要再次编码或下载。build_report.py和package_results.py还需要实际终端记录、科学审查和图表QA，不把脚本启动当作完成。内部formal标记表示完整批次，科学定位仍是开发实验。原材料、初始化来源、共同P6的匹配限制使本轮不能充当独立确认。
