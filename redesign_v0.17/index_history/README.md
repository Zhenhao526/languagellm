# 语言诞生研究

最新完成[v0.16策略与价值输入幅度实验](redesign_v0.16/results/branches_001/策略与价值输入幅度的通信形成研究报告.md)：24次新社会学习与24次既有参考组成2×2。未训练组合自然成功率为不缩放4.44%、仅策略6.90%、仅价值7.97%、两者13.63%；主要单支路差−1.08个百分点，四来源方向不一致。辅助比较中，策略已缩放时再缩放价值提高6.73个百分点，四来源均提高。事后全日志核查发现四组裁剪全程未激活，价值影响的计算路径收窄到策略优势基线及后续互动。执行、独立统计和日志核验通过，见[v0.16复现索引](redesign_v0.16/README.md)。下一候选是发送批次内置换基线对应关系，尚未运行；当前仍为旧图片、四个继承来源的开发证据。

此前完成[v0.15视觉编码幅度实验](redesign_v0.15/results/scaled_001/视觉编码幅度与共同符号形成研究报告.md)：新增12次社会学习，复用保留/重置各12个参考。未训练组合成功率为保留22.04%、重置4.44%、固定幅度补偿13.63%；补偿相对重置提高9.18个百分点，四来源均改善，保留减补偿仍差8.41个百分点（三正一负）。幅度影响能恢复部分成绩，不能将私人准备收益全部归于世界知识或组合能力。熟悉组合也86.53%→92.44%，保留为96.62%。完整执行和独立复算通过，见[v0.15复现索引](redesign_v0.15/README.md)。策略/价值支路拆分现已在v0.16完成；独立图片确认仍未完成。

独立图片的[第三批流程验收](paper_program/visual_confirmation_v1/pixel_workflow_003/README.md)完成17张原图下载及匿名评审，净增4张有限流程可用图；三批累计11/24（苹果、香蕉、橙子各3，水2），仍缺13张，另3项待核未计通过。确认48图未访问，新图尚未进入模型实验。后续水场景来源框仅为候选，见[可行性审查](paper_program/visual_confirmation_v1/pixel_workflow_003/后续图库可行性审查.md)。

此前完成[v0.14私人非语言经验迁移对照](redesign_v0.14/results/reset_001/私人非语言经验的迁移研究报告.md)：新增12次重置社会学习、复用同源12个control。保留私人接口时未训练sealed通信22.04%，重置后4.44%，四来源保留优势均为正、平均+17.60个百分点；熟悉组合也96.62%对86.53%，支持综合迁移收益，不能单独归为组合能力。编码范数同时明显改变，因此提出old训练支持校准的单标量尺度控制；该候选现已在v0.15执行。完整执行、独立重算及图表核对通过，见[v0.14复现索引](redesign_v0.14/README.md)。

此前[v0.13私人空间能力与共同符号形成](redesign_v0.13/results/spatial_001/私人空间能力与共同符号形成研究报告.md)：24组私人主体对训练后接24组通信学习。两组私人未训练组合双目标能力约95%，社会熟悉组合约97%；主要未训练组合的自然通信为22.04%与19.87%，增加空间一致性约束后四个来源2升2降，未显示稳定收益。对照的私人能力本来已强，新增约束的能力操纵很小。完整执行、分析、原始行为重算和固定消息样例见[v0.13复现索引](redesign_v0.13/README.md)。

此前[v0.12固定发送者接收诊断](redesign_v0.12/results/receiver_001/固定发送者的接收诊断研究报告.md)：96个接收拟合、4个继承主体对。共同24图的最优联合解码为73.26%，奖励接收者已达72.04%；监督虽把预测损失大幅降低，随机接收成功率却下降约4.85个百分点。该结果区分了消息歧义、接收目标和拟合不足，不能再把很大的预测损失直接解释为同样大的任务收益缺口。完整执行、解析和独立汇总核查通过，见[复现索引](redesign_v0.12/README.md)。

此前[v0.11旧伙伴约定压力实验](redesign_v0.11/results/anchoring_001/旧伙伴约定压力与新表达研究报告.md)完成36个适应运行，来自4个主体对、3个重复划分。中点解除旧伙伴要求，相对持续保留要求的新增图后半程平均成功率仅提高1.261个百分点，四种子差+0.804、+1.852、+2.390、−0.003个百分点；始终未训练图仍约4%—5%。各组都能扩展完整码的用途，尚不支持“旧约定是主要瓶颈”的强解释。

上一轮[v0.10未训练组合迁移实验](redesign_v0.10/results/generalization_001/新组合经验与未训练组合迁移研究报告.md)完成60个社会训练阶段与4组独立个人控制。两端学习使新增组合成功率达到45.11%，始终未训练组合为4.71%；只练旧图控制为1.82%，四种子主差−0.68、+6.91、+2.76、+2.57个百分点。新增约定学习与未训练组合迁移分开解释；尚未获得稳定组合泛化。

本任务近期的Qwen三人能力控制、共享计划执行诊断及新生态无通信边界，见[持续研究索引](research_program/README.md)。最新共享计划诊断完成144调用：原样重放4/12，添加共同计划10/12；这是执行控制，尚未形成新符号语言。视觉路线的既有实验另按下文保留。

视觉路线最新追加[策略学习权重对照](research_program/policy_gain_study/结果与下一步.md)：64个新运行与1024份检查点方向测量已完成并独立核验。提高策略权重并未稳定修复未见组合；加性J为4.16%→0.86%，联合为8.75%→7.38%。下文各版本原报告保留，最新机制解释与源码记录见该独立研究目录。

当前核心问题：**哪些非语言能力已经足以支持共同符号的诞生，哪些能力会改变诞生的过程和最终结构？**

第一阶段限定这台 Mac 的本地资源，使用官方 DINOv2-L 冻结视觉权重及独立行动／交流接口。此前v0.9完成36组新组合适应实验：在已有24种地图约定后加入其余6种，比较仅发送端、仅接收端、双方学习。新组合双目标成功率从共同起点7.19%分别到38.70%、10.79%、58.86%；双方学习在四个继承种子均优于单端，旧组合宏平均约93%。新图已参加适应训练，尚不能称零样本组合语言。v0.3 Minecraft/VPT继续暂停，此前结果保留。

- [v0.10完整报告](/Users/xia/Documents/ChatGPT/语言/redesign_v0.10/results/generalization_001/新组合经验与未训练组合迁移研究报告.md)与[复现索引](/Users/xia/Documents/ChatGPT/语言/redesign_v0.10/README.md)：18旧图→新增6图、另6图始终不训练；756,940项执行检查通过，协议检查重放230.4万个接收世界及428,265条匹配发信记录。
- [论文研究推进](/Users/xia/Documents/ChatGPT/语言/paper_program/README.md)与[近邻文献审查](/Users/xia/Documents/ChatGPT/语言/paper_program/literature_20260915/novelty_gap.md)：持续检验旧约定的扩展约束、独立数据和直接基线；论文证据尚未齐备。

- [v0.9新组合通信适应研究报告](/Users/xia/Documents/ChatGPT/语言/redesign_v0.9/results/adaptation_001/新组合通信适应研究报告.md)与[复现索引](/Users/xia/Documents/ChatGPT/语言/redesign_v0.9/README.md)：36个配对适应运行、四个继承种子，246,496项执行审计全部通过；区分接收码覆盖、新旧消息冲突和自然产出。
- [v0.9协议结构](/Users/xia/Documents/ChatGPT/语言/redesign_v0.9/results/adaptation_001/协议结构报告.md)与[固定消息实例](/Users/xia/Documents/ChatGPT/语言/redesign_v0.9/results/adaptation_001/消息实例.md)：仅发送端学习大幅改善自然通信，人工拼接近乎不变；仅接收端改善拼接但自然通信较弱。任务适应与片段复用分开解释。

此前v0.8完成60个资源互补收益实验与4组个人能力诊断。加和、混合、严格联合三组的训练地图双目标成功率为76.67%、92.80%、93.65%，未见组合为5.91%、7.13%、11.26%。严格联合相对加和的未见组合提升5.36个百分点，四种子中三个提高；熟悉地图的提升更明显，自主新组合表达仍不可靠。

- [v0.8资源互补收益实验报告](/Users/xia/Documents/ChatGPT/语言/redesign_v0.8/results/complementarity_001/analysis/资源互补收益实验报告.md)与[复现索引](/Users/xia/Documents/ChatGPT/语言/redesign_v0.8/README.md)：四个新种子、60个配对运行，固定输入、世界和两次行动，区分效用压力、学习信号与自然组合产出。
- [v0.8协议结构](/Users/xia/Documents/ChatGPT/语言/redesign_v0.8/results/complementarity_001/protocol_analysis.md)、[实际消息实例](/Users/xia/Documents/ChatGPT/语言/redesign_v0.8/results/complementarity_001/消息实例.md)及[完整执行审计](/Users/xia/Documents/ChatGPT/语言/redesign_v0.8/results/complementarity_001/audit_execution.json)：60/60运行、1,175,469项检查通过，2,880,000个社会世界及153,600次个人诊断选择独立重算；120方向、12,000次整体重编码参照。

v0.8的结构结果没有随联合收益单调提高：加和/混合/严格联合的片段替换为31.75%/48.22%/40.42%，训练供体人工拼接为21.96%/35.46%/19.30%。中等互补收益的片段指标更高，严格联合的自然双目标成绩更高，两类结果分开解释。

此前v0.7完成52个视觉地点局部性实验与12组个人能力诊断。保留地点槽、置换地点槽、可逆正交混合三种输入的未见组合单目标准确率分别为47.69%、43.93%、44.59%，自然消息同时支持两个目标仅2.82%、1.91%、1.65%。未获得稳定的一般局部性优势。v0.8每世界执行两个动作，不能直接用跨版本差值估计改变收益的因果效应。

- [v0.7视觉地点局部性实验报告](/Users/xia/Documents/ChatGPT/语言/redesign_v0.7/results/binding_001/analysis/视觉地点局部性实验报告.md)与[复现索引](/Users/xia/Documents/ChatGPT/语言/redesign_v0.7/README.md)：四个新种子、三种私有输入变换，匹配参数初值、世界批次和训练预算；可逆混合不等于移除世界知识。
- [v0.7协议干预与重编码](/Users/xia/Documents/ChatGPT/语言/redesign_v0.7/results/binding_001/protocol_analysis.md)与[完整执行审计](/Users/xia/Documents/ChatGPT/语言/redesign_v0.7/results/binding_001/audit_execution.json)：52/52运行、874,023项检查通过，2,496,000条社会与460,800条个人终点记录重算；104方向、10,400次整体重编码参照。

此前v0.6完成56个六地点组合留出实验。未见组合的单目标准确率为课程36.14%、混排37.74%、直接45.39%、原子码41.27%。直接条件的自然消息同时支持两个目标仅4.08%；人工拼接训练消息成分后15.40%，其整体重编码参照0.47%。v0.7新增共享非线性槽编码，不能把跨版本分数差直接当作局部性的因果效应。

- [v0.6六地点组合留出实验报告](/Users/xia/Documents/ChatGPT/语言/redesign_v0.6/results/recombination_001/analysis/六地点组合留出实验报告.md)与[复现索引](/Users/xia/Documents/ChatGPT/语言/redesign_v0.6/README.md)：56个固定预算运行、134,400次更新、2,688,000条终点评估独立核查；旧照片上的扩大探索，非新视觉确认。
- [v0.6协议干预与重编码](/Users/xia/Documents/ChatGPT/语言/redesign_v0.6/results/recombination_001/protocol_analysis.md)与[完整执行审计](/Users/xia/Documents/ChatGPT/语言/redesign_v0.6/results/recombination_001/audit_execution.json)：区分单目标成功、自然双目标复用和由实验者拼接的组合；完整保留课程负结果及原子码的部分泛化。

此前v0.5完成54个渐进探索运行，依次加入私人需求、观察延迟与三轮持续采集。隐藏目标任务中课程训练76.11%、直接训练89.27%、同批次混排91.03%；持续采集中有通信83.79%、始终无通信39.51%。发现了有限地图中的局部成分复用实例，未见组合仍不稳定。

- [v0.5阶段实验报告](/Users/xia/Documents/ChatGPT/语言/redesign_v0.5/results/progression_001/analysis/阶段实验报告.md)与[代码及复现索引](/Users/xia/Documents/ChatGPT/语言/redesign_v0.5/README.md)：三个新种子、54个运行、短期记忆与动态资源任务；保留课程不利及组合泛化不足的结果。
- [v0.5协议结构与重编码参照](/Users/xia/Documents/ChatGPT/语言/redesign_v0.5/results/progression_001/protocol_analysis_extended_null.md)、[固定情境消息干预](/Users/xia/Documents/ChatGPT/语言/redesign_v0.5/results/progression_001/fixed_context_probe.md)及[完整执行审计](/Users/xia/Documents/ChatGPT/语言/redesign_v0.5/results/progression_001/audit_execution.json)：区分有效通信、有限成分复用和未见地图组合；重编码扩展明确标为看到结果后的探索。

此前v0.4完成94个新增正式运行：十个形成种子的课程通信成功率98.23%，直接完整任务71.37%；七个新群体中，发送与接收均可学习的交叉成功率81.74%，但没有群体让四条交叉边全部双向达标。v0.4与v0.5任务及接口不同，百分比不能直接跨版本比较。

- [完整研究报告 Word](/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/完整研究报告/预训练视觉主体中符号约定的形成与适应.docx)与[可编辑正文](/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/完整研究报告/研究报告.md)：方法、全部主要结果、种子区间、理论边界、局限及12项直接相关文献。
- [完整研究执行方案](/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/完整研究_执行方案.md)与[复现说明](/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/完整研究报告/复现说明.md)：40组形成、7个新来源、47组机制新增运行；13,893,632个新增终点案例独立重放通过。
- [形成确认结果](/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/results/formation_confirm_001/形成确认_结果.md)与[机制结果](/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/results/mechanism_001/机制实验_结果汇总.md)：保留1209过程后退、1202历史准备重合及九种子敏感性，机制确认七群体与旧探索分列。
- [此前已有约定接触实验](/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/已有约定接触实验_结果.md)：九组检查点再适应，分别跟踪新配对、旧配对及发送／接收变化，保留单向失效和未通过结果。
- [接触执行方案](/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/已有约定接触实验_执行方案.md)与[训练审查](/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/接触实验_实施与运行审查.md)：50项运行前检查；[独立通信核查](/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/results/contact_001/独立接触通信核查.md)重放176万余案例、90个漂移检查点。
- [此前课程对照与伙伴比较](/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/课程对照与伙伴比较_结果.md)：课程组92.52%、直接完整任务组71.54%；四主体全六对固定66.91%、轮换97.42%，尚不等于新伙伴零样本泛化。
- [伙伴执行方案](/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/对照与伙伴实验_执行方案.md)与[伙伴实验独立核查](/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/伙伴实验_实施与运行审查.md)：起点、人均经验、世界批次及配对暴露核查，39项运行前检查通过。
- [此前课程与发信辅助实验](/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/课程实验_结果.md)：六组形成过程、辅助撤除、双向干预及约定结构；无辅助也能建立通信。
- [此前课程固定方案](/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/课程实验_执行方案.md)与[独立通信核查](/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/results/curriculum_001/独立通信核查.md)：同起点、同场景配对，保存全部种子；294,912 条最终案例重算和检查点推理吻合。
- [v0.4 第一轮尝试结果](/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/第一轮尝试结果.md)：六组本地实验、实际任务示例、固定偏好与消息干预。
- [v0.4 流程复核与平台期诊断](/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/debug_flow/流程检查报告.md)：通信正控、参数更新、密集过程记录及精确复跑。
- [v0.4 实施和复现](/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/README.md)：二人营地的资源配合、四主体伙伴轮换、私人视觉观察和有限符号交流。
- [最小资源协作实验 v0.4（原讨论稿）](/Users/xia/Documents/ChatGPT/语言/redesign_v0.4/最小资源协作实验_讨论稿.md)：保留初始方案，实际库存规则已按核查结果调整。
- [此前研究设计 v0.3](/Users/xia/Documents/ChatGPT/语言/redesign_v0.3/研究设计_v0.3_预训练主体与资源协作.md)：资源互补与信息分散的独立操纵、共同符号形成过程和实际分工测量。
- [VPT 2x 本地部署](/Users/xia/Documents/ChatGPT/语言/redesign_v0.3/deployment/README.md)：官方 2.485 亿参数策略已完成 CPU/MPS 录像动作推理。后来完成真实环境脚本对照，主体能力验证中途暂停；尚未进入社会学习。最新状态见[暂停记录](/Users/xia/Documents/ChatGPT/语言/redesign_v0.3/calibration/暂停记录.md)。
- [新方案、模型审查与补充文献索引](/Users/xia/Documents/ChatGPT/语言/redesign_v0.3/README.md)：模型训练经历审查、生态环境选择、12 项补充来源及 6 篇新增公开 PDF。

历史资料：

- [第一轮实际实验报告](/Users/xia/Documents/ChatGPT/语言/pilot_round1/第一轮实验报告.md)：14次本地运行已完成，包含结果、消息干预、局限与下一步问题。
- [实验代码与复现说明](/Users/xia/Documents/ChatGPT/语言/pilot_round1/README.md)：环境、固定设置、运行命令、日志和权重位置。
- [项目定位与本地先导方案](/Users/xia/Documents/ChatGPT/语言/项目定位与本地先导方案_v0.2.md)：保留实施前方案；实际轮数和结果以上述实验报告为准。
- [文献调研与论文库](/Users/xia/Documents/ChatGPT/语言/文献调研_2026-09-14/README.md)：58项文献，51项取得PDF，含综合报告和逐篇证据。
- [早期讨论记录](/Users/xia/Documents/ChatGPT/语言/语言涌现研究讨论记录_2026-09-14.md)：保留研究方向演变与原计划书审查。

已完成课程、有效消息通道、伙伴覆盖及完整发送／接收函数可塑性的受控检验，v0.5进一步探索短期记忆和动态资源，v0.6扩展地点与平衡留出，v0.7比较视觉信息的局部组织方式，v0.8比较资源收益的互补强度，v0.9检验新组合加入后共同约定的适应，持续区分局部片段复用和自然新组合产出。视觉预训练的独立贡献、长期记忆、自由分工和代际传递仍待研究。当前结果不识别人类语言起源的最小能力集合，分工也不能预设为其已知必要条件。
