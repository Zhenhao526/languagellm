# 语言诞生研究

- **最新完成[接收者角色 C−A 单槽与符号重编码确认探针](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_receiver_role_crossover_confirmatory_slot_recode_probe/单槽重编码_结果与下一步.md)：**在独立确认终点的 A/C 两个角色、8 个种子、`static/rematched × live/silent` 共 64 个冻结策略块上，把供体首窗包拆成四个单槽，并加入两种固定八符号双射。整包 aligned−placebo 计划转移为 A **+8.221/+6.891 pp**、C **+3.329/+1.984 pp**；单槽缩小到约 **+0.30–+1.86 pp**，没有单槽复现整包幅度；`recode_add1/xor4` 的 A、C 点估计均为负。C−A 整包 **−4.899 pp**（t(7) CI [−8.485,−1.314]），单槽约−0.68至−1.19 pp，重编码约+0.81 pp但区间跨零。独立审计重放4,866,048行、2,433,024条live，`max_abs_error=0`；新权威批次无优化更新、无外部模型调用。结果支持当前冻结接口依赖整包、槽位和绝对 token 身份的窄机制解释，仍不能称词义、组合语法或语言起源。

- **最新完成[两批角色语义转移合并描述](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_receiver_role_crossover_confirmatory_semantic_transfer_probe/两批区块合并_结果与下一步.md)：**合并两批独立源种子共 16 个种子后，A/B/C 的 static aligned−placebo 计划转移为 **+9.360/+8.176/+4.343 pp**，rematched 为 **+8.146/+6.678/+2.836 pp**；C−A **−5.163 pp**（t(15) CI [−7.401,−2.926]），B−A 区间跨零。该项是同协议的描述性二次汇总，没有新增前向或优化更新；它强化了 C 相对 A 的方向性，但仍是冻结策略任务响应，不是词义或语言起源证据。

- **最新完成[第三人冲突惩罚梯度实验](research_program/triadic_conflict_penalty_study/冲突惩罚_结果与下一步.md)：**在 8 个新种子、c0/c25/c50/c75/c100 五档 reciprocal 冲突成本和 strict 硬约束锚点上，交叉 PL/FI、static/rematched、live/silent，共 384 次训练。预注册目标搭档率的 live−silent 剂量斜率在 PL 为 **−12.94 pp/penalty**（中心化 AUC，t(7) CI [−14.02,−11.87]，8/8种子为负），c100−c0 为 −11.31 pp；FI 斜率 −0.49 pp，区间跨零。成本升高显著降低 PL 的第三人冲突率并提高物理执行和惩罚后 reward，但执行后目标搭档率转向 strict 的负向模式；这支持参与结构机制，不支持词义或语言起源。独立审计重放 1,920 个 checkpoint 和 384 个终点，`max_abs_error=0`；训练与评价共 8,413,839,360 次模块前向，四张图已人工查看。


- **最新完成[接收者角色分层 aligned/placebo 独立确认](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_receiver_role_crossover_confirmatory_semantic_transfer_probe/接收者角色分层_确认结果与下一步.md)：**使用独立源种子66709–66716的96个角色交叉终点，复用冻结的`kind/length/destination`同背景消息移植与循环错配placebo。A/B/C 的 static aligned−placebo 计划转移为 **+8.221/+7.594/+3.329 pp**，rematched 为 **+6.891/+5.869/+1.984 pp**；C−A 为 **−4.899 pp**（t(7) CI [−8.485,−1.314]），B−A 区间跨零。独立审计重放7,299,072行、3,649,536条live，`max_abs_error=0`；图表已检查。该批次复现了C相对A较弱的任务内容响应方向，但仍是后验确认，不能称词义、组合语法或语言起源。

- **最新完成[接收者角色交叉独立确认](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_receiver_role_crossover_confirmatory_study/接收者角色交叉确认_结果与下一步.md)：**在独立源种子66709–66716上重做A/B/C替换的新学习者适应，96次训练、576个checkpoint和96个终点评价。A/B/C 的 static/rematched Q AUC 为 **+16.341/+21.198、+19.718/+15.171、+18.139/+13.726 pp**；B−A、C−A区间均跨零。Q增益仍主要来自物理执行，六个角色×日程单元的目标搭档终点均下降。独立审计`max_abs_error=0`，图表已检查；这确认任务协议适应，不是语言形成证据。

- **最新完成[接收者角色分层 aligned/placebo 内容转移探针](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_receiver_role_crossover_semantic_transfer_probe/接收者角色分层_语义转移结果与下一步.md)：**在角色交叉的96个终点上，按 `kind/length/destination` 做同背景源端消息移植与同层循环错配 placebo。A/B/C 的 static aligned−placebo 计划转移为 **+10.499/+8.759/+5.356 pp**，rematched 为 **+9.401/+7.488/+3.688 pp**；C−A 为 **−5.428 pp**（t(7) CI [−9.094,−1.761]），B−A 区间跨零。搭档转移逐行恒为零，重配×通信交互均跨零。独立审计重放7,299,072行，`max_abs_error=0`，图表已检查。该结果是角色分层的后验任务内容响应证据，尚未证明词义或语言起源。

- **最新完成[接收者角色交叉：新学习者适应](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_receiver_role_crossover_study/接收者角色交叉_结果与下一步.md)：**在同一已审计的 `altpair_001` 完整策略上，分别替换 A、B、C 并只训练被替换角色；8 个种子 × 3 角色 × static/rematched × live/silent 共 96 次训练。三角色的 live−silent Q AUC 均稳定为正（A **+26.493/+17.476 pp**、B **+17.976/+18.671 pp**、C **+19.769/+20.380 pp**），但 B−A、C−A 区间都跨零。增益主要来自物理执行，目标搭档终点在六个角色×日程单元都下降；独立审计重放576个checkpoint和96个终点，`max_abs_error=0`，图表已检查。该结果是角色配平后的任务协议诊断，不是词义、组合语法或语言起源证据。

- **最新完成[规则 × 信息 × 通信全因子实验](research_program/triadic_rule_communication_study/规则通信_结果与下一步.md)：**8 个独立种子 × strict/reciprocal × PL/FI × static/rematched × live/silent 共 128 次训练全部完成。PL 中 reciprocal−strict 的目标搭档率交互为终点 **+13.34 pp**（t(7) CI [+11.79,+14.89]）、中心化 AUC **+12.79 pp**（[+11.07,+14.52]）；FI 中为 **+0.71/+0.34 pp**，区间均跨零。strict/PL 的 live 通信把物理执行率从约38%提高到88%，但执行后的目标搭档率从约82%降到67%；reciprocal/PL 将该负向选择效应压到约−2 pp。独立审计重放 768 个 checkpoint 和 128 个终点，`max_abs_error=0`；训练与评价共 4,768,174,080 次模块前向，图表已人工查看。结果只支持结算协议调节任务内通信收益，不能称词义、组合语法或语言起源。


- **最新完成[单槽联合组合留出转移探针](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_compositional_slot_transfer_probe/单槽转移_结果与下一步.md)：**在 `matched_control_005` 的64个冻结策略上，把第一窗供体包拆成四个单槽和一个整包掩码；每策略3,456个案例、11,232个世界，正式批次重放20,477,952次模块前向。四个单槽的 aligned−placebo 计划转移在 seen/all、static/rematched四格均跨零，整包参照复现先前近零结果；独立审计重放1,105,920行、552,960条live干预，`max_abs_error=0`。当前没有证据表明整体零效应只是被某个稳定语义槽位稀释；这仍是冻结策略行为诊断，不能称词义、组合语法或语言起源。

- **最新完成[同架构 A 全组合匹配控制](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_compositional_matched_control_semantic_transfer_probe/matched_control_结果与下一步.md)：**把只见联合组合 A 与同一初始化、同一父代 B/C、同一随机流下的全组合 A 放到相同留出 aligned/placebo 探针中。64 个冻结策略块、3,456 条案例/策略；seen A 的 static/rematched live 转移为 **+0.3595/−0.4221 pp**，all A 为 **+0.2904/−0.5308 pp**，all−seen 为 **−0.0691/−0.1087 pp**，区间均跨零。独立重放 221,184 行、110,592 条 live 干预，`max_abs_error=0`。全组合训练没有恢复稳定的内容转移，排除了“只因 A 没见过全组合”这一简单解释；不能称词义、组合语法或语言起源。

- **最新完成[联合组合留出上的 aligned/placebo 语义转移探针](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_compositional_holdout_semantic_transfer_probe/组合留出语义转移_结果与下一步.md)：**在 8 个种子、static/rematched × live/silent 共 32 个冻结策略上，测试只见部分联合组合训练的新接收者 A 是否把未见联合组合的源端需求内容转移到行动方案。每策略 3,456 条案例、11,232 个世界；A 的 aligned−placebo 计划转移为 static **+0.3595 pp**（t(7) CI [−0.5050,+1.2240]）、rematched **−0.4221 pp**（[−1.4202,+0.5760]），全组合父代 B/C 为 **+10.2564/+9.4929 pp**。独立重放 110,592 行、55,296 条 live 干预，`max_abs_error=0`。该结果给出联合组合留出上的内容可识别性负边界，不支持把任务收益称为组合语义、词义或语言起源证据。


- **最新完成[多合法行动 aligned/placebo 消息转移探针](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_factorized_neutral_altpartner_semantic_transfer_probe/结果与下一步.md)：**在已审计的两个可替代搭档、显式 neutral/engage 策略上，对 16 个种子 × static/rematched × live/silent 共 64 个冻结策略做同背景个人需求邻居的源端首窗消息移植，并用同发送者×轴×背景循环错配 placebo。独立重放 4,866,048 条案例、2,433,024 条 live 干预，最大误差 0。aligned−placebo 计划转移为 static **+11.3456 pp**、rematched **+9.4873 pp**；重配×通信交互 **−1.8582 pp**（t(15) CI [−4.1845,+0.4680]）。结果提供了比单纯执行率更接近内容响应的冻结策略证据，但不支持随机重配增强，也不能称词义、组合语法或语言起源。

- **最新完成[新接收者联合组合留出实验](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_new_receiver_compositional_holdout_study/组合留出_结果与下一步.md)：**A 只在 1,248 个三人联合需求组合上适应，留出 312 个完整语义轨道；训练与留出对每个角色都保留全部 24 个单体需求值，因此直接检验组合留出。8 个种子 × static/rematched × live/silent 共 32 个新训练运行，并以已审计 generation-2 all-needs 运行作匹配控制。留出 Q 的 live−silent：只见组合 static/rematched **+21.46/+17.11 pp**，all-needs 控制 **+23.05/+16.92 pp**；两日程按种子平均分别 **+19.29 pp**（95% t(7) CI [14.81,23.76]）和 **+19.98 pp**（[15.06,24.90]）。只见组合−控制的交互区间均跨零（static −1.59 pp；rematched +0.19 pp），留出组合相对已见组合在新布局上也没有额外惩罚。机制上通信主要提高物理执行率（约+41–58 pp），执行后目标搭档率下降；不能把结果称组合语言、词义或语言起源。独立审计重放 384 个 checkpoint 与 64 个最终状态，122,923,008 次模块前向，最大误差 0；当前仍为本地 NumPy tanh MLP，无 LLM/视觉模型/外部网络调用。


- **最新完成[因素留出 aligned/placebo 消息转移探针](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_factorial_formation_study/semantic_transfer_heldout_结果与下一步.md)：**在 `factorial_001_corrected` 的冻结策略上，对 16 个种子 × factorial/saturated × strict/reciprocal × live/silent 做源端首窗消息移植，并在相同发送者×变化轴×方向层内循环错配 placebo。共 128 个策略块、328,704 条案例、11,833,344 个世界；独立重放最大误差 0。`aligned−placebo source_pull` 的 factorial−saturated 交互为 strict **−0.1336 pp**（t(15) CI [−0.3513,+0.0841]），reciprocal **+0.0068 pp**（[−0.1828,+0.1963]）。消息能改变行动，但没有显示未见对象×长度组合的源端需求—消息对齐增量；不能称词义、组合语法或语言起源。


- **最新完成[两代学习者替换链](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_protocol_chain_study/链式传递_结果与下一步.md)：**在代1已审计 live full-C 终点上，代2替换A、代3再从代2 live终点替换B；8个种子×2代×static/rematched×live/silent共64次运行。最终留出Q的live−silent为代2 **+19.54 pp**（95% t(7) CI [+14.60,+24.49]），代3 **+14.06 pp**（[+10.74,+17.39]）；中心化监测AUC分别+17.34和+13.34 pp。live子代相对live父代的Q保真度变化为代2−1.81 pp（[−5.39,+1.76]），代3−6.67 pp（[−13.09,−0.25]），说明通信增益可沿短链保留但第二次替换出现衰减。增益主要来自物理执行率（+50.47/+36.30 pp），执行后目标搭档率仍下降（−11.22/−6.80 pp），不能称词义、组合语法或语言起源。独立审计重放384个checkpoint、64个最终评估和64个父代评估，129,392,640次模块前向，最大误差0；192,000行配对日志、冻结模块和代际链路均通过。当前仍为本地NumPy MLP，无LLM/视觉模型/外部网络调用。

- **最新完成[新接收者模块消融](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_new_receiver_module_ablation_study/模块消融_结果与下一步.md)：**在同一冻结 A/B、新 C 初始化和随机流下，比较 full、action-only、sender-only 的 live/silent 适应。8个源种子上最终新布局 Q 的 live−silent 为 full **+20.60 pp**（95% t(7) CI [+14.55,+26.65]）、action-only **+13.58 pp**（[+7.14,+20.01]）、sender-only **+3.11 pp**（[−5.55,+11.77]）；full−action-only为+7.02 pp，full−sender-only为+17.49 pp。结果把增益定位到新主体 action 适应为主、sender 适应提供额外贡献；执行后正确搭档率仍未改善。独立审计重放384个检查点、64个最终评估和97,044,480次模块前向，最大误差0，A/B全程冻结。当前策略均为本地 NumPy MLP，无 LLM/视觉模型/外部网络调用。

- **最新完成[独立语义转移确认实验](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_semantic_formation_confirmation_study/确认实验_结果与下一步.md)：**四个新种子49301–49304在`unique × PI_live/PI_silent`中从头训练8次；训练审计重放48个checkpoint、16个自然终点，`max_abs_error=0`。在240个有向固定角色案例、36个heldout背景上，PI-live aligned `source_pull`为**+10.2970 pp**，同层错配placebo为**+10.1656 pp**，预注册 aligned−placebo 仅**+0.1314 pp**（t(3)区间[−1.5910,+1.8538]）；PI-silent逐行零。消息移植与placebo独立审计均通过，最大误差0。该确认说明当前策略对消息有一般任务敏感性，但没有稳定的源端需求—消息对齐增量；不能称词义、组合语法或语言起源。
- **最新完成[首窗消息槽位子集探针](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_semantic_formation_confirmation_study/slot_probe_结果与下一步.md)：**在同一确认策略上枚举四槽的16个子集，保存30720条记录、1105920个世界并独立重放3110400次模块前向。PI-live单槽`source_pull`约+0.92–+1.74 pp，两槽+3.59–+6.81 pp，三槽+8.58–+9.46 pp，完整四槽+10.30 pp；PI-silent逐行零。结果只支持分布式整包／槽位交互的窄机制解释，不能称某个槽位为词素或组合语法。

- **最新完成[冻结搭档—新接收者协议传递实验](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_new_receiver_transmission_study/transmission_结果与下一步.md)：**冻结已有群体中的 A/B，替换随机初始化的新 C，比较 static/rematched × live/silent 共32个运行。8个源种子上，新布局团队 Q 的 live−silent 差异为 **+20.60个百分点**（95% t(7) 区间[+14.55,+26.65]），监控 Q 中心化时间AUC **+20.07 pp**；8/8种子为正。增益主要来自物理执行率（static/rematched +57.58/+51.39 pp），执行后目标搭档率反而下降，且新 C live Q（33.42%/29.39%）低于旧 C 源基线（45.14%/44.81%）。独立审计重放192个检查点和32个最终评估、48,522,240次模块前向，最大误差0，A/B参数全程不变。结果支持“跨主体通道促进新主体适应”，不支持“旧协议已完整传递”，更不能称词义、组合语法或人类语言起源；当前源策略和新策略均为本地 NumPy MLP，无 LLM/视觉模型/外部网络调用。

- **最新完成[固定角色定向消息移植探针](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_partner_ecology_study/semantic_transfer_结果与下一步.md)：**在unique生态中筛选120个保持听者参与、但源／目标完整成功行动集合不相交的案例，双向展开为240个源→目标方向；对4个PI种子做首窗源消息移植，4个种子×PI-live/PI-silent共1920行、69120个世界。轴均衡PI-live `source_pull`为**+5.9288个百分点**，描述性t(3)区间[−4.2952,+16.1528]；destination轴+22.8866 pp，kind/length略负，silent逐行零。独立重放最大误差0。结果把“消息有用”推进为“听者动作是否向源端内容移动”，但仍是后验任务敏感性，不是词义、组合语法或语言起源证据。
- **固定角色消息移植的错配 placebo 对照：**在相同发送者×听者×需求轴×方向层内循环错配源消息，aligned−placebo `source_pull`为**−1.2353个百分点**（描述性t(3)区间[−2.4676,−0.0029]），placebo动作改变率更高；独立重放1920行、414720次live前向，最大误差0。该阴性对照说明主探针正效应没有源端需求对齐增量，后续独立训练以aligned−placebo为主读数。完整报告见[错配placebo](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_partner_ecology_study/semantic_transfer_placebo_结果与下一步.md)。

- **最新完成[FI 单槽 payload／符号探针](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_factorized_neutral_altpartner_information_control_study/single_slot_结果与下一步.md)：**在正式FI批次的64个最终策略上，对A/B/C发送者的四个首窗槽位分别做payload清零、`(t+1) mod 8`和`xor 4`，共2,304条记录、129,392,640个世界；FI-live团队Q损失（static/rematched）为payload **0.25/0.22 pp**、符号加一 **0.55/0.47 pp**、XOR4 **0.44/0.52 pp**，四槽均有小幅影响，FI-silent逐行不变。独立重放2,304行和393,569,280次模块前向，`max_abs_error=0`；图表已查看。结果支持分布式跨人整包代码依赖，不能称词义、组合语法或语言起源。

- **最新完成[FI 多符号重编码与字段遮蔽探针](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_factorized_neutral_altpartner_information_control_study/recode_mask_结果与下一步.md)：**在同一 FI 正式批次的64个最终策略上，对A/B/C单个发送者的第一窗跨人包做六个固定符号双射、两种槽位重排、payload全置零和visibility全置零，共1,920行、107,827,200个世界；960条live干预由本地前向重算，960条FI-silent为解析上不变的对照。live的自然−干预团队Q损失（static/rematched）为符号双射 **3.81/4.21 pp**、槽位重排 **3.31/4.00 pp**、payload遮蔽 **2.78/3.47 pp**，visibility遮蔽仅 **0.05/0.02 pp**；物理执行损失同步出现，执行后的目标搭档率没有正向改善。独立审计1,920行、328,872,960次模块前向，`max_abs_error=0`；三张PNG/PDF已查看。该结果支持跨人整包的token和槽位依赖，不能称词义、组合语法或语言起源。

- **已完成（上一轮）[FI 消息局部性与重编码探针](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_factorized_neutral_altpartner_message_locality_probe/结果与下一步.md)：**在正式 FI 批次的64个最终策略上做512条只读路由干预，覆盖28,753,920个新布局世界。FI-live 的自然−干预团队Q损失为：关闭跨人通道static/rematched **14.83/15.14 pp**，固定符号双射 **16.63/17.25 pp**，四槽重排 **11.72–12.71 pp**；单发送者遮蔽约3.27–4.91 pp。FI-silent的符号、位置和发送者遮蔽均为零，说明效应来自跨主体路由；但执行后的目标搭档率没有同步下降，主要变化仍是物理执行和提案合法性。独立逐模式审计512次重放、258,785,280次模块前向，`max_abs_error=0`。这是后验机制证据，不是词义、组合语法或语言起源结论；下一步在新独立群体中预注册局部替换和跨需求新组合。

- **已完成[全信息观察与通信通道正式对照](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_factorized_neutral_altpartner_information_control_study/结果与下一步.md)：**16个独立种子66701–66716在同一两个可替代搭档生态中训练`static/rematched × FI_live/FI_silent`四格，共64次训练、384个checkpoint。预注册的执行后目标搭档率重配×通信交互中心化时间AUC为 **−0.0079 pp**，近似t15区间 **[−0.1756,+0.1599]**，6/16种子为正；新布局终点+0.0029 pp。次量物理执行+0.1817 pp、团队Q+0.4172 pp、`Q|physical`+0.3437 pp、提案合法率+0.3968 pp，区间均跨零。独立审计重放384个checkpoint、64,696,320个世界和582,266,880次模块前向，最大误差0；JSON聚合和三张图均通过。FI仍主要提高执行／提案合法率，没有稳定提高执行后的搭档选择；与已审计PL同种子对照只作描述性比较。对同一FI live策略做闭通道后验干预后，Q下降14.83/15.14 pp、物理执行下降25.46/25.64 pp（static/rematched），而FI silent策略下降均不到2 pp；目标搭档率仍无正向变化。该结果收窄信息权限解释，但不是词义、组合语法或语言起源证据；下一步做局部重编码与逐字段遮蔽。

- **最新完成[复合需求搭档方向探针](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_factorized_neutral_altpartner_partner_direction_probe/结果与下一步.md)：**对 `altpair_001` 的64个已审计最终策略枚举3,456个同时改变两个语义轴且改变合法搭档集合的需求邻居，乘36个背景共124,416个方向案例/策略，实时替换一个发送者W1包并重算W2与行动。预先规定的搭档边重配×通信交互 **−0.0000465 pp**，近似t15区间 **[−0.0001628,+0.0000698]**，9/16种子为正但绝对量近零；完整计划转移次量−1.7498 pp（[−4.0162,+0.5166]）。独立审计重放64条策略、7,962,624案例行和56,383,488次模块前向，最大误差0；复合搭档集合、192个sham和图表检查均通过。结果显示整包替换改变了部分联合行动质量，却没有稳定改变搭档边，不能称词义、组合语法或语言起源；下一轮在独立训练中预注册搭档／对象／长度／目的地／neutral响应和信息权限对照。

- **已完成[因子化中性行动的单发送者方向探针](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_factorized_neutral_altpartner_direction_probe/结果与下一步.md)：**对 `altpair_001` 的64个已审计最终策略做76,032个同背景单轴需求近邻案例/策略，实时替换一个发送者的W1包并重算W2与行动。静态训练方向转移+22.5837 pp，随机重配+19.9108 pp；预注册重配×通信交互 **−2.6729 pp**，近似t15区间 **[−5.8422,+0.4964]**，5/16种子为正。搭档边转移恒为0，因为单轴近邻保持合法搭档集合不变；后续复合需求近邻探针已完成且结果见上条。独立审计重放64条策略、4,866,048案例行和47,093,760次模块前向，最大误差0；修复前失败冻结记录保留。结果是后验任务机制证据，不能称词义、组合语法或语言起源。

- **最新完成[两个可替代搭档 × 显式中性动作确认实验](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_factorized_neutral_altpartner_study/结果与下一步.md)：**16个新种子66701–66716比较static/rematched × live/silent四格，共64次训练、384个checkpoint。每个世界有两个使用不同搭档对的合法方案；行动策略显式分为neutral/engage和16类proposal。预先规定的目标搭档率（物理执行后）的重配×通信交互中心化时间AUC为 **+0.0140个百分点**，近似t15区间 **[−0.4586,+0.4867]**，终点−0.0805，6/16种子为正。训练布局末点live把物理执行率从约37%提高到100%，但目标搭档率从约85%降到67%；随机重配没有增强这一内容选择交互。独立审计重放384个checkpoint、64,696,320个世界，582,266,880次模块前向，最大误差0；图表已通过视觉检查。结果支持把执行与搭档／方案内容分开的任务性结论，不能称词义、组合语法或语言起源；下一步做因子化行动上的单发送者方向干预、full-information/closed-channel与消息重编码对照。

- **最新完成[显式中性／取消动作 pilot](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_multi_legal_neutral_action_study/结果与下一步.md)：**4个新种子66701–66704比较`static/rematched × cancel{0.00,0.10} × live/silent`八格、32次训练。主量为取消收益对通信—条件`Q`作用的差中差中心化时间AUC **+0.1448个百分点**，近似t3区间 **[−0.4516,+0.7412]**，终点−0.1240个百分点；全员`cancel`终点率为0，通信仍主要提高物理执行，未稳定提高`Q|physical`。独立审计重放192个checkpoint、31,781,376个世界，最大误差0。该结果只否证低收益、三人同时选择的取消协议实现；下一步采用独立neutral/engage意图头与两个可替代搭档的合法方案。

- **最新完成[多合法行动 × 随机重配方向性消息探针](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_multi_legal_rematched_direction_probe/结果与下一步.md)：**对上述64个最终策略做单发送者首窗替换；供体与接收世界保持同一布局和站点所有者，只改变该发送者需求。固定训练的live−silent方向性方案转移为 **+3.6725个百分点**（近似t15区间[+3.2534,+4.0915]），随机重配为 **+3.3104个百分点**（[+3.0705,+3.5504]）；预注册重配×通信差中差 **−0.3620个百分点**（[−0.8317,+0.1076]），6/16种子为正。独立重放64条策略、9,704,448个案例，最大误差0；192个sham全部逐项相等。结果支持首窗包对方案分布有方向性影响，但不支持随机重配增强内容效应，仍不能称词义、组合语法或语言起源；下一步进入带显式中性动作的独立训练。

- **最新完成[多合法行动 × 随机重配确认实验](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_multi_legal_rematched_confirmation_study/结果与下一步.md)：**16个新种子66501–66516比较`static/rematched × partial_multi_strict_PL × live/silent`四格，共64次训练、384个checkpoint。预先规定的重配×通信主量是条件`Q|physical`（已发生严格物理执行后选中任一合法完整方案）的差中差中心化时间AUC，均值 **−0.3400个百分点**，近似t15区间 **[−1.3150,+0.6349]**，终点 **+0.0537个百分点**，9/16种子为正。训练布局末点物理执行率约36.6%/98.8%（静默/实时），条件Q约53.0%/28.4%，未条件化团队Q约19.4%/28.0%；通信提升了执行但没有显示稳定的执行后方案选择增益。独立审计重放384个checkpoint、60,217,344个世界，最大误差0，配对流和49,152,000个重配分配通过。该结果不支持随机重配增强通信方案选择，仍不能称词义、组合语法或语言起源；下一步加入显式中性动作并在重配条件下预注册方向干预。

- **最新完成[多合法行动定向首窗探针](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_multi_legal_direction_probe/结果与下一步.md)：**对多合法确认批次的32个最终策略（16个种子×live/silent）在6个新布局×6个私有站点所有者上做单发送者W1替换，供体只改变该发送者的个人需求。主量为计划概率从接收者独有方案向供体独有方案的带符号转移，live−silent均值 **+3.5153个百分点**（近似t15区间[+3.2096,+3.8209]），16/16种子为正；A/B/C分别+3.8832/+4.3651/+2.2974，搭档转移接近0。live自然→干预物理执行83.4911%→63.7447%，`Q|physical`26.5430%→27.6248%，silent严格不变。它是后验方向性机制探针，仍不能称词义、组合语法或语言起源。首版错误silent路由已移入`invalid_001`，修正版新增29,187,072模块前向、sham与独立重放审计均通过；完整方法、数据和图表见报告。下一步把执行／选择分离写入新的独立训练，加入显式中性动作、随机重配搭档和方向干预。

- **最新完成[多合法行动协调确认实验](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_multi_legal_coordination_study/结果与下一步.md)：**16个新种子66401–66416构造每世界两个等价 full-success 计划（同一搭档、不同地点），完成 `partial_multi × strict × live/silent` 32次训练、192个checkpoint。团队Q的live−silent中心化时间AUC为 **+7.2955个百分点**（近似t15区间[+6.9283,+7.6628]），16/16种子为正；物理执行率AUC为+53.6829。训练布局Q终点19.5475%→27.7695%，新布局15.2080%→22.4030%。然而后验 `Q | any physical execution` 为 **−20.8369个百分点**（[−29.0836,−12.5902]），终点−25.2566，说明未条件化Q增益主要由执行增加伴随，不能称公共词义、多均衡语言约定或语言起源证据。执行末端tuple-key序列化失败发生在全部训练完成后，JSON-only恢复和独立checkpoint审计通过；完整设计、原始运行树、诊断与图表见报告。下一步拆分执行／选择结果，加入显式中性动作、随机重配搭档和方向可识别消息干预。

- **最新完成[随机重配搭档确认实验](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_rematched_partner_confirmation_study/结果与下一步.md)：**16个新种子66301–66316比较`static/rematched × partial_strict_PL × live/silent`四格共64次训练、384个checkpoint；训练留出AB/AC/BC一个完整需求定义搭档，rematched臂在每个训练世界随机重分配私人需求到物理主体。预先规定的重配×通信主量为未见搭档主体精确行动率中心化时间AUC **−4.8676个百分点**（近似t15区间[−17.8716,+8.1363]），终点−6.7243个百分点，7/16种子为正；重配提高静默基线约+4.5249个百分点AUC，却没有提高live条件（−0.3427个百分点）。独立checkpoint审计通过、最大误差0；该结果不支持随机重配自动产生可迁移社会约定，不能称词义、组合语法或语言起源。下一步解除唯一动作／稀疏奖励，加入多合法行动和方向可识别接收指标。

- **最新完成[搭档组合留出确认实验](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_partner_holdout_confirmation_study/结果与下一步.md)：**16个新配对初始化66101–66116轮换留出AB/AC/BC一个完整搭档关系，完成 `partial/all_or_nothing × strict/reciprocal × live/silent` 八格共128次训练、768个checkpoint。预先规定的未见搭档目标主体精确行动率交互中心化时间AUC为 **+0.7922个百分点**（近似t15区间[−3.9510,+5.5354]），终点−0.5130个百分点，10/16种子为正；团队Q次量AUC为−9.6378个百分点。已见搭档新布局在通信下可协调，而未见搭档在all-or-nothing四个条件均为0→0，支持固定社会关系局部最优边界，不能称词义、组合语法或语言起源。最终汇总器因种子间有意轮换分区触发`Worker arrays differ`，JSON-only恢复和独立审计通过，失败记录保留；下一步改为随机重配搭档、多合法行动和方向可识别接收指标。

最新完成[双向跨分区语义边确认实验](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_cross_split_confirmation_study/结果与下一步.md)：16个新配对初始化65101–65116完成 `partial/all_or_nothing × strict/reciprocal × live/silent` 八格共128次训练，768个checkpoint和87,588,864个紧凑评价世界经独立审计重放。预先冻结的跨分区 Q 中心化时间AUC为 **+21.6285个百分点**（近似t15区间[+18.9209,+24.3362]），16/16种子为正；正确执行搭档率次量为 **+32.3462个百分点**（[+29.0738,+35.6185]）。Q是两端成功的AND，因此两个方向的Q数学上相同，不能计作两条独立证据；方向差异只在搭档次量中出现。训练完成后的汇总字段错误通过JSON-only恢复，失败记录和独立审计均保留。该结果支持有训练锚点时的跨分区行为响应，仍不能称词义、组合语法或语言起源；下一步加入新搭档、多合法行动和方向可识别的接收指标。

最新完成[收益生态×因素语义留出确认实验](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_factorial_payoff_study/结果与下一步.md)：16个新配对初始化64101–64116在`partial/all_or_nothing × strict/reciprocal × live/silent`八格完成128次训练，每次6,000步，1,152个评价文件、326,356,992个评价世界；独立审计通过。预先规定的“双端点都未见”对象×属性 Q 交互为 **0.0000个百分点**（近似t15区间[0,0]），但正确搭档协调的时间AUC为 **+22.4820个百分点**（[+20.3754,+24.5886]）。看到主量为零后追加的seen→heldout跨分区只读probe为探索性分析：Q端点交互 **+25.9319个百分点**（[+22.4101,+29.4536]），16/16种子为正；这不能改写为确认性主结果，也不能称词义、组合语法或语言起源。完整设计、审计、图表和下一项双向跨分区确认方案见[结果与下一步](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_factorial_payoff_study/结果与下一步.md)。

最新完成[因素组合留出形成实验（校正结果）](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_factorial_formation_study/结果与下一步.md)：16个配对初始化在 factorial/saturated × strict/reciprocal × live/silent 四格完成128次、每次6,000步训练，并保存1,152个评价文件。factorial臂排除wood-long与fiber-short但保留各单因素边际；校正后的预注册未见资源通信交互为 **−1.1221个百分点**，近似t15区间 **[−1.9474,−0.2968]**。这表明通信增益依赖训练覆盖，未见对象×属性组合没有自动获得可迁移响应。原后处理广播错误已用保存数组做JSON/NPZ-only校正，无重训；完整审计重放到最后scope断言后由独立范围恢复确认，详情见[校正运行树](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_factorial_formation_study/results/factorial_001_corrected/)和[审计范围恢复](/Users/xia/Documents/ChatGPT/语言/audit_corrected_scope_001/verification.json)。该结果支持区分固定任务协调码与可组合语义的负边界，仍不能称词义、组合语法或语言起源；下一步转向可替代搭档和多合法行动的生态操纵。

最新完成[六个随机符号双射集成对照](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_random_symbol_ensemble_study/结果与下一步.md)：在同一64条冻结策略上预先固定六个8符号随机双射，只替换指定发送者首窗W1包并实时重算W2／行动。覆盖2304条策略—时间记录、1152个live文件和1152个静默别名，独立逐条重放通过；末点六置换按种子平均的自然M减置换M为+11.7265个百分点，近似95%区间[+8.6680,+14.7849]，16/16区组为正。该结果支持固定整包编码敏感性并非单一置换孤例，但不证明词义、组合性或语言起源；下一步仍需保持等式模式的局部替换、未见组合泛化和新的独立形成群体。

随后完成[局部包结构对照](/Users/xia/Documents/ChatGPT/语言/research_program/triadic_local_structure_control_study/结果与下一步.md)：单槽替换、秩规范化和只保留等式模式的重标记共1152条记录，经独立逐条重放通过。末点三控制按种子平均的自然M减控制M为+8.4156个百分点，近似95%区间[+6.4489,+10.3823]；等式模式重标记+11.9877、秩规范化+10.5429、单槽循环+2.7163，后两种等式模式控制16/16区组为正。结果支持效应超出重复模式本身，仍不等于词义或语言形成。

已完成v0.29相同曝光下的共现支持对照：同一私人all来源，24次新通信训练；两臂各18训练布局，每步每资源位置恰42曝光，整体相关信息量同为1bit。共同P6自然J为14.236%/17.130%，路径3−路径2主差−2.894个百分点，四来源两正两负，AUC也两正两负；训练布局J为98.148%/96.952%。主预测未获一致支持。两预定方向的平均人工片段重组都超出199整码重编码参照，但来源异质、架构偏置未分离，不能等同自主组合语言。下一步优先解除共同P6一一匹配的识别限制，先设计多对应的共同目标及单资源表达上界，不继续扫图追正差。见[完整报告](/Users/xia/Documents/ChatGPT/语言/redesign_v0.29/results/support_001/训练共现结构与共同符号迁移研究报告.md)、[实际消息](/Users/xia/Documents/ChatGPT/语言/redesign_v0.29/results/support_001/实际消息示例.md)和[复现索引](/Users/xia/Documents/ChatGPT/语言/redesign_v0.29/README.md)。独立材料确认与论文中心贡献仍未齐备。

已完成v0.28新材料上的完整形成开发复现：12图固定8训练/4测试，48私人拟合、36新通信训练。私人目标12 J由63.657%升至100%；A/B通信目标12 J为6.308%/11.632%，主差+5.324个百分点，四新初始化来源三正一零；C为94.792%，目标布局已在通信训练。AUC主差+3.975个百分点四正。独立原始统计和全部终点前向复算通过。9水果已有开发暴露、单张测试水不构成多份独立材料，正式确认及论文中心创新仍未完成；不继续堆叠同类小图库复现。见[完整报告](/Users/xia/Documents/ChatGPT/语言/redesign_v0.28/results/formation_001/新材料上的私人经验与共同通信形成研究报告.md)、[消息实例](/Users/xia/Documents/ChatGPT/语言/redesign_v0.28/results/formation_001/实际消息示例.md)及[复现索引](/Users/xia/Documents/ChatGPT/语言/redesign_v0.28/README.md)。

本轮完成新水来源的有限可行性实验：8项来源支持，固定前4张下载，3张通过双AI视觉和技术验收，1张因无法可靠排除空杯而排除；没有新模型调用。新增3张水图单独记账，原v1确认48及后4候选未看像素。下一步以9张既有水果流程图与3张新水，固定8训练/4测试，完整重跑A/B/C形成流程，明确作为开发复现；正式独立确认仍未完成。见[本轮报告](/Users/xia/Documents/ChatGPT/语言/paper_program/water_source_followup_001/独立水图片可行性实验报告.md)及[逐图结果](/Users/xia/Documents/ChatGPT/语言/paper_program/water_source_followup_001/curation_result.json)。

此前完成v0.27已有A/B/C协议的新图片通信检验：复用v26的11张流程图表征，72方向、288表、336960个发送世界，无新训练或DINO推理。B（私人30/通信18）的目标12布局自然J由15.896%变为17.149%，主材料差+1.253个百分点，四来源均小正；C（私人30/通信30）为95.833%，其目标布局已通信训练。新图B−A均差+5.845个百分点，但一来源反向；B目标12消息TV为0.2223，训练过的18布局仅0.0011。私人能力成功与低自然表达的差距在这批材料上仍存在，未获得新组合语言或独立形成确认。该11图冻结探针分支现结束，下一工作为独立训练/测试材料准备及完整A/B/C形成过程检验，不能用更多同图测量代替。 见[完整报告](redesign_v0.27/results/transfer_001/私人行动与共同通信在新图片上的迁移研究报告.md)及[复现索引](redesign_v0.27/README.md)。

此前完成v0.26冻结私人主体的新图片能力探针：首次用当前11张有限流程可用图片进行实验模型推理，未新增训练。私人all在目标12布局上换食物、换水或同时换图后均保持100%双目标成功；8项来源×mask能力门槛全部通过。私人old在全部换图后为69.879%（旧图67.220%）。四个既有来源、48私人终点，共224,640场景前向；旧图编码与原行为对照通过。这说明这批小样本材料上的私人能力可以保持，没有复现共同语言形成，确认48图仍未访问。该冻结通信检查现已在v27完成；独立训练/测试材料上的形成过程确认仍待执行。 见[研究报告](redesign_v0.26/results/material_001/冻结私人主体的新图片能力研究报告.md)和[复现索引](redesign_v0.26/README.md)。

此前完成v0.25资源角色分别读入诊断：新增12次joint双主体训练，复用v24 mean/attention作历史参照。joint与mean初始49码概率/贪心完全相同，允许两资源读取权重随后分别学习。主old18 J由76.157%升至83.333%，差+7.176个百分点，四来源均正；old归一AUC差+19.898个百分点也四正。new12仍仅2.083%（mean1.042%，AT4.514%），来源两正一负一零。水／食物方向的人工重组由1.080%升至5.633%，四来源均正；食物／水方向2.045%升至2.932%，两正两负，不能择优称全面结构增强。三臂全程裁剪系数均为1；有效参数与读入优化路径仍有差异。见[研究报告](redesign_v0.25/results/joint_001/资源角色分别读入与共同通信学习研究报告.md)与[复现索引](redesign_v0.25/README.md)。熟悉协议学习得到改善，表达迁移未解决；该联合读入候选已完成，接下来收束此接口分支、统一已验证基线与信息权限，再为一个可反驳迁移问题准备独立材料确认，不继续扩大同图库超参数网格。

此前完成v0.24私人预测上的Ri式mean/attention任务内基线：24次新双主体训练，四继承来源。社会未见12组合J为1.042%/4.514%，attention−mean +3.472个百分点，三正一零；old18为76.157%/62.963%，共同30为46.111%/39.583%，不支持整体通信改善。两种预定符号槽位分配的重组结果方向不同，绝对成功率约1%–5%，不能择优宣称结构增强。私人行动已100%只限既有有限测试；多key保留与动态读取共变，额外角色预测并非纯视觉输入。所有训练和独立复核完成，见[研究报告](redesign_v0.24/results/attention_001/私人预测动态读取与共同符号结构研究报告.md)与[复现索引](redesign_v0.24/README.md)。本轮未获得稳定可迁移的共同符号结构，停止扩大该注意力超参数；当时的联合读入候选随后已在v0.25固定并完成；独立图片确认及中心新颖性仍未齐备。

此前完成[v0.23私人状态经验与共同符号表达迁移](redesign_v0.23/results/experience_001/私人状态经验与共同符号表达迁移研究报告.md)：四个新来源、48私人拟合、36社会训练。私人目标12图双成功由67.220%升到100%；随后都仅在old18训练通信时，目标12自然J由10.775%升到15.896%，主差+5.122个百分点，四来源均正。全30也进入通信训练后为95.627%，该条件不是泛化。AUC一来源反向；事后可见/历史资源分项分别87.109%→62.413%、21.224%→49.403%，存在取舍，不能说组合结构已改善。完整执行及独立复核通过，见[v0.23复现索引](redesign_v0.23/README.md)。当时的结构基线候选现已在v0.24执行；独立图片确认及论文中心贡献仍未齐备。

此前完成[v0.22保持布局与可见对象互换诊断](redesign_v0.22/results/visibility_001/布局保持与末帧可见对象的固定协议研究报告.md)：冻结24个私人头、48个既有通信方向，无新训练。社会主可见优势D为32.687个百分点，四来源都出现资源优势交叉；但old18的D仅2.344个百分点、双目标J为96.191%，新初态/布局12图的D为78.201个百分点、J为6.152%。私人头在这12图也仅45.627%，不能把失败归为通信独有或普遍遗忘。全部480世界完整前向重放与独立统计通过，见[v0.22复现索引](redesign_v0.22/README.md)。当时的私人/通信经验范围比较现已在v0.23完成；独立确认与论文中心贡献仍未齐备。

此前完成[v0.21资源可见性与共同通信实验](redesign_v0.21/results/communication_001/资源可见性与共同通信形成研究报告.md)：四来源、24次新社会训练。完整可见/局部＋历史的共同30自然双目标J为65.23%/63.07%，主差局部减完整−2.16个百分点，三负一正；熟悉布局约98%/97%，未训练12图只有16.08%/11.94%。局部协议已能传递历史信息：静止目标合法历史配对54.50%，条件置换使静止正确率74.68%→19.95%。资源分项出现移动更好、静止更差，但私人阶段和固定协议切换观察也有该方向，不能说通信创造了偏向。见[v0.21复现索引](redesign_v0.21/README.md)。当时的固定协议诊断现已在v0.22完成；目前仍缺独立确认和论文中心贡献。

此前完成[v0.20资源移动与历史利用实验](redesign_v0.20/results/temporal_001/资源移动后的私人记忆与更新研究报告.md)：四个新来源、48次私人训练。先看完整场景，再移动一个资源并只显示其新位置；完整反传/截断梯度的共同30组合延迟双成功为92.39%/91.48%，主差+0.91个百分点，四来源均正。清零历史后两组约20%，历史配对均为0，支持两者都利用历史。预定old能力差仅+0.047个百分点、未达10个百分点，整批验收失败；截断组仍有前向记忆，不能作为无记忆组或直接进入该能力社会对照。没有新共同通信。见[v0.20复现索引](redesign_v0.20/README.md)与[近邻及因果边界](redesign_v0.20/近邻与因果边界.md)。当时的环境信息压力候选现已在v0.21完成，完整论文目标仍在进行。

此前完成[v0.19新私人行动头实验](redesign_v0.19/results/readout_001/冻结视觉接口的新私人行动学习研究报告.md)：两种冻结视觉接口 × 奖励/监督两信号，共96头，来自四个既有来源。仅行动成败奖励下，未训练组合贪心双成功为保留97.62%、重置并匹配幅度96.44%；主差+1.18个百分点，四来源均正。保留接口第100步优势17.44个百分点，过程AUC优势2.58个百分点；重置端仍能学会该私人任务。随机行动Q的排序相反，不能混称整体能力提升。本轮限制了将旧头失配或通信低分解释为缺乏基本资源知识的说法，没有训练新通信。见[v0.19复现索引](redesign_v0.19/README.md)和[最新近邻文献](paper_program/literature_capability_20260916/短近邻定位.md)；重训读出本身已有前例，当时提出的事件保持与更新任务现已在v0.20完成，其能力差验收未通过。

此前完成[v0.18固定策略梯度方差诊断](redesign_v0.18/results/gradient_001/固定策略下的通信学习梯度方差研究报告.md)：72个既有策略检查点、1,296世界、63,504消息梯度，无新增训练。第0、100、600步匹配基线相对同策略解析置换的方差降幅为0.63%、0.22%、62.12%；600步四来源均为正，早期各一来源反向。该局部奖励梯度方差结果不等于完整训练梯度或组合能力，也未识别v0.17未见组合终点的中介作用。完整执行、独立复算和图表核验通过，见[v0.18复现索引](redesign_v0.18/README.md)。同期完成[全部79个水候选的元数据复核](paper_program/visual_confirmation_v2_water_review/元数据复核报告.md)，发现短语误配及作者解析局限；没有新增可用图像或模型确认。

此前完成[v0.17发送基线与场景对应实验](redesign_v0.17/results/baseline_001/发送基线与场景对应关系的通信形成研究报告.md)：12个新运行、12个复用参考。未训练组合成功率为匹配13.63%、置换14.39%，唯一主差匹配减置换−0.76个百分点，四来源两正两负。约99.60%的基线值实际改变，两臂全程均无非单位裁剪；本批未支持场景匹配带来稳定收益，不能证明等效或改称全局基线已被证实。熟悉组合为92.44%/93.50%。执行、原始复算与报告读审后封存，见[v0.17复现索引](redesign_v0.17/README.md)。该后续诊断现已在v0.18完成，不能反向改写本批主结果。

此前完成[v0.16策略与价值输入幅度实验](redesign_v0.16/results/branches_001/策略与价值输入幅度的通信形成研究报告.md)：24次新社会学习与24次既有参考组成2×2。未训练组合自然成功率为不缩放4.44%、仅策略6.90%、仅价值7.97%、两者13.63%；主要单支路差−1.08个百分点，四来源方向不一致。辅助比较中，策略已缩放时再缩放价值提高6.73个百分点，四来源均提高。事后全日志核查发现四组裁剪全程未激活，价值影响的计算路径收窄到策略优势基线及后续互动。执行、独立统计和日志核验通过，见[v0.16复现索引](redesign_v0.16/README.md)。发送批次内基线置换现已在v0.17完成；当前仍为旧图片、四个继承来源的开发证据。

此前完成[v0.15视觉编码幅度实验](redesign_v0.15/results/scaled_001/视觉编码幅度与共同符号形成研究报告.md)：新增12次社会学习，复用保留/重置各12个参考。未训练组合成功率为保留22.04%、重置4.44%、固定幅度补偿13.63%；补偿相对重置提高9.18个百分点，四来源均改善，保留减补偿仍差8.41个百分点（三正一负）。幅度影响能恢复部分成绩，不能将私人准备收益全部归于世界知识或组合能力。熟悉组合也86.53%→92.44%，保留为96.62%。完整执行和独立复算通过，见[v0.15复现索引](redesign_v0.15/README.md)。策略/价值支路拆分现已在v0.16完成；独立图片确认仍未完成。

独立图片的[第三批流程验收](paper_program/visual_confirmation_v1/pixel_workflow_003/README.md)完成17张原图下载及匿名评审，净增4张有限流程可用图；三批累计11/24（苹果、香蕉、橙子各3，水2），仍缺13张，另3项待核未计通过。确认48图未访问；11张流程图现已进入v0.26私人能力探针，不再是未暴露材料。后续已完成[新水来源元数据研究](paper_program/visual_confirmation_v2_water/元数据可行性报告.md)：33次请求、300详情得到79个自动暂合格关联簇，仍有泛类别和跨语言词表漏检；这些不是可用图片，无新像素或模型调用。原[可行性审查](paper_program/visual_confirmation_v1/pixel_workflow_003/后续图库可行性审查.md)保留。

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

最新[全局奖励尺度零假设对照](research_program/triadic_global_reward_control_study/全局奖励尺度对照_结果与下一步.md)：192 次配对训练、独立审计和解析梯度控制通过。PL 中全局 reward×0.25 与原生回报的团队 Q、物理执行和执行后目标搭档差均接近零；第三人冲突分支的效应仍表现为执行增加而执行后搭档选择下降。这是任务／优化控制，不能称语言形成。

