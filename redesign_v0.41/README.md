# redesign_v0.41：三资源双 token 协议形成

本轮在 v0.39 的三资源共同编码任务上增加了每个 sender 的第二个连续 token，并用资源身份置换检验架构槽位不对称。实验研究有限的 grounded 离散通信协议如何从随机通信头中形成，重点观察 token1 是否携带位置相关信息、是否依赖 token0 前缀，以及伙伴覆盖如何影响公共性。

- 正式矩阵：2 个资源分配 × 3 个伙伴拓扑 × 4 seeds × 3 partitions = 72 条独立链
- 每条链：1200 次更新，checkpoint 为 0、100、600、1200
- 世界：3 个资源占据 6 个地点中的有序三元组，共 120 张地图；每个 partition 使用 60 张训练地图和 60 张 target 地图
- 通信：三个资源 sender 各自连续发送两个 7 值 token，消息长度为 6，receiver 名义码空间为 7⁶=117649
- 训练：私有视觉前端继承自 v0.28 并冻结；通信头从随机状态初始化；共享 grounded 奖励
- 控制：canonical 资源顺序 `[0,1,2]` 与 cyclic `[1,2,0]`；固定 A、轮换 A/B、随机 A/B/C 伙伴拓扑

正式报告：[三资源双token协议形成研究报告.md](results/two_token_001/三资源双token协议形成研究报告.md)

正式结果：[two_token_analysis.json](results/two_token_001/two_token_analysis.json)、[two_token_raw_validation.json](results/two_token_001/two_token_raw_validation.json)、[two_token_audit.json](results/two_token_001/two_token_audit.json)、[completion_manifest.json](results/two_token_001/completion_manifest.json)

结果图：[01_two_token_outcomes.png](results/two_token_001/figures/01_two_token_outcomes.png)

源代码：[two_token_train.py](two_token_train.py)、[two_token_analysis.py](two_token_analysis.py)、[two_token_audit.py](two_token_audit.py)、[plot_two_token.py](plot_two_token.py)、[visual_qa.py](visual_qa.py)、[build_two_token_report.py](build_two_token_report.py)

本轮结果支持：冻结的非语言视觉表征、离散序列采样和共同后果反馈足以形成有限的双 token grounded 协议；token1 携带额外的位置相关信息，并对 token0 前缀有可测的条件变化。pair NMI 达到 100% 不能被解释为语法，因为当前只有六个地点，两个 token 可能只是冗余编码。下一轮应遍历全部资源与角色排列，加入 token 删除/交换和新地点干预，再把序列协议接入代际替换实验。

本目录的封存清单标记 `overall_research_goal_complete=false`：它是可复核的开发批次，不代表开放式语言起源问题已经完成。
