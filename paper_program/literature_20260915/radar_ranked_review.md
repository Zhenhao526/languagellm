# arXiv Watcher Agent Review

- Generated: 2026-09-15T21:10:51
- Field config: `radar.toml`
- Window: last 24000 hours
- Candidates after topic filtering: 8
- Full/abstract reviews available: 8
- Ranking: content-based reading priority; author history contributes only 5%

> Scores predict reading priority, not objective scientific importance. Confidence and concerns should be read with the score.

## 1. Emergent Communication: Generalization and Overfitting in Lewis Games

- arXiv ID: `2209.15342`
- Link: https://arxiv.org/abs/2209.15342
- PDF: https://arxiv.org/pdf/2209.15342v2
- Published: 2022-09-30
- Authors: Rita, Mathieu, Tallec, Corentin, Michel, Paul, Grill, Jean-Bastien, Pietquin, Olivier, Dupoux, Emmanuel, Strub, Florian
- Topics: 
- Reading priority: **87.9**
- Verdict / confidence: `must-read` / `high`
- Review basis: `full_text`
- Scores: relevance 100.0, evidence 90.0, novelty 85.0, impact 91.0, reproducibility 85.0
- Contribution: 把 Lewis 游戏目标分成信息损失与共同适应损失，并诊断听者过拟合如何影响说者形成的协议。
- Why read: 发收负担与适应研究必须直接比较。; 用充分训练的新听者区分编码歧义与现有听者限制。
- Concerns: 主泛化指标使用重新训练的探针听者，并非原配对自然成功。; 视觉任务中泛化提高并未同步提高 TopSim，不能泛化成组合性定律。

## 2. CtD: Composition through Decomposition in Emergent Communication

- arXiv ID: `2601.10169`
- Link: https://arxiv.org/abs/2601.10169
- PDF: https://arxiv.org/pdf/2601.10169v1
- Published: 2026-01-15
- Authors: Carmeli, Boaz, Meir, Ron, Belinkov, Yonatan
- Topics: 
- Reading priority: **85.9**
- Verdict / confidence: `must-read` / `high`
- Review basis: `full_text`
- Scores: relevance 98.0, evidence 86.0, novelty 84.0, impact 88.0, reproducibility 85.0
- Contribution: CtD 先利用多目标样本学习概念代码本，再组合代码表达新图像；部分条件无需第二阶段训练即可泛化。
- Why read: 最直接的课程/概念分解结构阳性方法。; 可移植为明确带额外监督的结构能力上界。
- Concerns: 方法含研究者构造的概念组与已知消息长度。; 不能把 2026 arXiv 上传日期写成首次发表年份；正式为 ICLR 2025。

## 3. The Curious Case of Representational Alignment: Unravelling Visio-Linguistic Tasks in Emergent Communication

- arXiv ID: `2407.17960`
- Link: https://arxiv.org/abs/2407.17960
- PDF: https://arxiv.org/pdf/2407.17960v1
- Published: 2024-07-25
- Authors: Kouwenhoven, Tom, Peeperkorn, Max, van Dijk, Bram, Verhoef, Tessa
- Topics: 
- Reading priority: **83.7**
- Verdict / confidence: `must-read` / `high`
- Review basis: `full_text`
- Scores: relevance 95.0, evidence 85.0, novelty 78.0, impact 87.0, reproducibility 84.0
- Contribution: 揭示视觉通信中表征对齐与 TopSim 的关系，加入对齐惩罚后指标改善却未改善严格视觉组合区分。
- Why read: 需要将功能迁移与统计结构分开。; 随机噪声/新照片/严格关系测试可检验视觉捷径。
- Concerns: 使用私有表征间可微信息的正则，不能直接作为完全独立训练。; Winoground 二选一是组合性代理，失败不定位唯一机制。

## 4. Emergent Communication between Heterogeneous Visual Agents through Decentralized Learning

- arXiv ID: `2605.11695`
- Link: https://arxiv.org/abs/2605.11695
- PDF: https://arxiv.org/pdf/2605.11695v1
- Published: 2026-05-12
- Authors: Ochiai, Mikako, Nagano, Masatoshi, Taniguchi, Tadahiro
- Topics: 
- Reading priority: **83.0**
- Verdict / confidence: `must-read` / `medium`
- Review basis: `full_text`
- Scores: relevance 99.0, evidence 80.0, novelty 79.0, impact 86.0, reproducibility 76.0
- Contribution: 用本地 MH 式接受与私人冻结 DINO/MAE 感知，从随机符号模块形成共享离散序列，并考察感知异质性。
- Why read: 最直接排除“首次非语言预训练视觉主体产生约定”。; 接受率匹配阴性控制与独立视觉测量空间值得学习。
- Concerns: 按 arXiv v1，未核实同行评审发表。; 仅三视觉配对、单数据集、三种子；精度与覆盖解释未完全区分。; MH 为近似的一步过滤，作者不声称精确后验采样。

## 5. Learning Translations: Emergent Communication Pretraining for Cooperative Language Acquisition

- arXiv ID: `2402.16247`
- Link: https://arxiv.org/abs/2402.16247
- PDF: https://arxiv.org/pdf/2402.16247v1
- Published: 2024-02-26
- Authors: Cope, Dylan, McBurney, Peter
- Topics: 
- Reading priority: **79.6**
- Verdict / confidence: `read` / `high`
- Review basis: `full_text`
- Scores: relevance 88.0, evidence 82.0, novelty 78.0, impact 82.0, reproducibility 78.0
- Contribution: 提出 CLAP 任务，以社区交互记录训练新人，并比较行为克隆与预先掌握环境技能后的双向消息翻译。
- Why read: 后续新人/旧协议适应应比较直接克隆与翻译适配器。; 揭示先掌握环境技能、再适应通信的分拆已有先例。
- Concerns: 监督示范与集中训练不可冒充独立奖励学习。; 大量数据和无 pit 时克隆可优于翻译；人类例子不等于人类语言形成实验。

## 6. From Grunts to Lexicons: Emergent Language from Cooperative Foraging

- arXiv ID: `2505.12872`
- Link: https://arxiv.org/abs/2505.12872
- PDF: https://arxiv.org/pdf/2505.12872v2
- Published: 2025-05-19
- Authors: Piriyajitakonkij, Maytus, Charakorn, Rujikorn, Tao, Weicheng, Pan, Wei, Sun, Mingfei, Tan, Cheston, Zhang, Mengmi
- Topics: 
- Reading priority: **78.3**
- Verdict / confidence: `must-read` / `medium`
- Review basis: `full_text`
- Scores: relevance 95.0, evidence 74.0, novelty 73.0, impact 81.0, reproducibility 69.0
- Contribution: 在局部观察的合作采集网格中，以独立 PPO 联合学习行动和离散交流，研究群体、自我互动、时空信息与行为通信。
- Why read: 用户最初原始采集设想已有很近的实现。; 后续行动任务需要至少对照其同步合作与隐式行为通道。
- Concerns: 按 arXiv v2；OpenReview 状态页遇验证，未确认 ICLR 接收。; 信息可解码及 TopSim 不充分证明人类语言式词法或位移语义。; 社会联系实验不能替代新增区分的稳定性—可塑性干预。

## 7. Emergent Compositional Communication for Latent World Properties

- arXiv ID: `2604.03266`
- Link: https://arxiv.org/abs/2604.03266
- PDF: https://arxiv.org/pdf/2604.03266v1
- Published: 2026-03-18
- Authors: Kaszyński, Tomek
- Topics: 
- Reading priority: **78.0**
- Verdict / confidence: `must-read` / `medium`
- Review basis: `full_text`
- Scores: relevance 99.0, evidence 66.0, novelty 77.0, impact 80.0, reproducibility 65.0
- Contribution: 用冻结 DINOv2/V-JEPA 视频特征、离散消息和迭代听者训练研究潜在物理属性的组合通信及下游迁移。
- Why read: 预训练视觉与符号结构最近邻。; 需要对照同容量原子码、接收者重置和实际信息/梯度路径。
- Concerns: 仅核实 arXiv v1，未核实同行评审发表。; §3.2/4.6 的单发送器与多发送器容量计数表述有歧义，不能照引“已完全排除带宽”。; 主要域受控，部分迁移失败；未复现其代码。

## 8. Learning to Communicate Across Modalities: Perceptual Heterogeneity in Multi-Agent Systems

- arXiv ID: `2601.22041`
- Link: https://arxiv.org/abs/2601.22041
- PDF: https://arxiv.org/pdf/2601.22041v1
- Published: 2026-01-29
- Authors: Pitzer, Naomi, Mihai, Daniela
- Topics: 
- Reading priority: **62.4**
- Verdict / confidence: `monitor` / `low`
- Review basis: `abstract_only`
- Scores: relevance 84.0, evidence 40.0, novelty 55.0, impact 63.0, reproducibility 65.0
- Contribution: 研究不同感知模态下多步二进制通信及有限微调后的跨系统互通。
- Why read: 如扩展到伙伴接触或模态变更，需先补读全文。
- Concerns: abstract_only，全文提取失败。; arXiv 自述将发表于 EvoLang XVI，尚未独立核对正式论文集。
