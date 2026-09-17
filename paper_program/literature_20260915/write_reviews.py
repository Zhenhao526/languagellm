from pathlib import Path
import json,subprocess
B=Path(__file__).resolve().parent
# Reading priority, not a claim of acceptance probability or intrinsic paper quality.
rows=[
('2209.15342','full_text','high','must-read',[100,90,85,91,85],
'把 Lewis 游戏目标分成信息损失与共同适应损失，并诊断听者过拟合如何影响说者形成的协议。',
'§2.3–3.3 给出分解、冻结说者训练探针听者及听者收敛干预；§4–5 在六种子属性重建及 CelebA/ImageNet 指称任务比较，听者正则有帮助但弱于早停参照。',
'相对 Chaabouni 2020 的泛化与组合性经验关系，提供目标分解和学习干预；我们 C_S 与之相邻，C_R 和旧任务约束前沿是不同问题，不能把多个 oracle 缺口相加当其分解。',
['发收负担与适应研究必须直接比较。','用充分训练的新听者区分编码歧义与现有听者限制。'],
['主泛化指标使用重新训练的探针听者，并非原配对自然成功。','视觉任务中泛化提高并未同步提高 TopSim，不能泛化成组合性定律。'],
'全文核查 §2–5、结论及相关探针定义。必须保留作为 v0.9 的理论最近邻。'),
('2601.10169','full_text','high','must-read',[98,86,84,88,85],
'CtD 先利用多目标样本学习概念代码本，再组合代码表达新图像；部分条件无需第二阶段训练即可泛化。',
'核查 ICLR 2025 正式 PDF 第2–8、34页：五数据集、GS/QT/代码本及无分解对照，组合划分按概念合取留出；附录 J 承认 oracle 目标分组、给定短语长度及传递潜向量。',
'相对 Mu 与 Goodman 2021 的多目标概念交流，加入代码本与分解→组合程序。我们既无其概念分组监督，也未达到其零样本结果；换整数接口不是贡献充分条件。',
['最直接的课程/概念分解结构阳性方法。','可移植为明确带额外监督的结构能力上界。'],
['方法含研究者构造的概念组与已知消息长度。','不能把 2026 arXiv 上传日期写成首次发表年份；正式为 ICLR 2025。'],
'全文方法与核心结果按正式论文核查；arXiv packet 另留存。建议使用 CtD-inspired 阳性对照，而非声称完整复现。'),
('2604.03266','full_text','medium','must-read',[99,66,77,80,65],
'用冻结 DINOv2/V-JEPA 视频特征、离散消息和迭代听者训练研究潜在物理属性的组合通信及下游迁移。',
'核查 §3–5、Table 2–10 与局限：Latin-square 组合留出、原子码、迭代学习消融、位置消融、跨属性听者及视觉骨干对照；发送编码器继承有标签比较 oracle 的准备参数。',
'与 CtD、Rita 的主要差异是动态潜属性和骨干比较；与我们视觉路线高度重合。冻结消息上训练新任务听者不等于零样本完整配对泛化。',
['预训练视觉与符号结构最近邻。','需要对照同容量原子码、接收者重置和实际信息/梯度路径。'],
['仅核实 arXiv v1，未核实同行评审发表。','§3.2/4.6 的单发送器与多发送器容量计数表述有歧义，不能照引“已完全排除带宽”。','主要域受控，部分迁移失败；未复现其代码。'],
'全文核心方法、结果及局限已读。强摘要机制归因需降格为作者主张；不能据本次审查确认其全部比较等价。'),
('2402.16247','full_text','high','read',[88,82,78,82,78],
'提出 CLAP 任务，以社区交互记录训练新人，并比较行为克隆与预先掌握环境技能后的双向消息翻译。',
'§3–5 给出发信/听信/行动分拆；两个合作导航环境，偏置示范覆盖、样本数、消息关闭及人类小型演示；正式版本为 IJCAI 2024 第40–48页。',
'相对零样本协调，允许适应前访问社区交互数据。与我们不同，它主要学习加入既有社区，不是旧配对在新增世界区分下共同重组。',
['后续新人/旧协议适应应比较直接克隆与翻译适配器。','揭示先掌握环境技能、再适应通信的分拆已有先例。'],
['监督示范与集中训练不可冒充独立奖励学习。','大量数据和无 pit 时克隆可优于翻译；人类例子不等于人类语言形成实验。'],
'全文方法、主要实验与正式出版元数据核查。ECTL 更适合社区加入分支，当前新增图实验可借鉴受限适配器基线但须标明改造。'),
('2505.12872','full_text','medium','must-read',[95,74,73,81,69],
'在局部观察的合作采集网格中，以独立 PPO 联合学习行动和离散交流，研究群体、自我互动、时空信息与行为通信。',
'§3–5 与附录目录核查了同步采集、独立参数/梯度、XP/XP+SP、无通信与可见性、伙伴规模和时序任务；主要结构证据为 TopSim 和探针解码。',
'相对 Graesser 2019 的社群交流加入具体采集与独立奖励学习；我们的采集、生存压力、独立主体、双方代码不同均不能单独主张新意。',
['用户最初原始采集设想已有很近的实现。','后续行动任务需要至少对照其同步合作与隐式行为通道。'],
['按 arXiv v2；OpenReview 状态页遇验证，未确认 ICLR 接收。','信息可解码及 TopSim 不充分证明人类语言式词法或位移语义。','社会联系实验不能替代新增区分的稳定性—可塑性干预。'],
'主文核心设置、结果和讨论已读，附录不作全证明审计。角色互换与伙伴自我理解也非我们首创。'),
('2605.11695','full_text','medium','must-read',[99,80,79,86,76],
'用本地 MH 式接受与私人冻结 DINO/MAE 感知，从随机符号模块形成共享离散序列，并考察感知异质性。',
'§3–6：118K COCO 训练/5K验证、DINO-B/B、B/S、B/MAE 三组；三种子和无通信、全接受、等接受率随机、ITM 接受控制；无预训练 BERT 权重。',
'相对先前 MHCG，将随机文本模块与异质冻结视觉组合。我们的非 LLM 视觉符号路线不是空白；可能区别在新需求适应、旧用途约束和行动后果。',
['最直接排除“首次非语言预训练视觉主体产生约定”。','接受率匹配阴性控制与独立视觉测量空间值得学习。'],
['按 arXiv v1，未核实同行评审发表。','仅三视觉配对、单数据集、三种子；精度与覆盖解释未完全区分。','MH 为近似的一步过滤，作者不声称精确后验采样。'],
'主文方法、控制、表3–6、局限和 BERT 初始化说明已核查。尚未直接操纵已建立约定的新区分扩展。'),
('2601.22041','abstract_only','low','monitor',[84,40,55,63,65],
'研究不同感知模态下多步二进制通信及有限微调后的跨系统互通。',
'本轮只核官方 arXiv 摘要及作者代码说明；radar 两次 PDF/HTML 提取失败，故不评判具体冻结端、统计重复与适应预算。',
'沿 Evtimova 等多模态多步游戏增加感知异质性与互操作分析；摘要已有微调恢复交流，不能将跨系统再适应本身当新发现。',
['如扩展到伙伴接触或模态变更，需先补读全文。'],
['abstract_only，全文提取失败。','arXiv 自述将发表于 EvoLang XVI，尚未独立核对正式论文集。'],
'保留为等待完整核读的近邻，不把其结果作为本轮机制结论。'),
('2407.17960','full_text','high','must-read',[95,85,78,87,84],
'揭示视觉通信中表征对齐与 TopSim 的关系，加入对齐惩罚后指标改善却未改善严格视觉组合区分。',
'§4–6：冻结 DINOv2+可训练私有层，REINFORCE 说者/交叉熵听者；COCO、噪声、Winoground 评估，15种子，42种容量开发；对齐惩罚维持表征但不提升 Winoground。',
'接续 Bouchacourt 与 Baroni 2018 的表征漂移问题，直接限制将高 TopSim 当组合能力的解释；我们严格片段和自然联合成功应独立报告。',
['需要将功能迁移与统计结构分开。','随机噪声/新照片/严格关系测试可检验视觉捷径。'],
['使用私有表征间可微信息的正则，不能直接作为完全独立训练。','Winoground 二选一是组合性代理，失败不定位唯一机制。'],
'全文核心方法、结果、讨论及局限核查；正式为 CMCL 2024 workshop，非 ACL 主会。')]
for aid,mode,confidence,verdict,s,contrib,evidence,comparison,reasons,concerns,summary in rows:
    obj=dict(arxiv_id=aid,review_mode=mode,confidence=confidence,verdict=verdict,paper_type='mechanism_or_method',scores=dict(zip(['relevance','evidence','novelty','impact','reproducibility'],s)),contribution=contrib,evidence_summary=evidence,novelty_comparison=comparison,recommendation_reasons=reasons,concerns=concerns,summary=summary)
    obj['scores'].update(author_prior=50,early_signal=50)
    p=B/'reviews'/f'{aid}.json';p.write_text(json.dumps(obj,ensure_ascii=False,indent=2))
    r=subprocess.run(['/Users/xia/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12','/Users/xia/.codex/plugins/cache/zhenhao-arxiv-tools/arxiv-watcher/0.2.0/skills/arxiv-paper-radar/scripts/radar.py','record-review',aid,'--config',str(B/'radar.toml'),'--input',str(p)],capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    print(aid,mode,'validated')
