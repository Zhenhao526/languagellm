"""Record the bounded official-source supplement; never alter native collector output."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
original=json.loads((HERE/'collector_candidates.json').read_text())
# These are paraphrases of official abstracts read in the current bounded review.
extra=[
('2604.03266','Emergent Compositional Communication for Latent World Properties','Tomek Kaszyński','2026-03-18','arXiv v1','https://arxiv.org/abs/2604.03266v1','以冻结视频表示、四主体消息和迭代学习提取潜在物理性质，报告属性定点干预、留出组合与规划测试。','full_text_packet','直接检验可迁移潜在内容与消息干预；需区分训练目标、代理解码器和部署效应。'),
(None,'Bridging semantics and pragmatics in information-theoretic emergent communication','Eleonora Gualdoni; Mycal Tucker; Roger P. Levy; Noga Zaslavsky','2024-12','NeurIPS 2024 formal','https://proceedings.neurips.cc/paper_files/paper/2024/hash/2548fbe155ed2405488b7d5373013a64-Abstract-Conference.html','在自然图像指称任务中结合语用效用和信息约束，比较有共享情境训练与脱离情境的词汇语义。','full_text_packet','最直接的情境信息操纵近邻；检查是否采用同任务与供体跨背景消息干预。'),
('2508.06659','In-Context Reinforcement Learning via Communicative World Models','Fernando Martinez-Lopez; Tao Li; Yingdong Lu; Juntao Chen','2025-08-08','arXiv v2 revised 2026-06-07','https://arxiv.org/abs/2508.06659v2','预训世界模型信息主体向新控制主体发送情境，以消息对动作的影响训练表示并报告新任务适应。','full_text_packet','最直接的因果影响与跨任务转移近邻；重点核离散性、信息角色、干预定义及内容适切证据。'),
('2411.10173','Semantics and Spatiality of Emergent Communication','Rotem Ben Zion; Boaz Carmeli; Orr Paradise; Yonatan Belinkov','2024-11-15','arXiv v1; NeurIPS 2024 claim on official page','https://arxiv.org/abs/2411.10173','定义跨实例语义一致性并比较辨别与重建目标的最优协议及距离意义。','retain_method_neighbor','已读旧库，直接限制成功率即语义的主张；非同任务世界可见性操纵。'),
(None,'Learning to Refer: How Scene Complexity Affects Emergent Communication in Neural Agents','Dominik Künkele; Simon Dobnik','2025-09','IWCS 2025 formal','https://aclanthology.org/2025.iwcs-main.25/','扩展CLEVR场景，研究更复杂的指称环境对神经主体词汇与属性落地的影响。','retain_ecology_neighbor','场景复杂性是直接生态先行，但摘要未显示本轮的同内容消息跨布局配对。'),
('2601.10169','CtD: Composition through Decomposition in Emergent Communication','Boaz Carmeli; Ron Meir; Yonatan Belinkov','2025','ICLR 2025 formal; arXiv upload 2026 separately','https://proceedings.iclr.cc/paper_files/paper/2025/hash/fb9d01fb202e360b2f78510c47e0fa3e-Abstract-Conference.html','多目标游戏先学习基础概念码本，再通过组合码本指称未见图像；部分组合不需再训练。','retain_composition_boundary','局部研究已多次深读；提醒完整包迁移不等于成分重组。'),
('2502.12624','Implicit Repair with Reinforcement Learning in Emergent Communication','Fábio Vital; Alberto Sardinha; Francisco S. Melo','2025-02-18','arXiv v2 2025-02-24; AAMAS2025 in comments','https://arxiv.org/abs/2502.12624v2','在输入与通道加噪下研究冗余协议、任务稳健性和泛化。','retain_secondary','噪声稳健性并非需求内容适切的同背景反事实比较。'),
('2601.03254','Automated Semantic Rules Detection (ASRD) for Emergent Communication Interpretation','Bastien Vanderplaetse; Xavier Siebert; Stéphane Dupont','2026-01-06','arXiv v1','https://arxiv.org/abs/2601.03254v1','自动找出Lewis游戏消息模式与输入属性的关联规则。','retain_interpretability_boundary','关联式解码不直接检验消息使原听者跨背景正确行动。'),
('2402.16247','Learning Translations: Emergent Communication Pretraining for Cooperative Language Acquisition','Dylan Cope; Peter McBurney','2024-02-26','arXiv v1','https://arxiv.org/abs/2402.16247v1','新加入者可读社群互动资料；比较模仿与涌现协议预训后翻译学习。','retain_transfer_boundary','跨社群习得需要新的学习；非冻结同政策跨物资布局干预。'),
(None,'One-to-Many Communication and Compositionality in Emergent Communication','Heeyoung Lee','2024-11','EMNLP2024 formal','https://aclanthology.org/2024.emnlp-main.1157/','广播多听者，区分人数、不同属性兴趣和联合成功压力对组合性影响。','retain_ecology_neighbor','需求与合作结构操纵已有直接先行；并非本轮世界信息可见性和整包功能迁移。'),
(None,'Frequency & Compositionality in Emergent Communication','Jean-Baptiste Sevestre; Emmanuel Dupoux','2025-11','EMNLP2025 formal','https://aclanthology.org/2025.emnlp-main.1387/','操纵Zipf频率，指出有限资料暴露与组合结构形成的关系。','retain_training_confound','经验频率与训练路径是环境比较的混淆；本轮不改变抽样需求/布局。'),
(None,'Morpheme Induction for Emergent Language','Brendon Boldt; David R. Mortensen','2025-11','EMNLP2025 formal','https://aclanthology.org/2025.emnlp-main.1284/','使用平行话语及其意义，按形式与意义的互信息贪心诱导词素；并非只从无意义字符串中发现词义。','retain_component_boundary','词素诱导与包级部署迁移是不同证据层；本轮未做成分干预。'),
]
triage={
'2609.01491':('retain_LLM_background','复杂局部信息合作与协议代际学习相关；LLM英语先验及postmortem讨论与当前随机MLP离散共同学习不同。'),
'2606.06380':('skip_current_question','意识方法论和自指通信的概念目标较远；摘要未给当前情境迁移的受控比较。'),
'2605.08613':('retain_generalization_boundary','信息瓶颈和未见网络状态泛化相邻；主要面向物理网络约束，不以内容适切整包干预为主。'),
'2607.03752':('retain_interpretability_boundary','从消息训练扩散模型重建原图；外部新解码器可恢复内容不等于原听者利用内容。'),
'2607.00233':('retain_LLM_background','记忆架构×通道对Lewis协调重要，但不是世界信息可见性×冻结包迁移。'),
'2605.09522':('skip_current_question','具身情绪分类对齐和命名游戏；当前行动与语义反事实问题较远。'),
'2609.06025':('skip_current_question','研究依存距离与增量处理，起点含人工语言训练；不同于无预置代码的情境材料映射。'),
'2606.19632':('skip_current_question','蒸馏策略的安全性质验证；不能替代原协议消息内容与跨背景适切性的因果证据。')}
papers=[]
for p in original['papers']:
 q=dict(p);q['paper_id']=p['arxiv_id'];q['retrieval']='native_deterministic_collector';q['triage_decision'],q['triage_reason']=triage[p['arxiv_id']];q['triage_basis']='title and complete collected abstract only';q['full_text_reviewed_this_round']=False;papers.append(q)
for i,(aid,title,authors,date,version,url,abstract,decision,reason) in enumerate(extra):
 papers.append(dict(paper_id=aid or 'official_supplement_'+str(i+1),arxiv_id=aid,title=title,authors=authors.split('; '),date=date,version=version,abs_url=url,abstract=abstract,abstract_format='Chinese paraphrase of official abstract checked this round',retrieval='bounded_official_source_supplement',triage_basis='title and official abstract only',triage_decision=decision,triage_reason=reason,full_text_reviewed_this_round=False))
assert len(papers)==20 and len({p['paper_id'] for p in papers})==20
result=dict(schema='bounded_mixed_official_pool_v1',created_at=datetime.now(timezone.utc).isoformat(),top_n=3,pool_count=20,native_collector_count=8,official_supplement_count=12,window='2024-01-01 through 2026-09-16; formal year and first/revised arXiv dates distinguished',collector_sha256=hashlib.sha256((HERE/'collector_candidates.json').read_bytes()).hexdigest(),limitations=['Not a complete historical or systematic review. Native category ceiling misses older papers.','Official venue papers retain null arxiv_id; no fake arXiv identifiers are inserted.','Author citation metadata was not used to choose the three full texts.','No current experiment interim/final metrics read to select papers.'],papers=papers)
(HERE/'candidate_pool20.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['# 20篇候选的逐一摘要分流','', '原生采集8篇，官方定向补充12篇。全部先按标题与摘要分流；full_text_packet只是后续阅读选择，不是质量评审。','', '|#|论文|来源/版本|分流|原因|','|---|---|---|---|---|']
for i,p in enumerate(papers,1):lines.append(f"|{i}|[{p['title']}]({p['abs_url']})|{p.get('version',p.get('versioned_id'))}|{p['triage_decision']}|{p['triage_reason']}|")
(HERE/'候选逐一分流.md').write_text('\n'.join(lines)+'\n')
print('20 candidates; 3 full-text selections recorded before packet reading')
