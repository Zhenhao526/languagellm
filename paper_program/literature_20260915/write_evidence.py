from pathlib import Path
import json,hashlib,datetime
B=Path(__file__).resolve().parent
c=json.loads((B/'targeted_candidates.json').read_text()); byid={p['arxiv_id']:p for p in c['papers']}
specs=[
('2209.15342','NeurIPS 2022，正式同行评审会议论文','https://papers.nips.cc/paper_files/paper/2022/file/093b08a7ad6e6dd8d34b9cc86bb5f07c-Paper-Conference.pdf','10.48550/arXiv.2209.15342','正式PDF第3–9页，§2.3–5.3；arXiv全文附录A/D','信息/共适应已有分解，听者收敛与正则已有因果操纵','原接收覆盖与保持旧任务的约束前沿不同，不能改名重复损失分解'),
('2601.10169','ICLR 2025，正式同行评审会议论文；2026年才上传本arXiv条目','https://proceedings.iclr.cc/paper_files/paper/2025/file/fb9d01fb202e360b2f78510c47e0fa3e-Paper-Conference.pdf','10.48550/arXiv.2601.10169','正式PDF第2–8页、34页§J','概念分解课程、代码本与未见概念合取已有强阳性结果','oracle分组和传递潜向量须披露；无这些信息的性能与机制才需检验'),
('2604.03266','arXiv v1，同行评审发表未核实','https://arxiv.org/abs/2604.03266','10.48550/arXiv.2604.03266','§3.1–3.4、§4.1–4.10、§5局限；Table7计数疑点','DINO/V-JEPA、组合留出、原子码与冻结发送器迁移已有','有比较oracle准备与接收损失反传；带宽匹配表述需复现后核清'),
('2402.16247','IJCAI 2024 Main Track，40–48页，正式同行评审论文','https://www.ijcai.org/proceedings/2024/5','10.24963/ijcai.2024/5','正式PDF§3–5；arXiv方法§4.2及实验§5','环境能力预训练、协议双向翻译及新人加入已有方法','示范监督的社区加入，与纯奖励的新世界区分扩展不同'),
('2505.12872','arXiv v2；OpenReview状态页遇验证，接收状态未核实','https://arxiv.org/abs/2505.12872','10.48550/arXiv.2505.12872','§3–6；附录A训练与D/E控制目录','独立PPO、局部观察、同步采集、社会网络和行为通信已有','不把TopSim/可解码直接当人类语法；当前单轮任务也非完整原始社会'),
('2605.11695','arXiv v1，同行评审发表未核实','https://arxiv.org/abs/2605.11695','10.48550/arXiv.2605.11695','§3–6，Tables1–6，Appendix B.1/B.6','冻结DINO/MAE、随机符号模块、私人本地接受学习已有','只读共同图像的不同增强视图；旧约定新增区分与保留代价未在本轮核读处实验'),
('2601.22041','arXiv v1；作者称将发表于EvoLang XVI，正式论文集未独立核实','https://arxiv.org/abs/2601.22041','10.48550/arXiv.2601.22041','仅官方摘要；全文提取失败','摘要已报告跨模态互通失败及有限微调恢复','不能据摘要断言具体冻结函数、种子和适应预算；保留待读'),
('2407.17960','CMCL 2024，ACL Anthology收录的同行评审workshop论文，非ACL主会','https://aclanthology.org/2024.cmcl-1.5/','10.18653/v1/2024.cmcl-1.5','§4–8；DINOv2、15种子、Winoground与噪声','TopSim提高与严格组合区分改善可以分离','需任务功能指标和视觉捷径控制；表征对齐正则会引入跨主体信息')]
rows=[]
for aid,status,url,doi,loc,support,limit in specs:
    p=byid[aid];r=json.loads((B/'reviews'/f'{aid}.json').read_text())
    rows.append(dict(id=aid,title=p['title'],authors=p['authors'],status=status,official_url=url,arxiv_url=p['abs_url'],doi=doi,read_mode=r['review_mode'],confidence=r['confidence'],evidence_location=loc,support=support,boundary=limit,packet=str(B/'packets'/f'{aid}.md'),review=str(B/'reviews'/f'{aid}.json')))
rows.extend([
 dict(id='Lee2024',title='One-to-Many Communication and Compositionality in Emergent Communication',authors=['Heeyoung Lee'],status='EMNLP 2024，20794–20811页，正式同行评审主会论文',official_url='https://aclanthology.org/2024.emnlp-main.1157/',doi='10.18653/v1/2024.emnlp-main.1157',read_mode='targeted_full_text_methods_results',confidence='high',evidence_location='PDF第3–6页：§3奖励、§4优化、§6.2新听者/§6.3群组联合正确',support='听者兴趣与群组全员正确压力影响结构已有直接实验',boundary='属性元数据、独立听者及交叉熵辅助，与单采集者两需求纯奖励不同；组规模效果不是普遍单调',local_text=str(B/'primary/lee2024.txt')),
 dict(id='CELEBI2025',title='A Compressive-Expressive Communication Framework for Compositional Representations',authors=['Rafael Elberg','Felipe del Rio','Mircea Petrache','Denis Parra'],status='NeurIPS 2025，正式同行评审主会论文',official_url='https://proceedings.neurips.cc/paper_files/paper/2025/hash/3310034c97fab48fdbcba18f90fd5364-Abstract-Conference.html',arxiv_url='https://arxiv.org/abs/2501.19182',doi='10.52202/085713-1195',read_mode='targeted_full_text_methods_results',confidence='high',evidence_location='PDF第3–7、10页：§2.3、§3.1–3.2、Table1、§7',support='冻结视觉VAE、逐符号解码、冻结旧听者的终态模仿已有方法',boundary='可微信道、重建目标与迭代学习；结构更高并不在每个数据集/指标均超过所有基线',local_text=str(B/'primary/celebi2025.txt'))])
(B/'citation_evidence.json').write_text(json.dumps(dict(date='2026-09-15',scope='bounded nearest-neighbour audit; not exhaustive',records=rows),ensure_ascii=False,indent=2))
md=['# 最近邻引用与证据表','','七篇全文层面评审、一篇摘要评审，另两篇正式原文定向方法核查。证据定位按所读版本；统计数字是作者报告，未独立复现。完整边界见 [novelty_gap.md](novelty_gap.md)。','', '| 论文与正式状态 | 核查定位与深度 | 已有证据覆盖 | 对本项目的限制 |','|---|---|---|---|']
for r in rows:
    md.append(f"| **[{r['title']}]({r['official_url']})**；{r['status']} | {r['evidence_location']}；{r['read_mode']} | {r['support']} | {r['boundary']} |")
md+=['','## 可直接引用的书目信息','']
for r in rows:
    md.append(f"- {', '.join(r['authors'])}. **{r['title']}**. {r['status']}。DOI：[{r['doi']}](https://doi.org/{r['doi']})。")
md+=['','本地文本与完整机器表见 [citation_evidence.json](citation_evidence.json)。新下载PDF均命名存于 `primary/`，文件头、全部页面提取及哈希见 [download_verification.json](download_verification.json)；既有 CtD、Kaszyński 和 Lee 原库文件只读复用，来源见 [primary/manifest.json](primary/manifest.json)。同一内容的雷达缓存与命名PDF用硬链接保留，未改插件或旧库。','','已有必须比较的较早背景继续复用本地文献：Kottur 2017（任务成功不保证自然结构）、Lowe 2019（通信因果测量）、Graesser 2019（社群接触）、Korbak 2021/2022（模板与接收经验迁移）、Ohmer 2022（视觉属性相关性与感知偏置）。这些本轮没有重新声称完成全文审阅，详见 [既有审计](../../research_program/论文证据与创新性审计_2026-09-15.md)。']
(B/'citation_evidence.md').write_text('\n'.join(md)+'\n')
print('wrote 10 citation evidence records')
