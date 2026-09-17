"""Materialize agent-authored reviews; validate only real arXiv IDs via the skill.

No model, experiment results, new retrieval, or plugin source changes.
The official NeurIPS review retains a null arxiv_id rather than fabricating one.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
RADAR = Path('/Users/xia/.codex/plugins/cache/zhenhao-arxiv-tools/arxiv-watcher/0.2.0/skills/arxiv-paper-radar/scripts/radar.py')
WEIGHTS = dict(relevance=.30, evidence=.20, novelty=.15, impact=.15,
               reproducibility=.10, author_prior=.05, early_signal=.05)

REVIEWS = [
 dict(paper_id='official_supplement_2', arxiv_id=None,
      title='Bridging semantics and pragmatics in information-theoretic emergent communication',
      review_mode='full_text', confidence='high', verdict='must-read', paper_type='empirical information-theoretic emergent communication',
      scores=dict(relevance=92,evidence=82,novelty=73,impact=77,reproducibility=82,author_prior=50,early_signal=50),
      contribution='把共享情境中的指称效用、表征信息保真与压缩放进同一目标，研究局部语用互动能否产生接近人类命名的词汇。',
      evidence_summary='审读正式20页PDF的正文与训练附录。训练时双方共享两对象场景，测试另遮挡背景以评估消息本身；损失权重扫描采用先效用训练、再退火。一般化效用与语义词汇评价使用不同分母，后者含全数据集。',
      novelty_comparison='直接扩展Tucker等VQVIB信息论通信框架，把语境内效用与语境外词汇评价联系起来。新意在目标与词汇证据的连接，非首次研究情境依赖；也没有给出本轮PL/LL同任务训练操纵加同/异需求跨布局整包移植。',
      recommendation_reasons=['是情境信息与词汇形成最直接的正式会议近邻。','有助区分语义评价、成功率和对原听者的因果消息干预。'],
      concerns=['语境遮挡发生在评价，不应写成训练时共享/局部世界信息的配对操纵。','单词式离散表征、预训练视觉先验及显式表征重建目标与本项目不同。','语义词汇评价含训练图像；权重搜索和最接近英语的方案选择不能包装成预先固定的单一比较。'],
      summary='已有严肃研究把语境内沟通和语境外词汇语义连接起来。其遮挡评价及信息重建约束不等于冻结三主体整包移植，且没有由此建立组合性。',
      reading_scope='Main text PDF pp1–10 and Appendix A/B p14; references and checklist inspected for provenance; no independent code execution.'),
 dict(paper_id='2604.03266', arxiv_id='2604.03266',
      title='Emergent Compositional Communication for Latent World Properties',
      review_mode='full_text', confidence='medium', verdict='must-read', paper_type='empirical preprint on discrete physical-property communication',
      scores=dict(relevance=94,evidence=65,novelty=66,impact=76,reproducibility=72,author_prior=50,early_signal=50),
      contribution='从冻结视频特征经多发送者离散瓶颈学习潜在物理性质的比较协议，报告位置消融、留出属性组合和冻结发送者的下游利用。',
      evidence_summary='完整24页packet已读。核心因果段先选已有组合性发送者并冻结，再训练新接收器；置零消息位置造成目标性质较大下降。比较标签训练、oracle预训练、接收器周期重置和固定属性头均为方法组成；跨域零样本有消息塌缩负结果。',
      novelty_comparison='相对Choi等视觉obverter、Ren等迭代学习及既有属性组合游戏，扩展到视频潜在物理性质与位置可寻址接口。与我们的整包迁移相邻，但新接收器训练、越出合法符号支持的置零干预不同于原接收器上的同/异需求供体消息替换。',
      recommendation_reasons=['直接约束“潜在世界内容可迁移”和“消息成分有因果作用”可以怎样主张。','包含组合统计较高却任务表现差、跨域消息塌缩等重要负例。'],
      concerns=['若干下游测试选择已有组合性较高的发送者，不能推广成所有独立训练社会都可迁移。','原文“匹配带宽”的N/K记法与Table7总输出维度说明需按作者实现澄清，本次未复现。','原文没有给发送者绝对性质标签，但接收器有比较真值监督，不能概括为没有任务监督。','正式录用与独立复现未核实；官方v1页面日期与编号月份不一致，保留原记录而不自行改日期。'],
      summary='是较直接但需谨慎的潜在属性与因果成分先行。冻结发送器后训练新解码器、零向量消融和整个既有社群在新背景的正确行动是不同证据。',
      reading_scope='Full24-page extracted packet, methods/results/limitations/appendices all read; no code or model reproduction.'),
 dict(paper_id='2508.06659', arxiv_id='2508.06659',
      title='In-Context Reinforcement Learning via Communicative World Models',
      review_mode='full_text', confidence='high', verdict='read', paper_type='communicative world-model reinforcement-learning method',
      scores=dict(relevance=90,evidence=77,novelty=70,impact=74,reproducibility=80,author_prior=50,early_signal=50),
      contribution='CORAL把信息世界模型与控制器分离，训练消息的世界预测、时序一致性及对动作的有用影响，评估新任务适应和冻结参数泛化。',
      evidence_summary='完整22页v2 packet已读。32维连续消息；双方看相同当下局部观察，但信息主体有历史上下文。ICE比较原消息与零消息的动作分布。随机消息可造成更大ICE却无收益，作者明确区分影响强度与有效性；另有全部权重冻结的零样本评价。',
      novelty_comparison='连接既有因果影响通信与世界模型/ICRL框架。主要线上适应允许新控制器学习，另设两者冻结测试；不能把整篇一概归为接收器重训。其核心干预不是固定物理背景下同/异需求供体的内容适切比较，也不是离散成分重组。',
      recommendation_reasons=['直接提醒我们动作改变或KL较大本身并非语义适切。','冻结泛化与新接收器学习两种条件在同文分列，适合约束迁移表述。'],
      concerns=['连续向量与固定不对称角色，不对应三主体有限符号共同约定。','世界预测、因果影响等辅助目标同时参与训练；不能把收益独立归因于信息分布。','本文未检验同/异需求整包的跨布局对照或成分替换，本次也未复现代码。'],
      summary='已有消息因果作用和冻结模型新情境泛化的直接先行。可借鉴其随机消息负对照，但其ICE不能替代需求目标上的适切率；本轮整包作用也不能独占“因果迁移”创新。',
      reading_scope='Full22-page v2 packet, main text and appendices A–G read; proof inspected only for stated restricted linear-attention scope, not independently verified.')
]

def main():
 out=HERE/'reviews'
 out.mkdir(exist_ok=True)
 receipt={'created_at':datetime.now(timezone.utc).isoformat(),'model_calls':0,
          'current_experiment_results_read':False,'commands':[],
          'native_limit':'NeurIPS formal paper has no verified arXiv ID; its review uses a transparent local schema extension with null ID.'}
 for row in REVIEWS:
  row=dict(row)
  row['scores']=dict(row['scores'])
  row['scores']['overall']=round(sum(row['scores'][k]*v for k,v in WEIGHTS.items()),1)
  row['reviewed_at']=receipt['created_at']
  path=out/(row['paper_id']+'.json')
  if path.exists(): raise FileExistsError(path)
  path.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
  if row['arxiv_id']:
   cmd=['python3',str(RADAR),'record-review',row['arxiv_id'],'--config',str(HERE/'radar.toml'),'--input',str(path)]
   proc=subprocess.run(cmd,capture_output=True,text=True)
   receipt['commands'].append(dict(command=cmd,exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
   if proc.returncode: raise RuntimeError(proc.stderr)
 # Preserve the original selected 20-item pool; only normalize metadata for CLI rendering.
 pool=json.loads((HERE/'candidate_pool20.json').read_text())
 pool['same_pool_source_sha256']=hashlib.sha256((HERE/'candidate_pool20.json').read_bytes()).hexdigest()
 pool.update(config=str(HERE/'radar.toml'),since_hours=23731,candidates_count=20)
 for p in pool['papers']:
  if p['paper_id']=='official_supplement_12':
   p['abstract']='使用平行话语及其意义，按形式与意义的互信息贪心诱导词素；并非只从无意义字符串中发现词义。'
   p['wording_correction']='Official abstract reread; selection and triage decision unchanged.'
  p.setdefault('published',p.get('date',''))
  p.setdefault('pdf_url','https://arxiv.org/pdf/'+p['arxiv_id'] if p['arxiv_id'] else '')
  p.setdefault('topics',['contextual_emergent_communication'])
 path=HERE/'finalize_same_pool.json'
 path.write_text(json.dumps(pool,ensure_ascii=False,indent=2)+'\n')
 cmd=['python3',str(RADAR),'finalize','--candidates',str(path),'--config',str(HERE/'radar.toml'),'--top-n','3','--output',str(HERE/'native_arxiv_reviews_only.md')]
 proc=subprocess.run(cmd,capture_output=True,text=True)
 receipt['commands'].append(dict(command=cmd,exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
 receipt['native_final_report_contains']=2 if proc.returncode==0 else 0
 receipt['native_final_report_status']='completed_arxiv_subset_only' if proc.returncode==0 else 'failed_mixed_pool_has_non_arxiv_id'
 receipt['mixed_review_count']=3
 receipt['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 (HERE/'review_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
 if proc.returncode: raise RuntimeError(proc.stderr)
 print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
