"""Read-only report QA, independently pooling the saved source results."""
from pathlib import Path
import hashlib,json,re
from urllib.parse import unquote
import numpy as np
from itertools import permutations

ROOT=Path(__file__).resolve().parent;BATCH=ROOT/'results/spatial_001'
REPORT=BATCH/'私人空间能力与共同符号形成研究报告.md'
SEEDS=(31101,31102,31103,31104);ARMS=('control','equivariant')
MAPS=np.array(list(permutations(range(6),2)))
MATCHINGS=(((0,1),(2,3),(4,5)),((0,2),(1,4),(3,5)),((0,3),(1,5),(2,4)))
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def held(k):return np.array([i for i,pair in enumerate(MAPS) if tuple(pair) in {x for pair in MATCHINGS[k] for x in (pair,pair[::-1])}])
def groups(p):
 a=held(p-1);s=held(p%3);return dict(old=np.setdiff1d(np.arange(30),np.r_[a,s]),added=a,sealed=s)
def zload(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
text=REPORT.read_text();checks=[];seed_rows=[];private={a:[] for a in ARMS};protocol={a:[] for a in ARMS};sources={};values={}
for seed in SEEDS:
 cell={}
 for arm in ARMS:
  results=[]
  for p in (1,2,3):
   folder=BATCH/f'social_s{seed}_p{p}_{arm}';normal=zload(folder/'final_normal.npz');true=np.take_along_axis(normal['positions'],normal['goals'],axis=1);success=(normal['place']==true).all(1);mids=np.array([np.flatnonzero((MAPS==x).all(1))[0] for x in normal['positions']]);scores={g:float(success[np.isin(mids,pool)].mean()) for g,pool in groups(p).items()};scores['all']=float(success.mean());results.append(scores)
   for mode in ('shuffle','blank'):
    raw=zload(folder/f'final_{mode}.npz');true=np.take_along_axis(raw['positions'],raw['goals'],axis=1);scores[mode]=float((raw['place']==true).all(1).mean())
   prot=read(folder/'protocol_2400.json')
   for d,r in enumerate(prot):
    raw=zload(folder/f'protocol_2400_d{d}.npz');n=r['metrics']['native']['common30'];g=r['metrics']['greedy']['common30'];protocol[arm].append(dict(oracle=n['oracle_joint_map']['J'],Q=n['stochastic']['J'],G=n['greedy']['J'],N=g['greedy']['J'],entropy=n['joint_conditional_entropy'],codes=len(np.unique(raw['greedy_message']@np.array([7,1])))))
   pf=BATCH/f'private_s{seed}_p{p}_{arm}';raw=zload(pf/'evaluation_2400.npz');saved=read(pf/'result.json')
   for who in (0,1):
    correct=raw['logits'][who].argmax(-1)==raw['positions'];private[arm].append(dict(old=correct[np.isin(raw['map_ids'],groups(p)['old'])].mean(),added=correct[np.isin(raw['map_ids'],groups(p)['added'])].all(-1).mean(),sealed=correct[np.isin(raw['map_ids'],groups(p)['sealed'])].all(-1).mean(),jsd=saved['scores']['persons'][who]['equivariance']['groups']['all']['jsd'],entropy=saved['scores']['persons'][who]['groups']['all']['policy_entropy']))
   for path in (folder/'final_normal.npz',folder/'protocol_2400.json',pf/'result.json',pf/'evaluation_2400.npz'):sources[str(path)]=sha(path)
  cell[arm]={k:float(np.mean([r[k] for r in results])) for k in results[0]}
 seed_rows.append(dict(seed=seed,**cell))
for arm in ARMS:
 values[arm]=dict(social={k:float(np.mean([r[arm][k] for r in seed_rows])) for k in seed_rows[0][arm]},private={k:float(np.mean([r[k] for r in private[arm]])) for k in private[arm][0]},protocol={k:float(np.mean([r[k] for r in protocol[arm]])) for k in protocol[arm][0]})
for group,label in [('old','熟悉组合自然N'),('added','added未训练组合自然N'),('sealed','sealed未训练组合自然N，主要终点')]:
 c=values['control']['social'][group];e=values['equivariant']['social'][group];row=f'| {label} | {c*100:.2f}% | {e*100:.2f}% | {(e-c)*100:+.2f}个百分点 |'.replace('-','−');checks.append(dict(check='social_report_table',key=group,expected=row,passed=row in text))
for r in seed_rows:
 c=r['control']['sealed'];e=r['equivariant']['sealed'];row=f"| {r['seed']} | {c*100:.2f}% | {e*100:.2f}% | {(e-c)*100:+.2f} |".replace('-','−');checks.append(dict(check='four_independent_seed_rows',expected=row,passed=row in text))
for key,label,precision,percent in [('old','熟悉组合单目标准确率',2,True),('added','added未训练组合双目标成功率',2,True),('sealed','sealed未训练组合双目标成功率',2,True),('jsd','全30图、全720置换政策JSD',6,False),('entropy','全30图平均政策熵，nats',6,False)]:
 nums=[values[a]['private'][key]*(100 if percent else 1) for a in ARMS];row=f'| {label} | '+ ' | '.join(f'{x:.{precision}f}'+('%' if percent else '') for x in nums)+' |';checks.append(dict(check='private_report_table',expected=row,passed=row in text))
for key,label,precision,percent in [('oracle','固定原生发送政策的联合MAP最优解码参照',2,True),('Q','双方原生策略的解析成功期望Q',2,True),('G','原生发送、贪心接收G',2,True),('N','逐符号贪心发送、贪心接收N',2,True),('entropy','给定消息后的地图联合条件熵，nats',6,False),('codes','实际贪心使用的完整码数',2,False)]:
 nums=[values[a]['protocol'][key]*(100 if percent else 1) for a in ARMS];row=f'| {label} | '+' | '.join(f'{x:.{precision}f}'+('%' if percent else '') for x in nums)+' |';checks.append(dict(check='protocol_report_table',expected=row,passed=row in text))
links=[]
for target in re.findall(r'\]\(([^)]+)\)',text):
 if target.startswith(('http:','https:')):continue
 path=Path(unquote(target));path=path if path.is_absolute() else REPORT.parent/path;path=path.resolve();pending='pixel_workflow_001/README.md' in str(path);links.append(dict(path=str(path),exists=path.exists(),allowed_temporarily_pending=pending));checks.append(dict(check='local_link',path=str(path),passed=path.exists() or pending))
audit=read(BATCH/'audit_execution.json');checks.append(dict(check='complete_execution_and_probability_audit',passed=audit['passed']))
result=dict(passed=all(x['passed'] for x in checks),report_sha256=sha(REPORT),analysis_sha256=sha(BATCH/'spatial_analysis.json'),script_sha256=sha(Path(__file__)),checks=checks,values=values,seed_rows=seed_rows,local_links=links,source_hashes=sources,scope='Independently pooled original natural outcomes, private greedy logits, and previously independently verified full-S6/protocol results. External-paper full text not re-reviewed here; covered by dedicated literature review. Source-group/seed nesting checked. Pending pixel workflow link is explicitly allowed by report author.')
(BATCH/'audit_report_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],report_sha256=result['report_sha256'],checks=len(checks),failures=[x for x in checks if not x['passed']],pending=[x for x in links if not x['exists']]),ensure_ascii=False,indent=2));assert result['passed']
