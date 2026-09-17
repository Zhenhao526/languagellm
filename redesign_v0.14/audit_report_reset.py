"""Independent pooling and local-link/claim QA for the v14 report."""
from pathlib import Path
from itertools import permutations,product
from urllib.parse import unquote
import hashlib,json,re
import numpy as np
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;BATCH=ROOT/'results/reset_001';SOURCE=PROJECT/'redesign_v0.13/results/spatial_001';MANIP=ROOT/'manipulation_002'
REPORT=BATCH/'私人非语言经验的迁移研究报告.md';SEEDS=(31101,31102,31103,31104);ARMS=('retained','reset');MAPS=np.array(list(permutations(range(6),2)));MATCHINGS=(((0,1),(2,3),(4,5)),((0,2),(1,4),(3,5)),((0,3),(1,5),(2,4)))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def held(k):return np.array([i for i,x in enumerate(MAPS) if tuple(x) in {q for pair in MATCHINGS[k] for q in (pair,pair[::-1])}])
def groups(p):
 a=held(p-1);s=held(p%3);return dict(old=np.setdiff1d(np.arange(30),np.r_[a,s]),added=a,sealed=s)
text=REPORT.read_text();checks=[];rows=[];hashes={};lookup={tuple(v):i for i,v in enumerate(MAPS)}
for seed,arm in product(SEEDS,ARMS):
 points=[];prots=[];aucs=[]
 for p in (1,2,3):
  folder=(SOURCE/f'social_s{seed}_p{p}_control') if arm=='retained' else BATCH/f'social_s{seed}_p{p}_reset'
  z=load(folder/'final_normal.npz');mids=np.array([lookup[tuple(x)] for x in z['positions']]);correct=(z['place']==np.take_along_axis(z['positions'],z['goals'],1)).all(1);metrics={g:float(correct[np.isin(mids,pool)].mean()) for g,pool in groups(p).items()};metrics['common30']=float(correct.mean())
  for mode in ('shuffle','blank'):
   raw=load(folder/f'final_{mode}.npz');metrics[mode]=float((raw['place']==np.take_along_axis(raw['positions'],raw['goals'],1)).all(1).mean())
  points.append(metrics);curve=read(folder/'curve.json');times=np.array([x['update'] for x in curve]);a={}
  for group in ('old','added','sealed'):
   values=np.array([x['scores']['normal']['map_groups'][group]['both_accuracy'] for x in curve]);a[group]=float(np.dot(np.diff(times),(values[:-1]+values[1:])/2)/(times[-1]-times[0]))
  aucs.append(a)
  for d,r in enumerate(read(folder/'protocol_2400.json')):
   raw=load(folder/f'protocol_2400_d{d}.npz');n=r['metrics']['native']['common30'];g=r['metrics']['greedy']['common30'];prots.append(dict(oracle=n['oracle_joint_map']['J'],Q=n['stochastic']['J'],G=n['greedy']['J'],N=g['greedy']['J'],H=n['joint_conditional_entropy'],codes=len(np.unique(raw['greedy_message']@np.array([7,1])))))
  for name in ('final_normal.npz','curve.json','protocol_2400.json'):hashes[str(folder/name)]=sha(folder/name)
 rows.append(dict(seed=seed,arm=arm,normal={k:float(np.mean([x[k] for x in points])) for k in points[0]},auc={k:float(np.mean([x[k] for x in aucs])) for k in aucs[0]},protocol={k:float(np.mean([x[k] for x in prots])) for k in prots[0]}))
agg={a:{part:{k:float(np.mean([r[part][k] for r in rows if r['arm']==a])) for k in rows[0][part]} for part in ('normal','auc','protocol')} for a in ARMS}
for part,key,label in [('normal','old','熟悉old组合终点'),('normal','added','未训练added组合终点'),('normal','sealed','未训练sealed终点（主要）'),('auc','old','old全程AUC'),('auc','sealed','sealed全程AUC')]:
 c=agg['retained'][part][key];z=agg['reset'][part][key];expected=f'| {label} | {c*100:.2f}% | {z*100:.2f}% | {(c-z)*100:+.2f}个百分点 |';checks.append(dict(check='natural_table_and_AUC',expected=expected,passed=expected in text))
for seed in SEEDS:
 c=next(r['normal']['sealed'] for r in rows if r['seed']==seed and r['arm']=='retained');z=next(r['normal']['sealed'] for r in rows if r['seed']==seed and r['arm']=='reset');expected=f'| {seed} | {c*100:.2f}% | {z*100:.2f}% | {(c-z)*100:+.2f}个百分点 |';checks.append(dict(check='four_seed_table',expected=expected,passed=expected in text))
for key,label,decimals,percent in [('oracle','原生发送联合MAP参照',2,True),('Q','双方原生政策解析成功期望Q',2,True),('G','原生发送、贪心接收G',2,True),('N','贪心发送、贪心接收N',2,True),('H','给定消息后的地图联合条件熵（nats）',6,False),('codes','贪心实际使用的完整码数',2,False)]:
 nums=[agg[a]['protocol'][key]*(100 if percent else 1) for a in ARMS];expected=f'| {label} | '+' | '.join(f'{v:.{decimals}f}'+('%' if percent else '') for v in nums)+' |';checks.append(dict(check='protocol_table',expected=expected,passed=expected in text))
geometry=[];head=[];deltas=[]
for seed,p,who in product(SEEDS,(1,2,3),(0,1)):
 path=MANIP/f's{seed}_p{p}_person{who}.npz';z=load(path);h=z['retained_h'].astype(float);q=z['reset_h'].astype(float);delta=h-q;deltas.append(delta);geometry.append([np.linalg.norm(h,axis=1).mean(),np.linalg.norm(q,axis=1).mean(),np.sqrt((delta*delta).mean())]);row={}
 for arm in ARMS:
  correct=(z[arm+'_logits'].argmax(-1)==z['positions']).all(1);row[arm]=dict(all=float(correct.mean()),sealed=float(correct[np.isin(z['map_ids'],groups(p)['sealed'])].mean()))
 head.append(row);hashes[str(path)]=sha(path)
geo=np.mean(geometry,axis=0);manip={a:{k:float(np.mean([r[a][k] for r in head])) for k in ('all','sealed')} for a in ARMS}
for label,value,fmt in [('retained_h_norm',geo[0],'.4f'),('reset_h_norm',geo[1],'.4f'),('mean_person_RMS',geo[2],'.4f'),('retained_sealed_private_head',manip['retained']['sealed']*100,'.2f'),('reset_sealed_fixed_head',manip['reset']['sealed']*100,'.2f'),('retained_all_head',manip['retained']['all']*100,'.2f'),('reset_all_fixed_head',manip['reset']['all']*100,'.2f')]:checks.append(dict(check='manipulation_numbers',key=label,value=value,passed=format(value,fmt) in text))
links=[]
for target in re.findall(r'\]\(([^)]+)\)',text):
 if target.startswith(('http:','https:')):continue
 path=Path(unquote(target));path=(path if path.is_absolute() else REPORT.parent/path).resolve();links.append(dict(path=str(path),exists=path.exists()));checks.append(dict(check='local_link',path=str(path),passed=path.exists()))
for phrase in ('四个开发来源不足以','不把分区、方向或照片数','固定读出的兼容性改变','不能直接减去改变发送政策后的N','本轮没有运行该条件'):
 checks.append(dict(check='essential_interpretation_boundary',phrase=phrase,passed=phrase in text))
checks.append(dict(check='RMS_aggregation_explained',passed=('先对每人计算' in text or '每人分别计算' in text or '每个人' in text and 'RMS' in text)))
checks.append(dict(check='all_execution_audits_passed',passed=read(BATCH/'audit_execution.json')['passed'] and read(SOURCE/'audit_execution.json')['passed']))
result=dict(passed=all(x['passed'] for x in checks),report_sha256=sha(REPORT),analysis_sha256=sha(BATCH/'reset_analysis.json'),script_sha256=sha(__file__),checks=checks,check_count=len(checks),aggregate=agg,seed_rows=rows,manipulation=dict(aggregated_head_scores=manip,retained_mean_h_norm=float(geo[0]),reset_mean_h_norm=float(geo[1]),mean_per_person_RMS=float(geo[2]),pooled_RMS_not_the_report_metric=float(np.sqrt((np.concatenate(deltas)**2).mean())),norm_ratio=float(geo[0]/geo[1])),local_links=links,source_hashes=hashes,scope='Independent pooling of raw natural outcomes, source-curve AUC, already independently audited probability tables, and raw fixed-head/latent diagnostics. No new training or forward pass. Interpretation read-review separate from code/metric checks; no independent rereading of external papers.')
path=BATCH/'audit_report_review.json'
if path.exists():path.rename(path.with_name('audit_report_review_'+sha(path)[:12]+'.json'))
path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],report_sha256=result['report_sha256'],checks=len(checks),failures=[x for x in checks if not x['passed']]),ensure_ascii=False,indent=2));assert result['passed']
