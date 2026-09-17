"""Independent three-arm pooling and local-link/claim QA for the v15 report."""
from pathlib import Path
from itertools import permutations,product
from urllib.parse import unquote
import hashlib,json,re
import numpy as np
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;BATCH=ROOT/'results/scaled_001';SOURCE=PROJECT/'redesign_v0.13/results/spatial_001';RESET=PROJECT/'redesign_v0.14/results/reset_001';CAL=ROOT/'calibration_001'
REPORT=BATCH/'视觉编码幅度与共同符号形成研究报告.md';SEEDS=(31101,31102,31103,31104);ARMS=('retained','reset','reset_scaled');MAPS=np.array(list(permutations(range(6),2)));MATCHINGS=(((0,1),(2,3),(4,5)),((0,2),(1,4),(3,5)),((0,3),(1,5),(2,4)))
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
  folder=(SOURCE/f'social_s{seed}_p{p}_control') if arm=='retained' else (RESET/f'social_s{seed}_p{p}_reset' if arm=='reset' else BATCH/f'social_s{seed}_p{p}_reset_scaled')
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
for part,key,label in [('normal','old','old熟悉组合终点'),('normal','added','added未训练组合终点'),('normal','sealed','sealed未训练组合终点（主要）'),('auc','old','old全程AUC'),('auc','added','added全程AUC'),('auc','sealed','sealed全程AUC')]:
 values=[agg[a][part][key] for a in ARMS];expected=f'| {label} | '+' | '.join(f'{v*100:.2f}%' for v in values)+' |';checks.append(dict(check='natural_table_and_AUC',expected=expected,passed=expected in text))
remaining=[];repair=[]
for seed in SEEDS:
 c,z,s=[next(r['normal']['sealed'] for r in rows if r['seed']==seed and r['arm']==arm) for arm in ARMS];remaining.append(c-s);repair.append(s-z)
 expected=f'| {seed} | {c*100:.2f}% | {z*100:.2f}% | {s*100:.2f}% | {(c-s)*100:+.2f}个百分点 | {(s-z)*100:+.2f}个百分点 |';checks.append(dict(check='four_seed_table',expected=expected,passed=expected in text))
for key,label,decimals,percent in [('oracle','原生发送联合MAP参照',2,True),('Q','双方原生策略解析期望Q',2,True),('G','原生发送、贪心接收G',2,True),('N','贪心发送、贪心接收N',2,True),('H','地图联合条件熵（nats）',6,False),('codes','贪心使用的完整码数',2,False)]:
 nums=[agg[a]['protocol'][key]*(100 if percent else 1) for a in ARMS];expected=f'| {label} | '+' | '.join(f'{v:.{decimals}f}'+('%' if percent else '') for v in nums)+' |';checks.append(dict(check='protocol_table',expected=expected,passed=expected in text))
for metric,prefix,suffix in [('common30','正常消息成功率为',''),('shuffle','打乱消息后为','')]:
 sentence=prefix+'、'.join(f"{agg[a]['normal'][metric]*100:.2f}%" for a in ARMS);checks.append(dict(check='intervention_summary',expected=sentence,passed=sentence in text))
checks.append(dict(check='blank_all_three_equal_report',passed=all(abs(agg[a]['normal']['blank']-1/30)<1e-15 for a in ARMS) and '固定空消息均为3.33%' in text))
calibration=[]
for seed,p,who in product(SEEDS,(1,2,3),(0,1)):
 folder=CAL/f's{seed}_p{p}';raw=load(folder/f'person{who}.npz');r=raw['retained_h'].astype(np.float64);z=raw['reset_h'].astype(np.float64)
 rn=np.sqrt((r*r).sum(1)).mean();zn=np.sqrt((z*z).sum(1)).mean();alpha=float(np.float32(rn/zn));effective=(raw['reset_h']*np.float32(alpha)).astype(np.float64);sn=np.sqrt((effective*effective).sum(1)).mean();relative=abs(sn/rn-1)
 calibration.append(dict(seed=seed,partition=p,person=who,alpha=alpha,relative_l2_error=float(relative)));hashes[str(folder/f'person{who}.npz')]=sha(folder/f'person{who}.npz')
alphas=[x['alpha'] for x in calibration];max_error=max(x['relative_l2_error'] for x in calibration)
for name,value,fmt in [('alpha_min',min(alphas),'.6f'),('alpha_max',max(alphas),'.6f'),('alpha_mean',np.mean(alphas),'.6f')]:checks.append(dict(check='calibration_numbers',key=name,value=float(value),passed=format(value,fmt) in text))
checks.append(dict(check='calibration_max_relative_error',value=max_error,passed=f'{max_error/1e-8:.2f}×10⁻⁸' in text))
for name,values in [('remaining',remaining),('repair',repair)]:
 mean=float(np.mean(values));checks.append(dict(check='predefined_primary_or_auxiliary_difference',key=name,value=mean,passed=f'{mean*100:.2f}个百分点' in text))
checks.append(dict(check='remaining_three_positive_one_negative',passed=sum(x>0 for x in remaining)==3 and sum(x<0 for x in remaining)==1 and '三正一负' in text))
checks.append(dict(check='repair_four_positive',passed=all(x>0 for x in repair) and '四个来源的修复差都为正' in text))
links=[]
for target in re.findall(r'\]\(([^)]+)\)',text):
 if target.startswith(('http:','https:')):continue
 path=Path(unquote(target));path=(path if path.is_absolute() else REPORT.parent/path).resolve();links.append(dict(path=str(path),exists=path.exists()));checks.append(dict(check='local_link',path=str(path),passed=path.exists()))
for phrase in ('n=4；不是36个独立重复','本轮不是独立确认','不能将恢复比例解释为已识别的因果中介占比','不能跨政策直接减N','AUC概括整个学习过程的成绩水平','新的条件尚未训练','价值预测既进入每批策略项的优势权重，又与消息策略共用发送端梯度裁剪','倍率使用了保留接口在训练支持上的一个统计量'):
 checks.append(dict(check='essential_interpretation_boundary',phrase=phrase,passed=phrase in text))
checks.append(dict(check='all_execution_audits_passed',passed=all(read(folder/'audit_execution.json')['passed'] for folder in (BATCH,SOURCE,RESET))))
checks.append(dict(check='final_report_revision',passed=sha(REPORT)=='a8d515d56b8c8e35c6705c33921a4ff92cf0675b2ad1086ba0bdb04454fef6ce'))
result=dict(passed=all(x['passed'] for x in checks),report_sha256=sha(REPORT),analysis_sha256=sha(BATCH/'scaled_analysis.json'),script_sha256=sha(__file__),checks=checks,check_count=len(checks),aggregate=agg,seed_rows=rows,remaining_seed_differences=remaining,repair_seed_differences=repair,calibration=calibration,local_links=links,source_hashes=hashes,scientific_read_review=dict(primary_direction_fixed=True,old_task_gain_not_composition_specific=True,remaining_sign_heterogeneity_reported=True,no_equivalence_or_mediation_claim=True,native_oracle_policy_and_support_boundary_explicit=True,sender_value_advantage_and_shared_clip_paths_distinguished=True,new_2x2_is_candidate_not_executed=True,four_inherited_sources_not_independent_confirmation=True),scope='Independent raw natural-outcome pooling, source-curve AUC, previously independently audited native probability tables, and raw calibration h. No new training or model forward; local links verified. Interpretive assessment is human-readable scientific review, not established by string matching alone. External literature was not independently re-searched.')
path=BATCH/'audit_report_review.json'
if path.exists():path.rename(path.with_name('audit_report_review_'+sha(path)[:12]+'.json'))
path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],report_sha256=result['report_sha256'],checks=len(checks),failures=[x for x in checks if not x['passed']]),ensure_ascii=False,indent=2));assert result['passed']
