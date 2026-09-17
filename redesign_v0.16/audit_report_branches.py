"""Independent raw pooling plus fixed-report numerical, link and scientific read review.
No training/model forward. Protocol tables inherit the completed independent execution audit.
"""
from pathlib import Path
from itertools import permutations, product
from urllib.parse import unquote
import hashlib, json, re
import numpy as np
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;BATCH=ROOT/'results/branches_001'
REPORT=BATCH/'策略与价值输入幅度的通信形成研究报告.md';EXPECTED='37d6ae10766c7390cfb249ca090e24c2a0d7a5ca681db9555e9e1e8db760ec00'
SEEDS=(31101,31102,31103,31104);ARMS=('neither','policy_only_scale','value_only_scale','both')
MAPS=np.array(list(permutations(range(6),2)));MATCHINGS=(((0,1),(2,3),(4,5)),((0,2),(1,4),(3,5)),((0,3),(1,5),(2,4)))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def held(k):return np.array([i for i,x in enumerate(MAPS) if tuple(x) in {q for pair in MATCHINGS[k] for q in (pair,pair[::-1])}])
def groups(p):
 a=held(p-1);s=held(p%3);return dict(old=np.setdiff1d(np.arange(30),np.r_[a,s]),added=a,sealed=s)
def folder(seed,p,arm):
 if arm=='neither':return PROJECT/f'redesign_v0.14/results/reset_001/social_s{seed}_p{p}_reset'
 if arm=='both':return PROJECT/f'redesign_v0.15/results/scaled_001/social_s{seed}_p{p}_reset_scaled'
 return BATCH/f'social_s{seed}_p{p}_{arm}'
text=REPORT.read_text();checks=[];rows=[];hashes={};lookup={tuple(x):i for i,x in enumerate(MAPS)}
def check(name,passed,**context):checks.append(dict(check=name,passed=bool(passed),**context))
def table(label,values,decimals=2,percent=True):
 expected='| '+label+' | '+' | '.join(f'{v*(100 if percent else 1):.{decimals}f}'+('%' if percent else '') for v in values)+' |'
 check('exact_formatted_report_table',expected in text,expected=expected)
for seed,arm in product(SEEDS,ARMS):
 points=[];prots=[];aucs=[]
 for p in (1,2,3):
  f=folder(seed,p,arm);metrics={}
  for mode in ('normal','shuffle','blank'):
   z=load(f/f'final_{mode}.npz');mids=np.array([lookup[tuple(x)] for x in z['positions']]);correct=(z['place']==np.take_along_axis(z['positions'],z['goals'],1)).all(1)
   if mode=='normal':
    metrics.update({g:float(correct[np.isin(mids,pool)].mean()) for g,pool in groups(p).items()});metrics['common30']=float(correct.mean())
   else:metrics[mode]=float(correct.mean())
   hashes[str(f/f'final_{mode}.npz')]=sha(f/f'final_{mode}.npz')
  points.append(metrics);curve=read(f/'curve.json');times=np.array([x['update'] for x in curve]);a={}
  check('eight_fixed_timepoints',np.array_equal(times,[0,100,300,600,1200,1800,2100,2400]),run=str(f))
  for group in ('old','added','sealed'):
   vals=np.array([x['scores']['normal']['map_groups'][group]['both_accuracy'] for x in curve]);a[group]=float(np.dot(np.diff(times),(vals[:-1]+vals[1:])/2)/2400)
  aucs.append(a)
  for d,r in enumerate(read(f/'protocol_2400.json')):
   raw=load(f/f'protocol_2400_d{d}.npz');n=r['metrics']['native']['common30'];g=r['metrics']['greedy']['common30']
   prots.append(dict(oracle=n['oracle_joint_map']['J'],Q=n['stochastic']['J'],G=n['greedy']['J'],N=g['greedy']['J'],H=n['joint_conditional_entropy'],codes=len(np.unique(raw['greedy_message']@np.array([7,1])))))
   hashes[str(f/f'protocol_2400_d{d}.npz')]=sha(f/f'protocol_2400_d{d}.npz')
  for name in ('curve.json','protocol_2400.json'):hashes[str(f/name)]=sha(f/name)
 rows.append(dict(seed=seed,arm=arm,normal={k:float(np.mean([x[k] for x in points])) for k in points[0]},auc={k:float(np.mean([x[k] for x in aucs])) for k in aucs[0]},protocol={k:float(np.mean([x[k] for x in prots])) for k in prots[0]}))
agg={a:{part:{k:float(np.mean([r[part][k] for r in rows if r['arm']==a])) for k in rows[0][part]} for part in ('normal','auc','protocol')} for a in ARMS}
for part,key,label in [('normal','old','old自然终点'),('normal','added','added自然终点'),('normal','sealed','sealed自然终点'),('auc','old','old全程AUC'),('auc','added','added全程AUC'),('auc','sealed','sealed全程AUC')]:table(label,[agg[a][part][key] for a in ARMS])
contrasts=[]
for seed in SEEDS:
 y00,y10,y01,y11=[next(r['normal']['sealed'] for r in rows if r['seed']==seed and r['arm']==arm) for arm in ARMS]
 c=dict(route_difference=y10-y01,policy_given_value0=y10-y00,policy_given_value1=y11-y01,value_given_policy0=y01-y00,value_given_policy1=y11-y10,interaction=y11-y10-y01+y00);contrasts.append(dict(seed=seed,**c))
 expected=f'| {seed} | '+' | '.join(f'{v*100:.2f}%' for v in (y00,y10,y01,y11))+f' | {c["route_difference"]*100:+.2f}个百分点 |';check('per_seed_table',expected in text,expected=expected)
means={k:float(np.mean([r[k] for r in contrasts])) for k in contrasts[0] if k!='seed'}
labels=[('policy_given_value0','价值未缩放时，策略缩放：Y10−Y00'),('policy_given_value1','价值已缩放时，策略缩放：Y11−Y01'),('value_given_policy0','策略未缩放时，价值缩放：Y01−Y00'),('value_given_policy1','策略已缩放时，价值缩放：Y11−Y10'),('interaction','交互：Y11−Y10−Y01+Y00')]
for key,label in labels:
 expected=f'| {label} | {means[key]*100:+.2f}个百分点 | '+'、'.join(f'{r[key]*100:+.2f}' for r in contrasts)+' |';check('all_predefined_auxiliary_contrasts',expected in text,expected=expected)
for key,label,decimals,percent in [('oracle','原生联合MAP参照',2,True),('Q','原生解析Q',2,True),('G','原生发送、贪心接收G',2,True),('N','贪心双方N',2,True),('H','地图联合条件熵（nats）',6,False),('codes','贪心使用完整码数',2,False)]:table(label,[agg[a]['protocol'][key] for a in ARMS],decimals,percent)
check('primary_mean_reported',f'−{abs(means["route_difference"])*100:.2f}个百分点' in text)
d_auc=(agg['policy_only_scale']['auc']['sealed']-agg['value_only_scale']['auc']['sealed'])*100
check('endpoint_and_AUC_opposite_direction_disclosed',means['route_difference']<0 and d_auc>0 and f'+{d_auc:.2f}个百分点' in text)
check('value_given_scaled_policy_four_positive',all(r['value_given_policy1']>0 for r in contrasts) and 'Y11−Y10四个来源均为正' in text)
check('interaction_heterogeneity_reported',sum(r['interaction']>0 for r in contrasts)==2 and '四来源两正两负' in text)
check('common30_natural_sentence','、'.join(f'{agg[a]["normal"]["common30"]*100:.2f}%' for a in ARMS) in text)
check('blank_and_shuffle_summary',all(abs(agg[a]['normal']['blank']-1/30)<1e-15 and abs(agg[a]['normal']['shuffle']-.033)<.0005 for a in ARMS) and '固定空消息均为3.33%' in text)
num=read(BATCH/'report_numbers.json')
for a in ARMS:
 for part in ('normal','auc'):
  for k,v in agg[a][part].items():
   reported=num['arms'][a][k]['common30'] if k in ('shuffle','blank') else num['arms'][a][part][k]
   check('root_report_numbers_independent_pooling',abs(v-reported)<1e-12,arm=a,part=part,key=k)
 for k,mode,rk in [('oracle','native','joint_oracle'),('Q','native','Q'),('G','native','G'),('N','greedy','G'),('H','native','joint_entropy'),('codes','greedy','used_messages')]:check('root_protocol_numbers_independent_pooling',abs(agg[a]['protocol'][k]-num['arms'][a]['protocol'][mode][rk])<1e-12,arm=a,key=k)
for k,v in means.items():check('root_contrast_numbers_independent',abs(v-num['contrast_means'][k])<1e-12,key=k)
execqa=read(BATCH/'audit_execution.json');clip=read(BATCH/'clip_activity_independent_qa.json');comp=read(BATCH/'independent_recount_comparison.json');display=read(BATCH/'display_revision.json')
check('execution_audit_counts',execqa['passed'] and execqa['check_count']==975860 and execqa['counts']['runs']==24 and execqa['counts']['training_communications']==29491200 and execqa['counts']['training_actions']==58982400)
check('clipping_prior_independent_all_history_audit',clip['passed'] and clip['check_count']==890 and clip['person_role_updates_checked']==460800)
check('reported_clipping_maxima','、'.join(f'{clip["summary"][a]["sender"]["max_norm"]:.6f}' for a in ARMS) in text)
check('clipping_narrow_zero_nonunit_result',all(clip['summary'][a][r]['active_count']==0 and clip['summary'][a][r]['min_coefficient']==1 for a,r in product(ARMS,('sender','receiver'))))
check('final_analysis_binding',comp['passed'] and comp['scalar_checks']==71936 and comp['analysis_sha256']==sha(BATCH/'branches_analysis.json') and comp['report_numbers_sha256']==sha(BATCH/'report_numbers.json'))
check('display_revision_numeric_invariance',display['passed'] and display['numeric_scalars_identical']==366147 and display['new_analysis_sha256']==sha(BATCH/'branches_analysis.json'))
curation_file=PROJECT/'paper_program/visual_confirmation_v1/pixel_workflow_003/curation_status_20260916.json';cur=read(curation_file)
check('pixel003_counts_and_not_model_confirmation',cur['cumulative']['usable_for_limited_workflow']==11 and cur['cumulative']['workflow_gap']==13 and cur['cumulative']['counts']==dict(apple=3,banana=3,orange=3,water=2) and cur['cumulative']['total_pending']==3 and cur['confirmation_pixels_accessed']==0 and cur['model_calls']==0)
phrases=['唯一主要终点是2400步sealed自然N的Y10−Y01','在任何开发训练前','n=4，不能把48次运行或照片数当独立来源','预定辅助比较','不是经等效检验证明的能力匹配','不能根据哪个更有利而替换主要结论','该分析明确为事后诊断','该结论限于本实现及记录的轨迹','没有证明基线更准确、降低了完整参数梯度方差','不能宣称两臂全程分布匹配','候选尚未训练','原生参照只能与对应Q/G比较','不能与9600世界主终点直接相减','不识别任何一般非语言能力的必要或充分条件','完成这一批不意味着满足ICLR论文的全部要求']
for phrase in phrases:check('reviewed_interpretation_boundary_present',phrase in text,phrase=phrase)
links=[]
for target in re.findall(r'\]\(([^)]+)\)',text):
 if target.startswith(('http:','https:')):continue
 path=Path(unquote(target));path=(path if path.is_absolute() else REPORT.parent/path).resolve();links.append(dict(path=str(path),exists=path.exists()));check('report_local_link',path.exists(),path=str(path))
threshold=PROJECT/'paper_program/证据与投稿门槛.md';threshold_links=[]
for target in re.findall(r'\]\(([^)]+)\)',threshold.read_text()):
 if target.startswith(('http:','https:')):continue
 path=Path(unquote(target));path=(path if path.is_absolute() else threshold.parent/path).resolve();threshold_links.append(dict(path=str(path),exists=path.exists()));check('threshold_local_link',path.exists(),path=str(path))
check('final_report_revision',sha(REPORT)==EXPECTED)
for f in (REPORT,threshold,curation_file,BATCH/'report_numbers.json',BATCH/'audit_execution.json',BATCH/'clip_activity_independent_qa.json',BATCH/'clip_activity_exploratory.json',BATCH/'independent_recount_comparison.json',BATCH/'display_revision.json',BATCH/'branches_analysis.json',ROOT/'固定执行方案.md',ROOT/'裁剪核查后的机制解释与候选.md'):hashes[str(f)]=sha(f)
result=dict(passed=all(x['passed'] for x in checks),report_sha256=sha(REPORT),analysis_sha256=sha(BATCH/'branches_analysis.json'),script_sha256=sha(__file__),check_count=len(checks),checks=checks,aggregate=agg,seed_rows=rows,seed_contrasts=contrasts,contrast_means=means,source_hashes=hashes,local_links=links,threshold_local_links=threshold_links,scientific_read_review=dict(primary_route_difference_fixed_before_development=True,auxiliary_four_positive_not_promoted_to_primary=True,primary_and_interaction_seed_heterogeneity_disclosed=True,old_rounded_equality_not_equivalence=True,clipping_result_explicitly_posthoc_and_trajectory_limited=True,neural_baseline_numerical_weight_and_feedback_remain_not_variance_proof=True,Adam_per_parameter_and_frozen_disjoint_upstream_preconditions_verified=True,common30_native_oracle_not_cross_policy_or_subgroup_sum=True,raw_vs_effective_h_aligned=True,baseline_permutation_only_unexecuted_candidate=True,new_visual_model_confirmation_not_completed=True,four_inherited_sources_not_forty_eight_independent_repetitions=True),scope='Independent raw endpoint pooling (normal/shuffle/blank), source-curve trapezoid AUC, previously independently audited protocol tables and raw message counts; all report tables/contrasts and local links. Existing execution/clipping/display receipts bound. No new training, model forward or external literature re-search. Scientific statements read in full; keyword checks only preserve review locations and do not establish scientific truth.')
out=BATCH/'audit_report_review.json'
if out.exists():out.rename(out.with_name('audit_report_review_'+sha(out)[:12]+'.json'))
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(passed=result['passed'],report_sha256=result['report_sha256'],checks=len(checks),failures=[x for x in checks if not x['passed']]),ensure_ascii=False,indent=2));assert result['passed']
