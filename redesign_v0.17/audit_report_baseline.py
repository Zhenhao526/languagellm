"""Independent raw pooling plus fixed-report numerical, link and scientific read review.
No training/model forward. Protocol tables inherit the completed independent execution audit.
"""
from pathlib import Path
from itertools import permutations, product
from urllib.parse import unquote
import hashlib, json, re
import numpy as np
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;BATCH=ROOT/'results/baseline_001'
REPORT=BATCH/'发送基线与场景对应关系的通信形成研究报告.md';EXPECTED='6807c27159c5b110bfeee384ceb011eb8eff12a9d360a5d0ea23c1ce5df77136'
SEEDS=(31101,31102,31103,31104);ARMS=('matched','shuffled')
MAPS=np.array(list(permutations(range(6),2)));MATCHINGS=(((0,1),(2,3),(4,5)),((0,2),(1,4),(3,5)),((0,3),(1,5),(2,4)))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def held(k):return np.array([i for i,x in enumerate(MAPS) if tuple(x) in {q for pair in MATCHINGS[k] for q in (pair,pair[::-1])}])
def groups(p):
 a=held(p-1);s=held(p%3);return dict(old=np.setdiff1d(np.arange(30),np.r_[a,s]),added=a,sealed=s)
def folder(seed,p,arm):
 if arm=='matched':return PROJECT/f'redesign_v0.15/results/scaled_001/social_s{seed}_p{p}_reset_scaled'
 return BATCH/f'social_s{seed}_p{p}_both_sender_baseline_shuffle'
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
for part,key,label in [('normal','old','old自然终点'),('normal','added','added自然终点'),('normal','sealed','sealed自然终点（主要）'),('auc','old','old全程AUC'),('auc','added','added全程AUC'),('auc','sealed','sealed全程AUC')]:table(label,[agg[a][part][key] for a in ARMS])
contrasts=[]
for seed in SEEDS:
 x,y=[next(r['normal']['sealed'] for r in rows if r['seed']==seed and r['arm']==arm) for arm in ARMS]
 contrasts.append(dict(seed=seed,matched=x,shuffled=y,difference=x-y))
 expected=f'| {seed} | {100*x:.2f}% | {100*y:.2f}% | {100*(x-y):+.2f}个百分点 |';check('per_seed_table',expected in text,expected=expected)
mean=float(np.mean([r['difference'] for r in contrasts]));check('primary_rounded_difference',f'−{abs(mean)*100:.2f}个百分点' in text and mean<0)
check('primary_two_positive_two_negative',sum(r['difference']>0 for r in contrasts)==2 and '两正两负' in text)
for key,label,decimals,percent in [('oracle','原生联合MAP参照',2,True),('Q','原生解析Q',2,True),('G','原生发送、贪心接收G',2,True),('N','贪心双方N',2,True),('H','地图联合条件熵（nats）',6,False),('codes','贪心使用完整码数',2,False)]:table(label,[agg[a]['protocol'][key] for a in ARMS],decimals,percent)
auc_delta=agg['matched']['auc']['sealed']-agg['shuffled']['auc']['sealed'];check('opposite_AUC_auxiliary_direction',auc_delta>0 and f'+{auc_delta*100:.2f}个百分点' in text)
for part in ('normal','auc'):
 delta=agg['matched'][part]['old']-agg['shuffled'][part]['old'];check('old_outcome_auxiliary_difference',delta<0 and f'{abs(delta)*100:.2f}个百分点' in text,part=part,difference=delta)
for mode,phrase in [('common30','正常通信成功率为'),('shuffle','打乱收到的消息后为')]:
 expected=phrase+'和'.join(f'{100*agg[a]["normal"][mode]:.2f}%' for a in ARMS);check('normal_and_shuffle_sentence',expected in text,expected=expected)
check('blank_summary',all(abs(agg[a]['normal']['blank']-1/30)<1e-15 for a in ARMS) and '固定空消息均3.33%' in text)
num=read(BATCH/'report_numbers.json')
for a in ARMS:
 for part in ('normal','auc'):
  for k,v in agg[a][part].items():
   reported=num['arms'][a][k]['common30'] if k in ('shuffle','blank') else num['arms'][a][part][k]
   check('independent_pooling_root_report_numbers',abs(v-reported)<1e-12,arm=a,part=part,key=k)
 for k,mode,rk in [('oracle','native','joint_oracle'),('Q','native','Q'),('G','native','G'),('N','greedy','G'),('H','native','joint_entropy'),('codes','greedy','used_messages')]:check('protocol_root_report_numbers',abs(agg[a]['protocol'][k]-num['arms'][a]['protocol'][mode][rk])<1e-12,arm=a,key=k)
check('primary_root_numbers',abs(mean-num['contrast_means']['state_correspondence'])<1e-12)
execution=read(BATCH/'audit_execution.json');diagnostic=read(BATCH/'independent_baseline_diagnostics.json');clip=read(BATCH/'clip_activity.json');comp=read(BATCH/'independent_recount_comparison.json');dc=read(BATCH/'independent_diagnostics_comparison.json');display=read(BATCH/'display_revision.json')
check('execution_counts',execution['passed'] and execution['check_count']==778759 and execution['counts']['runs']==12 and execution['counts']['personal_permutations_reconstructed']==57600 and execution['counts']['training_communications']==14745600 and execution['counts']['training_actions']==29491200)
check('baseline_diagnostic_scope_and_counts',diagnostic['passed'] and diagnostic['check_count']==346284 and diagnostic['person_permutations_checked']==57600 and diagnostic['raw_person_traces_checked']==48)
metrics=diagnostic['summary']['all']['metrics'];diagnostic_numbers={
 'average_fixed_indices':(metrics['fixed_point_count']['mean'],5),
 'fixed_percent':(metrics['fixed_point_count']['mean']/256*100,5),
 'changed_values':(metrics['changed_value_count']['mean'],3),
 'changed_percent':(metrics['changed_value_count']['mean']/256*100,5),
 'original_std':(metrics['original_std']['mean'],6),
 'mean_abs_change':(metrics['used_minus_original_mean_abs']['mean'],6),
 'mean_batch_rms':(metrics['used_minus_original_rms']['mean'],6),
 'pooled_rms':(diagnostic['summary']['all']['pooled_rms_change'],6)}
for key,(value,places) in diagnostic_numbers.items():check('manipulation_number_exact_format',f'{value:.{places}f}' in text,key=key,value=value)
check('clip_all_two_arms_roles',clip['passed'] and all(clip['summary'][a][role]['person_updates']==57600 and clip['summary'][a][role]['active_count']==0 and clip['summary'][a][role]['min_coefficient']==1 for a,role in product(ARMS,('sender','receiver'))))
for a,role in product(ARMS,('sender','receiver')):check('clipping_maximum_in_report',f'{clip["summary"][a][role]["max_norm"]:.6f}' in text,arm=a,role=role)
check('statistical_comparison_final_bindings',comp['passed'] and comp['scalar_checks']==34824 and comp['analysis_sha256']==sha(BATCH/'baseline_analysis.json') and comp['report_numbers_sha256']==sha(BATCH/'report_numbers.json'))
check('diagnostic_comparison_final_bindings',dc['passed'] and dc['scalar_checks']==2278 and dc['analysis_sha256']==sha(BATCH/'baseline_analysis.json') and dc['independent_diagnostics_sha256']==sha(BATCH/'independent_baseline_diagnostics.json') and dc['independent_clip_sha256']==sha(BATCH/'clip_activity.json'))
check('unchanged_display_data_and_source',display['passed'] and display['numeric_scalars_identical']==185967 and display['analysis_bytes_identical'] and display['source_bytes_identical'] and display['report_bytes_identical'] and display['analysis_sha256']==sha(BATCH/'baseline_analysis.json'))
# The bibliography is another agent's primary-source review; bind local content,
# do not misrepresent this report checker as an independent new literature search.
water=PROJECT/'paper_program/visual_confirmation_v2_water';v2=read(water/'status.json');vf=read(water/'feasibility_summary.json')
check('v2_metadata_counts',v2['status']=='metadata_collection_complete' and v2['actual_http_requests']==33 and v2['unique_frame_files']==623 and v2['metadata_count']==300 and v2['provisionally_eligible_records']==106 and v2['provisionally_eligible_clusters']==79)
check('v2_no_pixel_model_confirmation',v2['pixel_requests']==v2['image_views']==v2['model_calls']==v2['confirmation_pixel_access']==0 and not v2['allocations_performed'])
check('v2_provisional_nonbase_category_and_acceptance_boundary',len(vf['provisional_representatives_supported_only_by_nonbase_categories'])==24 and not vf['manual_identity_and_scene_acceptance_completed'] and not vf['pixel_acceptance_completed'] and vf['threshold80_is_unvalidated_proposal'])
phrases=['唯一主要比较“匹配减置换”','n=4；24个新旧运行不是24个独立来源','不能证明两种条件等效或对应关系完全无用','均匀置换允许固定点和重复值','不声称两臂全程基线分布或拟合速度保持一致','matched旧日志没有完整value幅度摘要','本轮在训练前即指定这些诊断','其他更新的数值幅度来自日志摘要','不同于平均批RMS','实际裁剪函数仍被调用','不跨政策直接减N','不拼接子组最优表','不能转而宣布全局基线或时间变化已经被证实','该候选尚未执行','不包含随机熵项梯度','79不是已验收水图或79位独立摄影者','不能描述为“只差一张”','本批完成是检验并限制强解释的开发进展']
for phrase in phrases:check('read_review_interpretation_boundaries',phrase in text,phrase=phrase)
links=[]
threshold=PROJECT/'paper_program/证据与投稿门槛.md'
for source in (REPORT,threshold):
 for target in re.findall(r'\]\(([^)]+)\)',source.read_text()):
  if target.startswith(('http:','https:')):continue
  path=Path(unquote(target));path=(path if path.is_absolute() else source.parent/path).resolve();links.append(dict(source=str(source),path=str(path),exists=path.exists()));check('local_link',path.exists(),source=str(source),path=str(path))
check('final_report_sha',sha(REPORT)==EXPECTED)
for path in (REPORT,threshold,water/'status.json',water/'feasibility_summary.json',water/'元数据可行性报告.md',ROOT/'文献定位与后续.md',BATCH/'report_numbers.json',BATCH/'baseline_analysis.json',BATCH/'audit_execution.json',BATCH/'independent_recount_comparison.json',BATCH/'independent_baseline_diagnostics.json',BATCH/'independent_diagnostics_comparison.json',BATCH/'clip_activity.json',BATCH/'display_revision.json'):hashes[str(path)]=sha(path)
result=dict(passed=all(x['passed'] for x in checks),report_sha256=sha(REPORT),analysis_sha256=sha(BATCH/'baseline_analysis.json'),script_sha256=sha(__file__),check_count=len(checks),checks=checks,aggregate=agg,seed_rows=rows,primary_seed_differences=contrasts,primary_mean=mean,diagnostic_numbers=diagnostic_numbers,source_hashes=hashes,local_links=links,scientific_read_review=dict(fixed_primary_matched_minus_shuffled=True,opposite_AUC_not_promoted=True,heterogeneity_and_no_equivalence_disclosed=True,manipulation_actual_values_not_only_indices=True,permutation_does_not_remove_all_state_information=True,within_batch_multiset_not_cross_arm_distribution_matching=True,logged_intensity_and_two_raw_traces_separated=True,mean_batch_RMS_and_pooled_RMS_distinguished=True,clip_prespecified_this_round_and_unit_multiplication_not_no_function_call=True,unexecuted_conditional_score_variance_diagnostic_not_full_update_variance=True,common_support_oracle_policy_boundary=True,v2_metadata_candidates_not_accepted_pixels_or_authors=True,provisional_79_not_only_one_missing=True,four_inherited_seeds_not_independent_confirmation=True),scope='Independent raw normal/shuffle/blank outcome pooling, saved-curve AUC, previous independently audited protocol table pooling and raw message counts. Completed execution, diagnostic and numeric-comparison receipts bound; no new model inference or full training replay. Main report read in full and all local links checked. Local literature review is bound but not independently re-searched here. v2 metadata status remains no pixel or model confirmation.')
out=BATCH/'audit_report_review.json'
if out.exists():out.rename(out.with_name('audit_report_review_'+sha(out)[:12]+'.json'))
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],report_sha256=result['report_sha256'],checks=len(checks),failures=[c for c in checks if not c['passed']]),ensure_ascii=False,indent=2));assert result['passed']
