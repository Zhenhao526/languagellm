"""Compare independent v17 log/trace diagnostics without model or project imports."""
from pathlib import Path
import json,hashlib
import numpy as np
B=Path(__file__).resolve().parent/'results/baseline_001'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=read(B/'baseline_analysis.json');clip=read(B/'clip_activity.json');raw=read(B/'independent_baseline_diagnostics.json')
assert a['status']=='complete' and clip['passed'] and raw['passed']
errors=[];counts={};maximum=0.
def check(x,y,kind,where):
 global maximum
 error=abs(float(x)-float(y));maximum=max(maximum,error);counts[kind]=counts.get(kind,0)+1
 if not np.isfinite(error) or error>1e-10:errors.append(dict(where=where,analysis=x,independent=y,error=error))
key=lambda r:tuple(r[k] for k in ('seed','partition','arm','person','role','phase'))
lookup={key(r):r for r in a['diagnostics']['role_rows']}
assert len(lookup)==len(clip['rows'])
for r in clip['rows']:
 z=lookup[key(r)]
 for left,right in [('person_updates','count'),('active_count','active_count'),('rate','rate'),('max_norm','max_norm'),('mean_norm','mean_norm'),('min_coefficient','min_coefficient')]:check(z[left],r[right],'clip_person_phase',str(key(r))+left)
for arm,roles in clip['summary'].items():
 for role,r in roles.items():
  z=a['diagnostics']['summary'][arm]['all'][role]
  for k in ('person_updates','active_count','rate','max_norm','min_coefficient'):check(z[k],r[k],'clip_summary',arm+role+k)
ml={(r['seed'],r['partition'],r['person'],r['phase']):r for r in a['diagnostics']['manipulation_rows'] if r['available']}
assert len(ml)==len(raw['rows'])
for r in raw['rows']:
 k=(r['seed'],r['partition'],r['person'],r['phase']);z=ml[k];n=z['examples']/z['person_updates'];v=r['values']
 check(z['person_updates'],r['updates'],'manipulation_person_phase',str(k)+'updates')
 check(z['fixed_point_rate']*n,v['fixed_point_count']['mean'],'manipulation_person_phase',str(k)+'fixed')
 check(z['changed_value_rate']*n,v['changed_value_count']['mean'],'manipulation_person_phase',str(k)+'changed')
 for own,ind,stat in [('mean_original_std','original_std','mean'),('mean_abs_change','used_minus_original_mean_abs','mean'),('max_abs_change','used_minus_original_max_abs','maximum')]:check(z[own],v[ind][stat],'manipulation_person_phase',str(k)+own)
for phase,r in raw['summary'].items():
 z=a['diagnostics']['summary']['shuffled'][phase]['manipulation'];v=r['metrics']
 check(z['rms_change'],r['pooled_rms_change'],'manipulation_summary',phase+'pooled_rms')
 check(z['examples']/256,r['person_updates'],'manipulation_summary',phase+'updates')
 check(z['fixed_point_rate']*256,v['fixed_point_count']['mean'],'manipulation_summary',phase+'fixed')
 check(z['changed_value_rate']*256,v['changed_value_count']['mean'],'manipulation_summary',phase+'changed')
 for own,ind,stat in [('mean_original_std','original_std','mean'),('mean_abs_change','used_minus_original_mean_abs','mean'),('max_abs_change','used_minus_original_max_abs','maximum')]:check(z[own],v[ind][stat],'manipulation_summary',phase+own)
# The independent summary reports mean(batch RMS), whereas the analyzer reports
# pooled RMS. Recount squared RMS from saved logged values for the same support.
pooled={phase:[] for phase in ('all','entropy_on','entropy_off')};files={}
for run in a['runs']:
 if run['arm']!='shuffled':continue
 path=Path(run['path'])/'training.jsonl';files[str(path)]=sha(path);logs=[json.loads(line) for line in path.read_text().splitlines()]
 for who in (0,1):
  for phase in pooled:
   selected=[r for r in logs if phase=='all' or (r['update']<=2100)==(phase=='entropy_on')]
   values=[r['agents'][who]['sender_baseline']['used_minus_original_rms'] for r in selected]
   pooled[phase].extend(values);rms=float(np.sqrt(np.mean(np.square(values))))
   check(ml[run['seed'],run['partition'],who,phase]['rms_change'],rms,'pooled_rms_from_logs',str((run['seed'],run['partition'],who,phase)))
for phase,values in pooled.items():check(a['diagnostics']['summary']['shuffled'][phase]['manipulation']['rms_change'],np.sqrt(np.mean(np.square(values))),'pooled_rms_from_logs',phase)
check(len(a['diagnostics']['trace_checks'])*2,raw['raw_person_traces_checked'],'trace_scope','persons')
check(sum(r['person_updates'] for r in ml.values() if r['phase']=='all'),raw['person_permutations_checked'],'permutation_scope','count')
result=dict(passed=not errors,scalar_checks=sum(counts.values()),checks_by_category=counts,max_absolute_error=maximum,tolerance=1e-10,failures=errors,
 analysis_sha256=sha(B/'baseline_analysis.json'),independent_diagnostics_sha256=sha(B/'independent_baseline_diagnostics.json'),independent_clip_sha256=sha(B/'clip_activity.json'),comparison_source_sha256=sha(__file__),input_hashes=files,
 scope='All personal phase clipping rows, corresponding independently audited manipulation summary fields and 1/2101 trace scope. Pooled RMS is explicitly recomputed from all logged batch RMS values; it is not mean(batch RMS). No all-trajectory model/value replay.')
(B/'independent_diagnostics_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='input_hashes'},ensure_ascii=False));assert not errors
