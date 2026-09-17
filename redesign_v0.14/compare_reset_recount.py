"""Bind the independent raw recount to the analysis; no model or metric imports."""
from pathlib import Path
import hashlib,json,itertools
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/reset_001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
a=read(OUT/'reset_analysis.json');b=read(OUT/'independent_recount.json')
assert a['status']=='complete' and b['passed']
errors=[];counts={};max_error=0.
def check(x,y,category,where):
    global max_error
    err=abs(float(x)-float(y));max_error=max(max_error,err);counts[category]=counts.get(category,0)+1
    if err>1e-9:errors.append(dict(where=where,analysis=x,independent=y,error=err))
lookup={(r['seed'],r['partition'],r['arm']):r for r in a['runs']}
curves={}
for r in b['rows']:
    key=(r['seed'],r['partition'],r['arm']);v=lookup[key];tag=str(key)
    for mode,groups in r['modes'].items():
        for group,values in groups.items():
            z=v['final'][mode] if group=='common30' else v['final'][mode]['map_groups'][group]
            for k,k2 in [('J','both_accuracy'),('single','single_accuracy'),('reward','mean_reward'),('worlds','n')]:check(z[k2],values[k],'raw_endpoint',tag+mode+group+k)
    for t,directions in r['protocol'].items():
        entry=next(q for q in v['curve'] if q['update']==int(t))
        for d in (0,1):
            for policy,groups in directions[d].items():
                for group,vals in groups.items():
                    z=entry['protocol'][d]['metrics'][policy][group]
                    metrics=dict(Q=z['stochastic']['J'],G=z['greedy']['J'],joint_oracle=z['oracle_joint_map']['J'],
                        reward_oracle=z['oracle_mixed_reward']['mixed_reward'],joint_entropy=z['joint_conditional_entropy'],used_messages=z['positive_message_count'],worlds=z['world_count'])
                    for k,value in vals.items():check(metrics[k],value,'raw_protocol',tag+t+str(d)+policy+group+k)
    raw=read(Path(v['path'])/'curve.json');curves[key]=raw
    for trow in raw:
        target=next(q for q in v['curve'] if q['update']==trow['update'])['scores']['normal']
        for group in ('old','added','sealed'):
            for d in (0,1):
                for met in ('both_accuracy','single_accuracy','mean_reward','n'):
                    check(target['direction_groups'][d][group][met],trow['scores']['normal']['direction_groups'][d][group][met],'saved_learning_curve',tag+str(trow['update'])+group+str(d)+met)
cell_lookup={(c['seed'],c['arm']):c for c in a['summary']['seed_cells']}
independent={}
for seed,arm in itertools.product(a['seeds'],('retained','reset')):
    cell=cell_lookup[seed,arm];own={};per_time={}
    for t in a['times']:
        per_time[t]={}
        target=next(q for q in cell['curve'] if q['update']==t)['metrics']
        for g in ('old','added','sealed'):
            values=[]
            for p in a['partitions']:
                raw=next(q for q in curves[seed,p,arm] if q['update']==t)['scores']['normal']
                values.extend(x[g]['both_accuracy'] for x in raw['direction_groups'])
            per_time[t][f'normal.{g}.N']=float(np.mean(values))
            check(target[f'normal.{g}.N'],np.mean(values),'seed_learning_curve',str((seed,arm,t,g)))
        for g in ('old','added','sealed','common30'):
            for k,policy,met in [('native_joint_oracle','native','joint_oracle'),('Q','native','Q'),('G','native','G'),('N','greedy','G'),('joint_entropy','native','joint_entropy')]:
                vals=[r['protocol'][str(t)][d][policy][g][met] for r in b['rows'] if r['seed']==seed and r['arm']==arm for d in (0,1)]
                key=f'protocol.{g}.{k}';per_time[t][key]=float(np.mean(vals));check(target[key],np.mean(vals),'seed_protocol_curve',str((seed,arm,t,key)))
    own.update(per_time[a['times'][-1]])
    for g in ('old','added','sealed'):
        tt=a['times'];vv=[per_time[t][f'normal.{g}.N'] for t in tt]
        area=sum((tt[i+1]-tt[i])*(vv[i]+vv[i+1])/2 for i in range(len(tt)-1))/(tt[-1]-tt[0])
        own[f'auc.{g}.N']=area;check(cell['metrics'][f'auc.{g}.N'],area,'seed_auc',str((seed,arm,g)))
    independent[seed,arm]=own
    for t,vals in per_time.items():
        for k,v in vals.items():
            agg=next(q for q in a['summary']['aggregate'][arm]['curve'] if q['update']==t)['metrics'][k]
            allvals=[]
            for s in a['seeds']:
                cc=cell_lookup[s,arm];allvals.append(next(q for q in cc['curve'] if q['update']==t)['metrics'][k])
            check(agg,np.mean(allvals),'aggregate_curve',str((arm,t,k)))
for k in next(iter(independent.values())):
    deltas=[independent[s,'retained'][k]-independent[s,'reset'][k] for s in a['seeds']]
    for idx,s in enumerate(a['seeds']):check(a['summary']['seed_differences'][k][idx],deltas[idx],'paired_differences',str((s,k)))
    check(a['summary']['mean_differences'][k],np.mean(deltas),'paired_mean',k)
    check(a['summary']['difference_ranges'][k][0],min(deltas),'paired_range',k)
    check(a['summary']['difference_ranges'][k][1],max(deltas),'paired_range',k)
for x in b['primary_by_seed']:
    for arm in ('retained','reset'):check(cell_lookup[x['seed'],arm]['metrics']['normal.sealed.N'],x[arm],'root_primary',str((x['seed'],arm)))
    check(a['summary']['seed_differences']['normal.sealed.N'][a['seeds'].index(x['seed'])],x['delta'],'root_primary',str(x['seed']))
check(a['summary']['mean_differences']['normal.sealed.N'],b['primary_mean_difference'],'root_primary','mean')
result=dict(passed=not errors,scalar_checks=sum(counts.values()),checks_by_category=counts,max_absolute_error=max_error,tolerance=1e-9,failures=errors,
    analysis_sha256=sha(OUT/'reset_analysis.json'),independent_recount_sha256=sha(OUT/'independent_recount.json'),comparison_source_sha256=sha(__file__),
    scope='All independent endpoint and protocol scalars; saved learning curves; four source-seed aggregation and paired endpoint/AUC differences. No training, new inference or new estimand.')
(OUT/'independent_recount_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False));assert not errors
