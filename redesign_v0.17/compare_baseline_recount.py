"""Read-only final comparison of v17 raw recount, all seed metrics and the fixed contrast."""
from pathlib import Path
from itertools import product
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/baseline_001'
ARMS=('matched','shuffled')
GROUPS=('old','added','sealed')
MODES=('normal','shuffle','blank','stochastic','erase_memory')
WEIGHTS={'state_correspondence':{'matched':1,'shuffled':-1}}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def softmax(v):
    v=np.asarray(v,np.float64);v=np.exp(v-v.max(-1,keepdims=True));return v/v.sum(-1,keepdims=True)
a=read(OUT/'baseline_analysis.json');b=read(OUT/'independent_recount.json');numbers=read(OUT/'report_numbers.json')
assert a['status']=='complete' and b['passed'] and len(a['runs'])==len(b['rows'])==24
errors=[];counts={};max_error=0.;extra_inputs={}
def check(x,y,category,where):
    global max_error
    err=abs(float(x)-float(y));max_error=max(max_error,err);counts[category]=counts.get(category,0)+1
    if not np.isfinite(err) or err>1e-9:errors.append(dict(where=where,analysis=x,independent=y,error=err))
def mask_groups(mids,partition):
    maps=[(f,w) for f in range(6) for w in range(6) if f!=w]
    edges=[[(0,1),(2,3),(4,5)],[(0,2),(1,4),(3,5)],[(0,3),(1,5),(2,4)]]
    chosen=[]
    for j in (partition-1,partition%3):
        chosen.append([i for i,pair in enumerate(maps) if pair in edges[j] or pair[::-1] in edges[j]])
    added,sealed=chosen
    return dict(old=~np.isin(mids,added+sealed),added=np.isin(mids,added),sealed=np.isin(mids,sealed),common30=np.ones(len(mids),bool))
def native_rewards(path,partition):
    z=np.load(path);pi=(softmax(z['first_logits'])[:,:,None]*softmax(z['second_logits'])).reshape(-1,49)
    q=softmax(z['receiver_logits']);pos=z['positions']
    rf=q[:,0,pos[:,0]].T;rw=q[:,1,pos[:,1]].T
    expected=(pi*(.25*(rf+rw)+.5*rf*rw)).sum(1)
    extra_inputs[str(path)]=sha(path)
    return {g:float(expected[mask].mean()) for g,mask in mask_groups(z['map_ids'],partition).items()}
lookup={(r['seed'],r['partition'],r['arm']):r for r in a['runs']}
recount={(r['seed'],r['partition'],r['arm']):r for r in b['rows']}
curves={};rewards={}
for key,v in lookup.items():
    r=recount[key];tag=str(key);folder=Path(v['path'])
    for mode,groups in r['modes'].items():
        for group,values in groups.items():
            z=v['final'][mode] if group=='common30' else v['final'][mode]['map_groups'][group]
            for k,k2 in [('J','both_accuracy'),('single','single_accuracy'),('reward','mean_reward'),('worlds','n')]:check(z[k2],values[k],'raw_endpoint',tag+mode+group+k)
    for t,directions in r['protocol'].items():
        entry=next(q for q in v['curve'] if q['update']==int(t))
        rewards[key,int(t)]=[native_rewards(folder/f'protocol_{int(t):04d}_d{d}.npz',key[1]) for d in (0,1)]
        for d in (0,1):
            for policy,groups in directions[d].items():
                for group,vals in groups.items():
                    z=entry['protocol'][d]['metrics'][policy][group]
                    metrics=dict(Q=z['stochastic']['J'],G=z['greedy']['J'],joint_oracle=z['oracle_joint_map']['J'],reward_oracle=z['oracle_mixed_reward']['mixed_reward'],joint_entropy=z['joint_conditional_entropy'],used_messages=z['positive_message_count'],worlds=z['world_count'])
                    for k,value in vals.items():check(metrics[k],value,'raw_protocol',tag+t+str(d)+policy+group+k)
                    if policy=='native':check(z['stochastic']['mixed_reward'],rewards[key,int(t)][d][group],'extra_raw_reward',tag+t+str(d)+group)
    raw=read(folder/'curve.json');extra_inputs[str(folder/'curve.json')]=sha(folder/'curve.json');curves[key]=raw
    for row in raw:
        target=next(q for q in v['curve'] if q['update']==row['update'])['scores']['normal'];orig=row['scores']['normal']
        for group in GROUPS:
            for d in (0,1):
                for met in ('both_accuracy','single_accuracy','mean_reward','n'):check(target['direction_groups'][d][group][met],orig['direction_groups'][d][group][met],'saved_learning_curve',tag+str(row['update'])+group+str(d)+met)
        for met in ('both_accuracy','single_accuracy','mean_reward','n'):check(target[met],orig[met],'saved_learning_curve',tag+str(row['update'])+met)

cell_lookup={(c['seed'],c['arm']):c for c in a['summary']['seed_cells']};independent={};time_cells={}
for seed,arm in product(a['seeds'],ARMS):
    cell=cell_lookup[seed,arm];own={};per_time={};keys=[(seed,p,arm) for p in a['partitions']]
    for t in a['times']:
        values={};rawstats=[next(q for q in curves[k] if q['update']==t)['scores']['normal'] for k in keys]
        for g in GROUPS:
            for label,metric in [('N','both_accuracy'),('single','single_accuracy')]:values[f'normal.{g}.{label}']=float(np.mean([d[g][metric] for s in rawstats for d in s['direction_groups']]))
        values['normal.all.N']=float(np.mean([s['both_accuracy'] for s in rawstats]));values['normal.all.reward']=float(np.mean([s['mean_reward'] for s in rawstats]))
        for g in (*GROUPS,'common30'):
            for label,policy,met in [('native_joint_oracle','native','joint_oracle'),('Q','native','Q'),('G','native','G'),('N','greedy','G'),('native_reward_oracle','native','reward_oracle'),('joint_entropy','native','joint_entropy')]:
                values[f'protocol.{g}.{label}']=float(np.mean([recount[k]['protocol'][str(t)][d][policy][g][met] for k in keys for d in (0,1)]))
            values[f'protocol.{g}.native_reward']=float(np.mean([rewards[k,t][d][g] for k in keys for d in (0,1)]))
        for label,policy in [('native_positive_codes','native'),('greedy_used_codes','greedy')]:values[f'protocol.common30.{label}']=float(np.mean([recount[k]['protocol'][str(t)][d][policy]['common30']['used_messages'] for k in keys for d in (0,1)]))
        target=next(q for q in cell['curve'] if q['update']==t)['metrics'];assert set(target)==set(values)
        for k,value in values.items():check(target[k],value,'seed_curve_all_metrics',str((seed,arm,t,k)))
        per_time[t]=values;time_cells[seed,arm,t]=values
    own.update(per_time[a['times'][-1]])
    for g in GROUPS:
        tt=a['times'];vv=[per_time[t][f'normal.{g}.N'] for t in tt]
        own[f'auc.{g}.N']=sum((tt[i+1]-tt[i])*(vv[i]+vv[i+1])/2 for i in range(len(tt)-1))/(tt[-1]-tt[0])
    for mode in MODES:
        for g in (*GROUPS,'all'):
            for label,met in [('joint','J'),('reward','reward')]:own[f'intervention.{mode}.{g}.{label}']=float(np.mean([recount[k]['modes'][mode]['common30' if g=='all' else g][met] for k in keys]))
    assert set(own)==set(cell['metrics'])
    for k,value in own.items():check(cell['metrics'][k],value,'seed_endpoint_auc_all_metrics',str((seed,arm,k)))
    independent[seed,arm]=own
for arm in ARMS:
    for t in a['times']:
        target=next(q for q in a['summary']['aggregate'][arm]['curve'] if q['update']==t)['metrics']
        for k,value in target.items():check(value,np.mean([time_cells[s,arm,t][k] for s in a['seeds']]),'aggregate_curve_all_metrics',str((arm,t,k)))
    for k,value in a['summary']['aggregate'][arm]['metrics'].items():check(value,np.mean([independent[s,arm][k] for s in a['seeds']]),'aggregate_endpoint_auc_all_metrics',str((arm,k)))
for name,weights in WEIGHTS.items():
    target=a['summary']['contrasts'][name];assert target['weights']==weights
    for k in next(iter(independent.values())):
        deltas=[sum(w*independent[s,arm][k] for arm,w in weights.items()) for s in a['seeds']]
        for i,s in enumerate(a['seeds']):check(target['seed_differences'][k][i],deltas[i],'fixed_contrast_seed_all_metrics',str((name,s,k)))
        check(target['mean_differences'][k],np.mean(deltas),'fixed_contrast_mean_all_metrics',name+k)
        check(target['difference_ranges'][k][0],min(deltas),'fixed_contrast_range_all_metrics',name+k+'min')
        check(target['difference_ranges'][k][1],max(deltas),'fixed_contrast_range_all_metrics',name+k+'max')
for row in b['primary_by_seed']:
    i=a['seeds'].index(row['seed'])
    for arm in ARMS:check(cell_lookup[row['seed'],arm]['metrics']['normal.sealed.N'],row[arm],'root_primary',str((row['seed'],arm)))
    for name in WEIGHTS:check(a['summary']['contrasts'][name]['seed_differences']['normal.sealed.N'][i],row['contrasts'][name],'root_primary',str((row['seed'],name)))
for name,value in b['contrast_means'].items():check(a['summary']['contrasts'][name]['mean_differences']['normal.sealed.N'],value,'root_primary',name)
for arm,values in numbers['arms'].items():
    target=a['summary']['aggregate'][arm]['metrics']
    for mode in MODES:
        for group,value in values[mode].items():check(target[f'intervention.{mode}.{"all" if group=="common30" else group}.joint'],value,'report_numbers',arm+mode+group)
    for group,value in values['auc'].items():check(target[f'auc.{group}.N'],value,'report_numbers',arm+group+'auc')
    for label,policy,metric in [('native_joint_oracle','native','joint_oracle'),('Q','native','Q'),('G','native','G'),('N','greedy','G'),('joint_entropy','native','joint_entropy'),('greedy_used_codes','greedy','used_messages')]:check(target[f'protocol.common30.{label}'],values['protocol'][policy][metric],'report_numbers',arm+label)
assert set(a['summary']['contrasts'])==set(WEIGHTS)
result=dict(passed=not errors,scalar_checks=sum(counts.values()),checks_by_category=counts,max_absolute_error=max_error,tolerance=1e-9,failures=errors,
 analysis_sha256=sha(OUT/'baseline_analysis.json'),independent_recount_sha256=sha(OUT/'independent_recount.json'),report_numbers_sha256=sha(OUT/'report_numbers.json'),comparison_source_sha256=sha(__file__),extra_input_hashes=extra_inputs,
 scope='All independent raw endpoint and protocol values, saved learning curves, every analyzed metric in four source seeds, normalized AUC and the fixed matched-minus-shuffled contrast; raw NumPy expected reward supplements the independent recount. No training, inference, new estimand or result selection.')
(OUT/'independent_recount_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='extra_input_hashes'},ensure_ascii=False));assert not errors
