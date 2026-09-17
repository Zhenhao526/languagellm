"""Independent NumPy recount of every saved private-readout evaluation."""
from pathlib import Path
from itertools import permutations,product
import argparse,json,hashlib
import numpy as np

MAPS=np.asarray(list(permutations(range(6),2)),np.int64)
MATCHINGS=(((0,1),(2,3),(4,5)),((0,2),(1,4),(3,5)),((0,3),(1,5),(2,4)))
METRICS=('J','single','Q','native_single','entropy','nll','exact_max_tie_rate')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def pools(partition):
    def group(k):
        pairs={x for a,b in MATCHINGS[k] for x in ((a,b),(b,a))}
        return np.asarray([i for i,x in enumerate(MAPS) if tuple(x) in pairs])
    added=group(partition-1);sealed=group(partition%3)
    old=np.asarray([i for i in range(30) if i not in set(added)|set(sealed)])
    return dict(old=old,added=added,sealed=sealed,all=np.arange(30),common30=np.arange(30))

def recount(z,p):
    logits=z['logits'].astype(np.float64);position=z['positions']
    shifted=logits-np.max(logits,axis=2,keepdims=True)
    logp=shifted-np.log(np.sum(np.exp(shifted),axis=2,keepdims=True));prob=np.exp(logp)
    good=logits.argmax(2)==position
    true_prob=np.take_along_axis(prob,position[:,:,None],axis=2)[:,:,0]
    true_logp=np.take_along_axis(logp,position[:,:,None],axis=2)[:,:,0]
    entropy=-np.sum(prob*logp,axis=2);ties=np.sum(logits==np.max(logits,axis=2,keepdims=True),axis=2)>1
    result={}
    for key,ids in pools(p).items():
        take=np.isin(z['map_ids'],ids)
        result[key]=dict(n=int(take.sum()),J=float(np.mean(np.all(good[take],axis=1))),
            single=float(np.mean(good[take])),Q=float(np.mean(true_prob[take,0]*true_prob[take,1])),
            native_single=float(np.mean(true_prob[take])),entropy=float(np.mean(entropy[take])),
            nll=float(-np.mean(true_logp[take])),correct_joint=int(np.sum(np.all(good[take],axis=1))),
            correct_goals=int(good[take].sum()),exact_max_tie_goal_count=int(ties[take].sum()),
            exact_max_tie_rate=float(ties[take].mean()))
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    root=args.out.resolve();done=read(root/'training_complete.json');inv=read(root/'invocation.json')
    assert done['status']=='complete'
    rows=[];checks=0;error=0.
    for name in done['runs']:
        folder=root/name;cfg=read(folder/'config.json');result=read(folder/'result.json');curve=read(folder/'curve.json')
        assert cfg['checkpoints']==[r['update'] for r in curve]
        points=[]
        for point in curve:
            path=folder/f"evaluation_{point['update']:04d}.npz";assert sha(path)==point['raw_sha256']
            with np.load(path,allow_pickle=False) as z:
                assert np.array_equal(z['positions'],MAPS[z['map_ids']])
                score=recount(z,cfg['partition'])
            for group,values in score.items():
                for key,value in values.items():
                    diff=abs(value-point['scores'][group][key]);checks+=1;error=max(error,diff)
                    assert diff<=2e-12+2e-12*abs(value),(name,point['update'],group,key,diff)
            points.append(dict(update=point['update'],scores=score))
        for group,values in points[-1]['scores'].items():
            for key,value in values.items():
                diff=abs(value-result['scores'][group][key]);checks+=1;error=max(error,diff)
                assert diff<=2e-12+2e-12*abs(value),(name,'final',group,key,diff)
        auc={group:{k:float(np.trapz([x['scores'][group][k] for x in points],[x['update'] for x in points])/cfg['updates'])
            for k in METRICS} for group in ('old','added','sealed','common30')}
        rows.append(dict(**{k:cfg[k] for k in ('seed','partition','direction','interface','signal')},curve=points,auc=auc))
    seed_rows=[];aggregate=[]
    for seed,interface,signal in product(inv['seeds'],inv['interfaces'],inv['signals']):
        children=[r for r in rows if (r['seed'],r['interface'],r['signal'])==(seed,interface,signal)]
        assert len(children)==len(inv['partitions'])*2
        curve=[]
        for i,t in enumerate(inv_times:= [p['update'] for p in children[0]['curve']]):
            scores={g:{k:float(np.mean([c['curve'][i]['scores'][g][k] for c in children])) for k in METRICS} for g in ('old','added','sealed','common30')}
            curve.append(dict(update=t,scores=scores))
        auc={g:{k:float(np.mean([c['auc'][g][k] for c in children])) for k in METRICS} for g in ('old','added','sealed','common30')}
        seed_rows.append(dict(seed=seed,interface=interface,signal=signal,curve=curve,auc=auc))
    for interface,signal in product(inv['interfaces'],inv['signals']):
        children=[r for r in seed_rows if (r['interface'],r['signal'])==(interface,signal)]
        curve=[]
        for i,t in enumerate(inv_times):
            curve.append(dict(update=t,scores={g:{k:float(np.mean([c['curve'][i]['scores'][g][k] for c in children])) for k in METRICS} for g in ('old','added','sealed','common30')}))
        aggregate.append(dict(interface=interface,signal=signal,curve=curve,
            auc={g:{k:float(np.mean([c['auc'][g][k] for c in children])) for k in METRICS} for g in ('old','added','sealed','common30')}))
    contrasts=[]
    for seed in inv['seeds']:
        lookup={(r['interface'],r['signal']):r for r in seed_rows if r['seed']==seed}
        values={f'{i}_{s}':lookup[i,s]['curve'][-1]['scores']['sealed']['J'] for i,s in product(inv['interfaces'],inv['signals'])}
        contrasts.append(dict(seed=seed,values=values,
            primary_reward_retained_minus_reset_scaled=values['retained_reward']-values['reset_scaled_reward'],
            secondary_ce_retained_minus_reset_scaled=values['retained_ce']-values['reset_scaled_ce'],
            secondary_reset_scaled_ce_minus_reward=values['reset_scaled_ce']-values['reset_scaled_reward'],
            secondary_retained_ce_minus_reward=values['retained_ce']-values['retained_reward']))
    primary=float(np.mean([r['primary_reward_retained_minus_reset_scaled'] for r in contrasts]))
    payload=dict(passed=True,checks=checks,max_absolute_comparison_error=error,formal=done['formal'],
        source_sha256=sha(__file__),training_receipt_sha256=sha(root/'training_complete.json'),runs=rows,
        seed_rows=seed_rows,aggregate=aggregate,contrasts=contrasts,primary_mean=primary,
        scope='Independent all saved evaluation logits, categorical metrics, curves/AUC and paired four-source summary; training updates audited separately.')
    dest=root/'independent_recount.json';assert not dest.exists()
    dest.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,checks=checks,max_error=error,primary_mean=primary,
        endpoints=[dict(interface=r['interface'],signal=r['signal'],scores=r['curve'][-1]['scores']) for r in aggregate]),ensure_ascii=False))

if __name__=='__main__':main()
