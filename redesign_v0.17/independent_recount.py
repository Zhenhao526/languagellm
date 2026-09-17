"""Independent NumPy recount: no model, runner or project metric imports."""
from pathlib import Path
from itertools import permutations,product
import argparse,hashlib,json
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def softmax(a):
    a=np.asarray(a,np.float64);a=a-a.max(-1,keepdims=True);p=np.exp(a)
    return p/p.sum(-1,keepdims=True)
def groups(p):
    maps=list(permutations(range(6),2))
    matches=[[(0,1),(2,3),(4,5)],[(0,2),(1,4),(3,5)],[(0,3),(1,5),(2,4)]]
    def ids(edges):return [i for i,t in enumerate(maps) if t in edges or t[::-1] in edges]
    added=ids(matches[p-1]);sealed=ids(matches[p%3])
    return dict(old=[i for i in range(30) if i not in added+sealed],added=added,sealed=sealed,common30=list(range(30)))

def protocol(path,p):
    z=np.load(path);pos=z['positions'];mids=z['map_ids']
    native=(softmax(z['first_logits'])[:,:,None]*softmax(z['second_logits'])).reshape(-1,49)
    assert np.allclose(native.sum(1),1,atol=1e-12)
    msg=z['greedy_message'];greedy=np.eye(49)[msg[:,0]*7+msg[:,1]]
    receiver=softmax(z['receiver_logits']);action=z['receiver_logits'].argmax(-1)
    output={}
    for mode,pmsg in [('native',native),('greedy',greedy)]:
        output[mode]={}
        for name,pool in groups(p).items():
            mask=np.isin(mids,pool);c=np.zeros((49,6,6))
            for k in np.flatnonzero(mask):c[:,pos[k,0],pos[k,1]]+=pmsg[k]/mask.sum()
            pm=c.sum((1,2));fm=c.sum(2);wm=c.sum(1)
            entropy=0.
            for m in range(49):
                positive=c[m]>0
                if pm[m]>0:entropy-=float((c[m][positive]*np.log(c[m][positive]/pm[m])).sum())
            q=float((c*receiver[:,0,:,None]*receiver[:,1,None,:]).sum())
            g=float(c[np.arange(49),action[:,0],action[:,1]].sum())
            oracle=float(c.reshape(49,36).max(1).sum())
            reward_oracle=float((.25*(fm[:,:,None]+wm[:,None,:])+.5*c).reshape(49,36).max(1).sum())
            output[mode][name]=dict(Q=q,G=g,joint_oracle=oracle,reward_oracle=reward_oracle,
                joint_entropy=entropy,used_messages=int((pm>0).sum()),worlds=int(mask.sum()))
    return output

def main():
    ap=argparse.ArgumentParser();ap.add_argument('batch',type=Path);args=ap.parse_args();batch=args.batch.resolve()
    inv=read(batch/'invocation.json');seeds=inv['args']['seeds'];parts=inv['args']['partitions'];arms=['matched','shuffled'];scaled_batch=Path(inv['args']['scaled_batch'])
    updates=inv['args']['social_updates'];assert read(batch/'training_complete.json')['status']=='complete'
    folders=[(scaled_batch/f'social_s{s}_p{p}_reset_scaled' if a=='matched' else batch/f'social_s{s}_p{p}_both_sender_baseline_shuffle',a) for s,p,a in product(seeds,parts,arms)]
    assert all((f/'result.json').exists() for f,a in folders)
    rows=[];hashes={};value_checks=0
    for folder,arm in folders:
        cfg=read(folder/'config.json');p=cfg['partition'];row={k:cfg[k] for k in ('seed','partition','arm')};row['arm']=arm;row['modes']={}
        for mode in ('normal','shuffle','blank','stochastic','erase_memory'):
            path=folder/f'final_{mode}.npz';hashes[str(path)]=sha(path);z=np.load(path)
            target=np.take_along_axis(z['positions'],z['goals'],axis=1)
            actual=np.take_along_axis(z['menu'],z['action'][:,:,None],axis=2).squeeze(-1)
            good=(target==actual);assert np.array_equal(good,z['successes'])
            reward=.25*good.sum(1)+.5*good.all(1);assert np.array_equal(reward,z['reward'])
            mids=np.array([list(permutations(range(6),2)).index(tuple(x)) for x in z['positions']])
            row['modes'][mode]={}
            saved=read(folder/'result.json')['scores'][mode]
            for group,pool in groups(p).items():
                mask=np.isin(mids,pool)
                values=dict(J=float(good[mask].all(1).mean()),single=float(good[mask].mean()),reward=float(reward[mask].mean()),worlds=int(mask.sum()))
                original=saved if group=='common30' else saved['map_groups'][group]
                for k,k2 in [('J','both_accuracy'),('single','single_accuracy'),('reward','mean_reward'),('worlds','n')]:
                    assert abs(values[k]-original[k2])<1e-12;value_checks+=1
                row['modes'][mode][group]=values
        row['protocol']={}
        for step in cfg['checkpoints']:
            row['protocol'][str(step)]=[]
            saved=read(folder/f'protocol_{step:04d}.json')
            for d in (0,1):
                path=folder/f'protocol_{step:04d}_d{d}.npz';hashes[str(path)]=sha(path);values=protocol(path,p)
                for mode in ('native','greedy'):
                    for group,r in values[mode].items():
                        other=saved[d]['metrics'][mode][group]
                        targets=dict(Q=other['stochastic']['J'],G=other['greedy']['J'],joint_oracle=other['oracle_joint_map']['J'],
                            reward_oracle=other['oracle_mixed_reward']['mixed_reward'],joint_entropy=other['joint_conditional_entropy'],
                            used_messages=other['positive_message_count'],worlds=other['world_count'])
                        for k,v in r.items():assert abs(v-targets[k])<1e-10;value_checks+=1
                row['protocol'][str(step)].append(values)
        rows.append(row)
    paired=[]
    for seed in seeds:
        means={a:float(np.mean([r['modes']['normal']['sealed']['J'] for r in rows if r['seed']==seed and r['arm']==a])) for a in arms}
        contrasts=dict(state_correspondence=means['matched']-means['shuffled'])
        paired.append(dict(seed=seed,**means,delta=contrasts['state_correspondence'],contrasts=contrasts))
    result=dict(passed=True,combined_social_runs=len(rows),new_social_runs=len(rows)//2,reused_social_runs=len(rows)//2,checked_scalar_values=value_checks,rows=rows,primary_by_seed=paired,
        primary_mean_difference=float(np.mean([x['delta'] for x in paired])),contrast_means={k:float(np.mean([x['contrasts'][k] for x in paired])) for k in paired[0]['contrasts']},source_hashes=hashes,contrast='Primary state_correspondence: matched baseline minus shuffled baseline; four inherited source seeds; old/added and process outcomes auxiliary',recount_source_sha256=sha(__file__),
        scope='NumPy-only direct recorded-world and full-protocol recount; no training replay or new model inference')
    (batch/'independent_recount.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_hashes')}))

if __name__=='__main__':main()
