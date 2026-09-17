"""Independent numerical recount and linkage of endpoint probes to real trajectories."""
from collections import Counter
from itertools import permutations
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
BATCH=ROOT/'results/generalization_001'
MAPS=np.asarray(list(permutations(range(6),2)))
MATCHINGS={1:((0,1),(2,3),(4,5)),2:((0,2),(1,4),(3,5)),3:((0,3),(1,5),(2,4))}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}
def probs(x):
    x=np.array(x,dtype='float64');x-=x.max(axis=-1,keepdims=True)
    x=np.exp(x);x/=np.sum(x,axis=-1,keepdims=True)
    return x


def groups(p):
    a={v for e in MATCHINGS[p] for v in (e,e[::-1])}
    h={v for e in MATCHINGS[p%3+1] for v in (e,e[::-1])}
    return {'old':np.asarray([i for i,v in enumerate(MAPS) if tuple(v) not in a|h]),
            'added':np.asarray([i for i,v in enumerate(MAPS) if tuple(v) in a]),
            'sealed':np.asarray([i for i,v in enumerate(MAPS) if tuple(v) in h]),'all':np.arange(30)}


def main():
    data=json.loads((BATCH/'protocol_analysis.json').read_text())
    assert data['complete'] and len(data['runs'])==60
    counter=Counter();failures=[]
    def check(ok,name,context):
        counter[name]+=1
        if not bool(ok):failures.append(dict(name=name,context=context))
    bases={}
    cells=[]
    for record in data['runs']:
        seed,p,arm=record['seed'],record['partition'],record['arm']
        folder=BATCH/f's{seed}_p{p}_{arm}'
        traces={mode:load(folder/f'final_{mode}.npz') for mode in ('normal','shuffle','blank','erase_memory')}
        check(digest(folder/'final.pt')==record['source_sha256'],'source_checkpoint',folder.name)
        for d in record['directions']:
            who=d['scout'];label=[folder.name,who]
            z=load(BATCH/'protocol'/d['arrays'])
            check(digest(BATCH/'protocol'/d['arrays'])==d['sha256'],'array_hash',label)
            n=len(z['positions']);mids=z['positions'][:,0]*5+z['positions'][:,1]-(z['positions'][:,1]>z['positions'][:,0])
            check(n==480 and np.array_equal(np.bincount(mids,minlength=30),np.full(30,16)),'balanced_maps',label)
            f=probs(z['sender_first_logits']);s=probs(z['sender_second_logits']);r=probs(z['receiver_logits'])
            gp0=f.argmax(1);gp1=np.asarray([s[i,t].argmax() for i,t in enumerate(gp0)])
            msg=np.column_stack((gp0,gp1));decoder=z['receiver_logits'].argmax(-1)
            check(np.array_equal(msg,z['greedy_message']),'autoregressive_greedy',label)
            joint_message=np.stack([f[:,a]*s[:,a,b] for a in range(7) for b in range(7)],axis=1)
            check(np.allclose(joint_message.sum(1),1,rtol=0,atol=1e-12),'message_distribution_normalized',label)
            coverage=[];natural=[];expected=[];best=[];reward=[]
            for i,(food,water) in enumerate(z['positions']):
                success=(decoder[:,0]==food)&(decoder[:,1]==water)
                pair=r[:,0,food]*r[:,1,water]
                coverage.append(bool(success.any()));natural.append(bool(success[7*msg[i,0]+msg[i,1]]))
                expected.append(float(np.dot(joint_message[i],pair)));best.append(float(np.max(pair)))
                reward.append(float(np.dot(joint_message[i],.25*(r[:,0,food]+r[:,1,water])+.5*pair)))
            vectors={'N':np.asarray(natural),'U':np.asarray(coverage),'Q':np.asarray(expected),'B':np.asarray(best),'expected_mixed_reward':np.asarray(reward)}
            check((vectors['Q']<=vectors['B']+1e-12).all(),'pointwise_soft_upper_bound',label)
            count=np.zeros((49,30),dtype=np.int64)
            for mid,message in zip(mids,msg):count[7*message[0]+message[1],mid]+=1
            computed={}
            for name,pool in groups(p).items():
                mask=np.isin(mids,pool);den=int(mask.sum())
                values={k:float(v[mask].mean()) for k,v in vectors.items()}
                values['C_S']=sum(int(max(row[pool])) for row in count)/den
                saved=d['bounds'][name]
                check(saved['n']==den and saved['natural_correct']==int(vectors['N'][mask].sum()) and
                    all(abs(values[k]-saved[k])<1e-12 for k in values),'group_probability_and_count_recount',[*label,name])
                computed[name]=values
            cells.append(dict(seed=seed,arm=arm,groups=computed))
            if arm=='base':bases[seed,p,who]=z
            elif arm=='expand_receiver':
                b=bases[seed,p,who]
                check(all(np.array_equal(z[k],b[k]) for k in ('sender_first_logits','sender_second_logits','greedy_message')),'frozen_sender_full_probability',label)
            elif arm=='expand_sender':
                check(np.array_equal(z['receiver_logits'],bases[seed,p,who]['receiver_logits']),'frozen_receiver_logits',label)
            lookup={(int(mid),tuple(photo)):message for mid,photo,message in zip(mids,z['photo_ids'],msg)}
            for mode,t in traces.items():
                ix=np.flatnonzero(t['scout']==who)
                codes=7*t['delivered'][ix,0]+t['delivered'][ix,1]
                predicted=np.take_along_axis(decoder[codes],t['goals'][ix],axis=1)
                check(np.array_equal(predicted,t['place'][ix]),'decoder_replay',[*label,mode])
                counter['decoder_worlds_replayed']+=len(ix)
                if mode=='erase_memory':continue
                pairs=[]
                for j in ix:
                    pos=t['positions'][j];mid=int(pos[0]*5+pos[1]-(pos[1]>pos[0]))
                    key=(mid,tuple(t['photo_ids'][j]))
                    if key in lookup:pairs.append((j,lookup[key]))
                check(bool(pairs) and np.array_equal(t['sent'][[j for j,_ in pairs]],np.asarray([m for _,m in pairs])),
                      'sender_replay_on_probe_photos',[*label,mode])
                counter['sender_worlds_replayed']+=len(pairs)
    for cell in data['seed_cells']:
        selected=[v for v in cells if (v['seed'],v['arm'])==(cell['seed'],cell['arm'])]
        check(len(selected)==6,'six_within_seed_directions',[cell['seed'],cell['arm']])
        for name,values in cell['groups'].items():
            check(all(abs(np.mean([v['groups'][name][key] for v in selected])-val)<1e-12 for key,val in values.items()),
                  'within_seed_aggregation',[cell['seed'],cell['arm'],name])
    for arm,group in data['aggregate'].items():
        selected=[c for c in data['seed_cells'] if c['arm']==arm]
        check(len(selected)==4 and all(abs(np.mean([c['groups'][name][key] for c in selected])-v)<1e-12
            for name,values in group.items() for key,v in values.items()),'four_seed_aggregation',arm)
    for path,value in data['source_hashes'].items():check(digest(path)==value,'frozen_probe_source',path)
    report=dict(passed=not failures,counts=dict(counter),failures=failures,records=120,
        source_sha256=digest(__file__),protocol_sha256=digest(BATCH/'protocol_analysis.json'))
    (BATCH/'audit_protocol.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    assert report['passed']


if __name__=='__main__':main()
