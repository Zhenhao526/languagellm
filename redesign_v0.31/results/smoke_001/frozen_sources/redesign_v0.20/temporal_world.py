"""Two visual observations of a single resource relocation; no linguistic input."""
from functools import lru_cache
import itertools
import numpy as np
import torch

MAPS=np.asarray(list(itertools.permutations(range(6),2)),np.int64)
MATCHINGS=(((0,1),(2,3),(4,5)),((0,2),(1,4),(3,5)),((0,3),(1,5),(2,4)))

def partition(p):
    assert p in (1,2,3)
    def group(i):
        pairs={pair for a,b in MATCHINGS[i] for pair in ((a,b),(b,a))}
        return np.asarray([i for i,row in enumerate(MAPS) if tuple(row) in pairs],np.int64)
    added=group(p-1);sealed=group(p%3)
    return dict(old=np.setdiff1d(np.arange(30),np.r_[added,sealed]),added=added,sealed=sealed,common30=np.arange(30))

@lru_cache(None)
def events(p,training=False):
    old=partition(p)['old'];targets=old if training else np.arange(30)
    rows=[]
    for source,target,mover in itertools.product(old,targets,(0,1)):
        a,b=MAPS[source],MAPS[target]
        if a[1-mover]==b[1-mover] and a[mover]!=b[mover]:rows.append((source,target,mover))
    arr=np.asarray(rows,np.int64)
    assert arr.shape==((72 if training else 144),3)
    return arr

def balanced_events(p):
    rows=events(p)
    repeats=np.where(np.isin(rows[:,1],partition(p)['old']),3,2)
    return np.repeat(rows,repeats,axis=0)

def rng(seed,p,who,purpose,step):
    return np.random.default_rng(np.random.SeedSequence([20020,seed,p,who,purpose,step]))

def init_seed(seed,p,who):
    return int(np.random.SeedSequence([20020,seed,p,who,0,0]).generate_state(1,dtype=np.uint64)[0]>>np.uint64(1))

def fixture(bank,seed,p,who,step,batch=256):
    random=rng(seed,p,who,1,step);eid=random.integers(72,size=batch);row=events(p,True)[eid]
    ids=np.column_stack([random.choice(bank.pools['train',k],batch) for k in (0,1)])
    return dict(event_id=eid,source_map=row[:,0],target_map=row[:,1],mover=row[:,2],
        photo_ids=ids,positions=MAPS[row[:,1]],goals=random.integers(2,size=batch),
        action_uniform=rng(seed,p,who,2,step).random((batch*2,1)).astype(np.float32))

def evaluation_worlds(bank,p):
    rows=balanced_events(p)
    photos=np.asarray(list(itertools.product(*[np.sort(bank.pools['test',k])[:4] for k in (0,1)])),np.int64)
    repeat=np.repeat(rows,16,axis=0)
    return dict(source_map=repeat[:,0],target_map=repeat[:,1],mover=repeat[:,2],positions=MAPS[repeat[:,1]],
        photo_ids=np.tile(photos,(len(rows),1)))

def full_frame(projected,positions,ids):
    n=len(ids);out=projected.new_zeros((n,6,64));exists=projected.new_zeros((n,6));rows=torch.arange(n)
    for k in (0,1):
        sites=torch.from_numpy(positions[:,k]);out[rows,sites]=projected[torch.from_numpy(ids[:,k])];exists[rows,sites]=1.
    return torch.cat((out.flatten(1),exists),-1)

def frames(projected,world,mode):
    assert mode in ('immediate','delayed')
    ids=world['photo_ids'];n=len(ids)
    first=full_frame(projected,MAPS[world['source_map']],ids)
    if mode=='immediate':second=full_frame(projected,world['positions'],ids)
    else:
        rows=torch.arange(n);m=world['mover'];dest=world['positions'][np.arange(n),m]
        pixels=projected.new_zeros((n,6,64));exists=projected.new_zeros((n,6));sites=torch.from_numpy(dest)
        pixels[rows,sites]=projected[torch.from_numpy(ids[np.arange(n),m])];exists[rows,sites]=1.
        second=torch.cat((pixels.flatten(1),exists),-1)
    bits=projected.new_ones((n,2));bits[:,1]=float(mode=='immediate')
    return torch.stack((first,second),1),bits

def training_inputs(projected,world):
    a,ab=frames(projected,world,'immediate');b,bb=frames(projected,world,'delayed')
    return torch.cat((a,b)),torch.cat((ab,bb)),np.tile(world['goals'],2),np.tile(world['positions'],(2,1))

def history_pairs(actions,world,mask):
    """Both stationary predictions must be correct for two distinct histories.

    Identical-current-event contexts use the same photo pair. Pairs with the
    same stationary target are excluded; multiplicities are exact eval weights.
    """
    selected=np.flatnonzero(mask);groups={}
    for r in selected:
        m=int(world['mover'][r]);dest=int(world['positions'][r,m]);station=int(world['positions'][r,1-m])
        key=(m,dest,*map(int,world['photo_ids'][r]))
        classes=groups.setdefault(key,{})
        count,correct=classes.get(station,(0,0))
        classes[station]=(count+1,correct+int(actions[r,1-m]==station))
    denominator=numerator=0
    for classes in groups.values():
        for a,b in itertools.combinations(classes.values(),2):
            denominator+=a[0]*b[0];numerator+=a[1]*b[1]
    assert denominator>0
    return dict(history_pair_J=float(numerator/denominator),history_pair_correct=numerator,history_pairs=denominator)

def metrics(logits,world,p):
    raw=np.asarray(logits,np.float64);z=raw-raw.max(-1,keepdims=True)
    lp=z-np.log(np.exp(z).sum(-1,keepdims=True));prob=np.exp(lp);actions=raw.argmax(-1)
    target=world['positions'];correct=actions==target;rows=np.arange(len(raw));m=world['mover']
    tp=np.take_along_axis(prob,target[...,None],-1)[...,0]
    tlp=np.take_along_axis(lp,target[...,None],-1)[...,0];entropy=-(prob*lp).sum(-1)
    out={}
    for group,maps in partition(p).items():
        mask=np.isin(world['target_map'],maps);good=correct[mask]
        out[group]=dict(n=int(mask.sum()),J=float(good.all(-1).mean()),single=float(good.mean()),
            moved=float(correct[rows,m][mask].mean()),stationary=float(correct[rows,1-m][mask].mean()),
            Q=float(tp[mask].prod(-1).mean()),entropy=float(entropy[mask].mean()),nll=float(-tlp[mask].mean()),
            exact_max_tie_rate=float(((raw==raw.max(-1,keepdims=True)).sum(-1)>1)[mask].mean()),
            correct_joint=int(good.all(-1).sum()),correct_goals=int(good.sum()))
        # Each matching alone has no distinct stationary target for a fixed last event.
        if group in ('old','common30'):out[group].update(history_pairs(actions,world,mask))
        else:out[group].update(history_pair_J=None,history_pairs=0,history_pair_correct=0)
    return out

def self_test():
    receipts=[]
    for p in (1,2,3):
        r=balanced_events(p);assert r.shape==(360,3)
        assert np.array_equal(np.bincount(r[:,1],minlength=30),np.full(30,12))
        assert np.array_equal(np.bincount(r[:,0],minlength=30)[partition(p)['old']],np.full(18,20))
        assert np.array_equal(np.bincount(r[:,2]),[180,180])
        groups={}
        for source,target,m in r:
            a,b=MAPS[source],MAPS[target]
            assert a[1-m]==b[1-m] and a[m]!=b[m] and b[m] not in a
            groups.setdefault((int(m),int(b[m])),[]).append(int(b[1-m]))
        for (m,d),station in groups.items():
            count=np.bincount(station,minlength=6);assert count[d]==0 and np.all(count[np.arange(6)!=d]==6)
        receipts.append(dict(partition=p,train_events=72,eval_unique_events=144,weighted_eval_events=360,
            common30_current_only_J_upper=.2,old_current_only_J_upper=1/3,matching_only_current_J_upper=1.))
    return dict(passed=True,partitions=receipts,scope='Exact transition support and balanced last-event ambiguity; no model training')

if __name__=='__main__':
    import json
    print(json.dumps(self_test()))
