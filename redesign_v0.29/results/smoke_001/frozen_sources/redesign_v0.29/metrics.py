"""Finite evaluation, with prespecified support scopes and token interventions."""
from pathlib import Path
import numpy as np
import support


def probabilities(logits):
    z=np.asarray(logits,np.float64);z=z-z.max(-1,keepdims=True)
    p=np.exp(z);return p/p.sum(-1,keepdims=True)


def donor_indices(raw,p,arm):
    train=np.isin(raw['map_id'],support.groups(p,arm)['train18'])
    targets=np.flatnonzero(np.isin(raw['map_id'],support.groups(p,arm)['common_target6']))
    food=[];water=[]
    for i in targets:
        same=train&(raw['photo_ids']==raw['photo_ids'][i]).all(-1)&(raw['shown']==raw['shown'][i])
        f=np.flatnonzero(same&(raw['positions'][:,0]==raw['positions'][i,0]))
        w=np.flatnonzero(same&(raw['positions'][:,1]==raw['positions'][i,1]))
        assert len(f)==len(w)==3
        food.append(np.repeat(f,3));water.append(np.tile(w,3))
    return targets,np.asarray(food),np.asarray(water)


def recombination(raw,p,arm,permutations=None):
    targets,fd,wd=donor_indices(raw,p,arm)
    oldcodes=raw['tokens'][:,0]*7+raw['tokens'][:,1]
    acts=raw['receiver_logits'].argmax(-1)
    perms=np.arange(49)[None,:] if permutations is None else np.asarray(permutations)
    assert perms.ndim==2 and perms.shape[1]==49
    assert (np.sort(perms,axis=1)==np.arange(49)).all()
    remapped=perms[:,oldcodes]
    inverse=np.argsort(perms,axis=1)
    ff=remapped[:,fd];ww=remapped[:,wd]
    result={}
    for name,newcode in [('FW',7*(ff//7)+ww%7),('WF',7*(ww//7)+ff%7)]:
        decoded=acts[inverse[np.arange(len(perms))[:,None,None],newcode]]
        result[name]=(decoded==raw['positions'][targets][None,:,None,:]).all(-1).mean(-1)
    return dict(target_rows=targets,**result)


def social(raw,p,arm):
    token=raw['tokens'];codes=token[:,0]*7+token[:,1];rl=raw['receiver_logits'];acts=rl.argmax(-1)
    correct=acts[codes]==raw['positions'];rp=probabilities(rl);sp=probabilities(raw['sender_log_probs'])
    q=(sp*rp[:,0,raw['positions'][:,0]].T*rp[:,1,raw['positions'][:,1]].T).sum(-1)
    targets=raw['positions'];null=(acts[0]==targets).all(-1).astype(float)
    hist=np.bincount(codes,minlength=49)/len(codes)
    matched=(acts[None,:,:]==targets[:,None,:]).all(-1)
    shuffle=matched@hist
    rec=recombination(raw,p,arm)
    recvalues={k:np.full(len(codes),np.nan) for k in ('FW','WF')}
    for k in recvalues:recvalues[k][rec['target_rows']]=rec[k][0]
    result={}
    for scope,ids in support.groups(p,arm).items():
        selected=np.isin(raw['map_id'],ids);block={}
        for label,mask in [('pooled',selected),('food_only',selected&(raw['shown']==0)),('water_only',selected&(raw['shown']==1))]:
            c=correct[mask]
            local_hist=np.bincount(codes[mask],minlength=49)/int(mask.sum())
            block[label]=dict(n=int(mask.sum()),J=float(c.all(-1).mean()),food=float(c[:,0].mean()),water=float(c[:,1].mean()),Q=float(q[mask].mean()),
                blank_J=float(null[mask].mean()),shuffle_J=float(shuffle[mask].mean()),within_shuffle_J=float((matched[mask]@local_hist).mean()))
            if scope=='common_target6':
                for k,v in recvalues.items():block[label][f'recombine_{k}_J']=float(v[mask].mean())
        result[scope]=block
    return result


def save_null(raw,p,arm,path):
    permutations=np.load(Path(__file__).parent/'code_relabelings.npy')
    result=recombination(raw,p,arm,permutations)
    np.savez_compressed(path,**result)
