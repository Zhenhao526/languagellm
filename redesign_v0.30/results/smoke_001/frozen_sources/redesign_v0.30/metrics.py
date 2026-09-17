"""Finite evaluation, with prespecified support scopes and token interventions."""
from pathlib import Path
import numpy as np
import support


def probabilities(logits):
    z=np.asarray(logits,np.float64);z=z-z.max(-1,keepdims=True)
    p=np.exp(z);return p/p.sum(-1,keepdims=True)


def donor_indices(raw,p,arm):
    train=np.isin(raw['map_id'],support.groups(p,arm)['train12'])
    targets=np.flatnonzero(np.isin(raw['map_id'],support.groups(p,arm)['target12']))
    food=[];water=[]
    for i in targets:
        same=train&(raw['photo_ids']==raw['photo_ids'][i]).all(-1)&(raw['shown']==raw['shown'][i])
        f=np.flatnonzero(same&(raw['positions'][:,0]==raw['positions'][i,0]))
        w=np.flatnonzero(same&(raw['positions'][:,1]==raw['positions'][i,1]))
        assert len(f)==len(w)==2
        food.append(np.repeat(f,2));water.append(np.tile(w,2))
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
    info=information(raw,p,arm)
    result={}
    for scope,ids in support.groups(p,arm).items():
        selected=np.isin(raw['map_id'],ids);block={}
        for label,mask in [('pooled',selected),('food_only',selected&(raw['shown']==0)),('water_only',selected&(raw['shown']==1))]:
            c=correct[mask]
            local_hist=np.bincount(codes[mask],minlength=49)/int(mask.sum())
            block[label]=dict(n=int(mask.sum()),J=float(c.all(-1).mean()),food=float(c[:,0].mean()),water=float(c[:,1].mean()),Q=float(q[mask].mean()),
                blank_J=float(null[mask].mean()),shuffle_J=float(shuffle[mask].mean()),within_shuffle_J=float((matched[mask]@local_hist).mean()))
            if scope=='target12':
                for k,v in recvalues.items():block[label][f'recombine_{k}_J']=float(v[mask].mean())
                for k,v in info.items():block[label][k]=float(v[mask].mean())
                for kind,probs in [('bayes_message_Q',sp),('bayes_greedy_J',np.eye(49)[codes])]:
                    mass=np.stack([probs[mask&(raw['map_id']==mid)].sum(0) for mid in ids])/int(mask.sum())
                    block[label][kind]=float(mass.max(0).sum())
        result[scope]=block
    return result


def information(raw,p,arm):
    """Exact conditional interventions on two-alternative common targets.

    Pairing holds photograph pair and shown resource fixed. Conditional mutual
    information is descriptive channel information, not token compositionality.
    """
    target=np.isin(raw['map_id'],support.groups(p,arm)['target12']);ix=np.flatnonzero(target)
    codes=7*raw['tokens'][:,0]+raw['tokens'][:,1];acts=raw['receiver_logits'].argmax(-1)
    correct=(acts[codes]==raw['positions']).all(-1);sp=probabilities(raw['sender_log_probs'])
    result={}
    for role,label,other in [(0,'food','water'),(1,'water','food')]:
        pairs=[]
        for i in ix:
            matches=np.flatnonzero(target&(raw['photo_ids']==raw['photo_ids'][i]).all(-1)&(raw['shown']==raw['shown'][i])&(raw['positions'][:,role]==raw['positions'][i,role]))
            matches=matches[matches!=i];assert len(matches)==1;pairs.append(int(matches[0]))
        pairs=np.asarray(pairs,np.int64)
        assert np.array_equal(raw['positions'][ix,role],raw['positions'][pairs,role])
        assert np.all(raw['positions'][ix,1-role]!=raw['positions'][pairs,1-role])
        swap_correct=(acts[codes[pairs]]==raw['positions'][ix]).astype(float)
        swapped=swap_correct.all(-1).astype(float)
        pair=(correct[ix]&correct[pairs]).astype(float)
        pm=sp[ix];mixture=.5*(pm+sp[pairs])
        # Stable zero-mass convention for KL in bits; logits are finite.
        positive=pm>0;terms=np.zeros_like(pm);terms[positive]=pm[positive]*np.log2(pm[positive]/mixture[positive])
        for name,values in [(f'swap_same_{label}_J',swapped),(f'pair_same_{label}_J',pair),(f'I_{other}_given_{label}_bits',terms.sum(-1)),(f'I_greedy_{other}_given_{label}_bits',(codes[ix]!=codes[pairs]).astype(float)),(f'swap_same_{label}_food',swap_correct[:,0]),(f'swap_same_{label}_water',swap_correct[:,1])]:
            result[name]=np.full(len(codes),np.nan);result[name][ix]=values
    return result


def save_null(raw,p,arm,path):
    permutations=np.load(Path(__file__).parent/'code_relabelings.npy')
    result=recombination(raw,p,arm,permutations)
    np.savez_compressed(path,**result)
