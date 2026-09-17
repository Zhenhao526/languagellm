"""Production summaries; independent verification is in analyze_experience.py."""
import numpy as np
from world import partition


def probabilities(logits):
    z=np.asarray(logits,np.float64);z=z-z.max(-1,keepdims=True)
    p=np.exp(z);return p/p.sum(-1,keepdims=True)


def masks(w,p):
    sets=partition(p);sets['new12']=np.r_[sets['added'],sets['sealed']]
    return {k:np.isin(w['map_id'],ids) for k,ids in sets.items()}


def summarize(correct,q,w,p,extra=None):
    result={}
    for group,selected in masks(w,p).items():
        block={}
        for label,mask in [('pooled',selected),('food_only',selected&(w['shown']==0)),('water_only',selected&(w['shown']==1))]:
            c=correct[mask];block[label]=dict(n=int(mask.sum()),J=float(c.all(-1).mean()),food=float(c[:,0].mean()),water=float(c[:,1].mean()),Q=float(q[mask].mean()))
            if extra:
                for name,v in extra.items():block[label][name]=float(v[mask].mean())
        result[group]=block
    return result


def private(raw,p):
    probs=probabilities(raw['logits']);correct=raw['logits'].argmax(-1)==raw['positions']
    target=np.take_along_axis(probs,raw['positions'][...,None],-1)[...,0]
    return summarize(correct,target.prod(-1),raw,p)


def social(raw,p):
    token=raw['tokens'];codes=token[:,0]*7+token[:,1];rl=raw['receiver_logits'];acts=rl.argmax(-1)
    correct=acts[codes]==raw['positions'];rp=probabilities(rl);lp=np.asarray(raw['sender_log_probs'],np.float64);sp=np.exp(lp-lp.max(-1,keepdims=True));sp/=sp.sum(-1,keepdims=True)
    q=(sp*rp[:,0,raw['positions'][:,0]].T*rp[:,1,raw['positions'][:,1]].T).sum(-1)
    targets=raw['positions'];null=(acts[0]==targets).all(-1).astype(float)
    # Exact expected outcome of assigning an independently sampled complete
    # greedy message from the same finite pooled evaluation support.
    hist=np.bincount(codes,minlength=49)/len(codes)
    matched=(acts[None,:,:]==targets[:,None,:]).all(-1)
    shuffle=matched@hist
    return summarize(correct,q,raw,p,dict(blank_J=null,shuffle_J=shuffle))
