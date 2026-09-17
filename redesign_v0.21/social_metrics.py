"""Finite-support communication statistics; no training or fitted decoder."""
import itertools
import numpy as np
from social_world import temporal,local_keys,group_indices

def entropy_codes(c):
    f=np.bincount(c,minlength=49).astype(np.float64);p=f[f>0]/len(c)
    return float(-(p*np.log2(p)).sum())

def metrics(sender_lp,receiver_logits,tokens,w,p):
    lp=np.asarray(sender_lp,np.float64);sp=np.exp(lp-lp.max(-1,keepdims=True));sp/=sp.sum(-1,keepdims=True)
    rl=np.asarray(receiver_logits,np.float64);rp=np.exp(rl-rl.max(-1,keepdims=True));rp/=rp.sum(-1,keepdims=True)
    code=tokens[:,0]*7+tokens[:,1];ra=rl.argmax(-1);actions=ra[code];target=w['positions'];correct=actions==target
    allgood=ra[None,:,:]==target[:,None,:];allJ=allgood.all(-1)
    rcorrect=np.stack([rp[:,k,target[:,k]].T for k in (0,1)],-1)
    q=(sp*rcorrect.prod(-1)).sum(-1)
    n=len(code);m=w['mover'];idx=np.arange(n)
    globals_freq=np.bincount(code,minlength=49)/n
    gshuffle=allJ@globals_freq
    conditional=np.zeros(n);conditional_static=np.zeros(n)
    for rows in group_indices(local_keys(w)):
        freq=np.bincount(code[rows],minlength=49)/len(rows)
        conditional[rows]=allJ[rows]@freq
        static=allgood[rows,:,1-int(m[rows[0]])]
        conditional_static[rows]=static@freq
    out={}
    for group,maps in temporal.partition(p).items():
        mask=np.isin(w['target_map'],maps);rows=np.flatnonzero(mask);good=correct[rows]
        result=dict(n=len(rows),J=float(good.all(-1).mean()),single=float(good.mean()),Q=float(q[rows].mean()),
            moved=float(correct[idx,m][rows].mean()),stationary=float(correct[idx,1-m][rows].mean()),
            receiver_coverage=float(allJ[rows].any(-1).mean()),
            native_sender_entropy_bits=float(-(sp[rows]*np.log2(np.maximum(sp[rows],1e-300))).sum(-1).mean()),
            unique_greedy_codes=int(len(np.unique(code[rows]))),
            global_shuffle_J=float(gshuffle[rows].mean()),conditional_shuffle_J=float(conditional[rows].mean()),
            conditional_shuffle_stationary=float(conditional_static[rows].mean()),constant00_J=float(allJ[rows,0].mean()))
        # Subgroup scores use the same global/common-support intervention. Never
        # let a subgroup-specific oracle redefine the message marginal.
        if group in ('old','common30'):
            result.update(temporal.history_pairs(actions,w,mask))
            mi=[];sepnum=sepden=jointpair=0
            for loc in group_indices(local_keys(w)[rows]):
                rr=rows[loc];station=target[rr,1-int(m[rr[0]])]
                h=entropy_codes(code[rr]);within=0.;classes=[]
                for s in np.unique(station):
                    ss=rr[station==s];within+=len(ss)/len(rr)*entropy_codes(code[ss]);classes.append(ss)
                mi.append((len(rr),h-within))
                for a,b in itertools.combinations(classes,2):
                    sepden+=len(a)*len(b)
                    sepnum+=int((code[a,None]!=code[b][None,:]).sum())
                    jointpair+=int(correct[a].all(-1).sum())*int(correct[b].all(-1).sum())
            result['static_message_cmi_bits']=float(sum(n*v for n,v in mi)/sum(n for n,_ in mi))
            result['history_message_separation']=float(sepnum/sepden)
            result['history_pair_both_joint_J']=float(jointpair/sepden)
            assert result['history_pair_J']<=result['history_message_separation']+1e-12
        else:result.update(history_pair_J=None,history_pairs=0,history_pair_correct=0,static_message_cmi_bits=None,history_message_separation=None,history_pair_both_joint_J=None)
        # Macro-average terminal map/photo groups. Remove exact weight repeats,
        # and compare different unique legal (source,mover) routes only.
        route_agree=[];route_h=[];route_correct=[]
        keys=np.column_stack((w['target_map'][rows],w['photo_ids'][rows]))
        for loc in group_indices(keys):
            rr=rows[loc]
            _,first=np.unique(np.column_stack((w['source_map'][rr],m[rr])),axis=0,return_index=True)
            rr=rr[first];assert len(rr)>1
            pairs=list(itertools.combinations(rr,2));same=[code[a]==code[b] for a,b in pairs]
            route_agree.append(np.mean(same));route_h.append(entropy_codes(code[rr]))
            route_correct.append(np.mean([code[a]==code[b] and correct[a].all() and correct[b].all() for a,b in pairs]))
        result.update(route_code_agreement=float(np.mean(route_agree)),route_code_entropy_bits=float(np.mean(route_h)),
            route_samecode_bothcorrect=float(np.mean(route_correct)))
        assert result['J']<=result['receiver_coverage']+1e-12
        out[group]=result
    return out
