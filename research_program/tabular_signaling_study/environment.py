"""Vectorized-enough rollout and centralized upper bound."""
from __future__ import annotations

import numpy as np
from . import design, policy


def require(ok,msg):
    if not ok: raise ValueError(msg)


def _permute(tokens):
    if len(tokens)<=1: return tokens.copy()
    return tokens[np.roll(np.arange(len(tokens)),1)]


def rollout(p,ep,information,channel,*,message_mode='natural',sample=True):
    n=len(ep['site_type']); H=design.HORIZON
    require(channel in design.CHANNELS and message_mode in ('natural','closed','permuted'),'bad channel')
    remain=np.stack([ep['capacity'],ep['capacity']],axis=1).astype(np.int16)
    messages=np.full((n,H,2),design.NULL_MESSAGE,dtype=np.int64)
    actions=np.zeros((n,H,2),dtype=np.int64)
    rewards=np.zeros((n,H,2),dtype=np.float64)
    msg_probs=[None]*H; action_probs=[None]*H
    state_idx=[None]*H; local_idx=[None]*H; inv_idx=[None]*H; goal_idx=[None]*H
    memory_state=np.full(n,design.NULL_MESSAGE,dtype=np.int64)
    incoming=np.full(n,design.NULL_MESSAGE,dtype=np.int64)
    c=ep['context']; local_sender=ep['site_type'][:,0].astype(np.int64)
    sp=policy.softmax(p['sender_logits'][c,local_sender])
    msg=policy.sample(sp,ep['message_uniforms'][:,0,0]) if sample else sp.argmax(axis=-1)
    msg=msg.astype(np.int64)
    if channel=='silent' or message_mode=='closed': msg_del=np.full(n,design.NULL_MESSAGE,dtype=np.int64)
    elif message_mode=='permuted': msg_del=_permute(msg)
    else: msg_del=msg.copy()
    messages[:,0,0]=msg_del; msg_probs[0]=sp
    for t in range(H):
        if t>0:
            incoming=messages[:,t-1,0].copy()
        if p['memory']=='recurrent':
            seen=incoming!=design.NULL_MESSAGE
            memory_state[seen]=incoming[seen]
            state=memory_state.copy()
        else:
            state=incoming.copy()
        local=ep['site_type'][:,1].astype(np.int64)
        inv=np.clip(remain[:,1],0,design.HORIZON).astype(np.int64)
        vg=ep['goal'][:,0,t].astype(np.int64) if information=='FI' else np.zeros(n,dtype=np.int64)
        ap=policy.softmax(p['worker_logits'][state,t,local,inv,vg])
        act=policy.sample(ap,ep['action_uniforms'][:,t,1]) if sample else ap.argmax(axis=-1)
        act=np.where(t<design.ACTION_START,0,act).astype(np.int64)
        actions[:,t,1]=act; action_probs[t]=ap; state_idx[t]=state; local_idx[t]=local; inv_idx[t]=inv; goal_idx[t]=vg
        if t>=design.ACTION_START:
            for i in range(n):
                a=int(act[i])
                if a==0: continue
                site=a-1; avail=int(remain[i,site]);
                if avail>0: remain[i,site]-=1
                success=avail>0 and int(ep['site_type'][i,site])==int(ep['goal'][i,0,t])
                val=design.CORRECT_REWARD if success else design.WRONG_REWARD
                rewards[i,t,1]=val; rewards[i,t,0]=val
    return {'messages':messages,'actions':actions,'rewards':rewards,
            'team_return':rewards.mean(axis=(1,2)),'message_probs':msg_probs,
            'action_probs':action_probs,'state_idx':state_idx,'local_idx':local_idx,
            'inv_idx':inv_idx,'goal_idx':goal_idx,'final_inventory':remain,
            'episodes':ep,'message_mode':message_mode,'memory':p['memory'],
            'information':information,'channel':channel}


def oracle_team_return(ep):
    site=np.asarray(ep['site_type'],dtype=np.int8); goal=np.asarray(ep['goal'],dtype=np.int8)
    cap=np.asarray(ep['capacity'],dtype=np.int16); n=len(site)
    require(np.all(cap==cap[0]),'capacity must be constant')
    c=int(cap[0]); dp=np.zeros((n,c+1,c+1),dtype=np.float64)
    for t in range(design.HORIZON-1,-1,-1):
        nxt=np.full_like(dp,-np.inf)
        acts=[0] if t<design.ACTION_START else range(design.ACTION_COUNT)
        for r0 in range(c+1):
            for r1 in range(c+1):
                best=np.full(n,-np.inf)
                for a in acts:
                    if a==0: cand=dp[:,r0,r1]
                    else:
                        s=a-1; avail=(r0,r1)[s]
                        rr0=r0-(1 if s==0 and avail>0 else 0); rr1=r1-(1 if s==1 and avail>0 else 0)
                        val=np.where((avail>0)&(site[:,s]==goal[:,0,t]),design.CORRECT_REWARD,design.WRONG_REWARD)
                        cand=val+dp[:,rr0,rr1]
                    best=np.maximum(best,cand)
                nxt[:,r0,r1]=best
        dp=nxt
    return dp[:,c,c]/float(design.HORIZON)
