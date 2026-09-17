"""Division-of-labour signaling environment and centralized oracle."""
from __future__ import annotations

import numpy as np
from . import design, model


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def _sender_permutation(messages, sender, t):
    """Permute only scout tokens within each scout-identity subgroup."""
    out=messages[:,t].copy()
    for a in design.AGENTS:
        idx=np.flatnonzero(sender==a)
        if len(idx)>1:
            out[idx,a]=messages[np.roll(idx,1 if a==0 else -1),t,a]
    return out


def run_episode_batch(nets, episodes, memory, information, channel, *,
                      sample=True, controls=None, collect=True):
    message_mode=(controls or {}).get('message_mode','natural')
    require(message_mode in ('natural','closed','permuted'), 'unknown message mode')
    site_types=np.asarray(episodes['site_type'], dtype=np.int8)
    sender=np.asarray(episodes['sender'], dtype=np.int8)
    require(site_types.ndim==2 and site_types.shape[1]==2, 'bad site_type')
    require(sender.shape==(len(site_types),), 'bad sender')
    n=len(sender); H=design.HORIZON
    cap=np.asarray(episodes['capacity'], dtype=np.int16).copy()
    remain=np.stack([cap,cap],axis=1)
    last=np.zeros((n,2),dtype=np.int64)
    h=[np.zeros((n,design.HIDDEN),dtype=np.float64) for _ in design.AGENTS]
    prev_recv=[np.zeros((n,design.RECV_DIM),dtype=np.float64) for _ in design.AGENTS]
    for q in prev_recv: q[:,-1]=1.
    messages=np.full((n,H,2),design.NULL_MESSAGE,dtype=np.int64)
    actions=np.zeros((n,H,2),dtype=np.int64)
    rewards=np.zeros((n,H,2),dtype=np.float64)
    cache_trace=[[None,None] for _ in range(H)]
    obs_trace=[[None,None] for _ in range(H)]
    probs_msg=[[None,None] for _ in range(H)]
    probs_act=[[None,None] for _ in range(H)]
    for t in range(H):
        step_msg=[]; step_act=[]; step_cache=[]
        for a in design.AGENTS:
            x=np.zeros((n,design.OBS_DIM),dtype=np.float64)
            for i in range(n):
                is_sender=bool(sender[i]==a)
                own_goal=int(episodes['goal'][i,a,t]) if is_sender else 0
                partner_goal=int(episodes['goal'][i,1-a,t]) if not is_sender else 0
                x[i]=design.encode_observation(
                    own_goal=own_goal, local_type=int(site_types[i,a]),
                    local_inventory=int(remain[i,a]),
                    last_result=int(last[i,a]) if design.EXPOSE_OUTCOME else 0,
                    time=t, partner_goal=partner_goal,
                    remote_type=int(site_types[i,1-a]),
                    remote_inventory=int(remain[i,1-a]), information=information,
                    sender_role=is_sender)
            hh,ml,al,cache=model.forward(nets[a],x,prev_recv[a],h[a],memory)
            pm=model.softmax(ml); pa=model.softmax(al)
            mu=episodes['message_uniforms'][:,t,a]
            au=episodes['action_uniforms'][:,t,a]
            if sample:
                m=model.onehot_sample(pm,mu) if t in design.MESSAGE_ROUNDS else np.full(n,design.NULL_MESSAGE,dtype=np.int64)
                act=model.onehot_sample(pa,au)
            else:
                m=pm.argmax(axis=-1) if t in design.MESSAGE_ROUNDS else np.full(n,design.NULL_MESSAGE,dtype=np.int64)
                act=pa.argmax(axis=-1)
            m=np.where(sender==a,m,design.NULL_MESSAGE).astype(np.int64)
            act=np.where((sender==a)|(t<design.ACTION_START),0,act).astype(np.int64)
            messages[:,t,a]=m; actions[:,t,a]=act
            step_msg.append(pm); step_act.append(pa); step_cache.append((cache,hh,ml,al,x))
            h[a]=hh
        if channel=='silent' or message_mode=='closed':
            messages[:,t,:]=design.NULL_MESSAGE
        elif message_mode=='permuted' and t in design.MESSAGE_ROUNDS:
            messages[:,t,:]=_sender_permutation(messages,sender,t)
        for a in design.AGENTS:
            prev_recv[a].fill(0.); incoming=messages[:,t,1-a]
            prev_recv[a][np.arange(n),incoming]=1.
        for i in range(n):
            w=1-int(sender[i]); chosen=int(actions[i,t,w])
            if chosen==0:
                continue
            site=chosen-1; avail=int(remain[i,site])
            if avail>0: remain[i,site]-=1
            goal=int(episodes['goal'][i,int(sender[i]),t])
            success=avail>0 and int(site_types[i,site])==goal
            val=design.CORRECT_REWARD if success else design.WRONG_REWARD
            rewards[i,t,w]=val; rewards[i,t,int(sender[i])]=val
            last[i,w]=1 if success else 2
        for a in design.AGENTS:
            probs_msg[t][a]=step_msg[a]; probs_act[t][a]=step_act[a]
        if collect:
            cache_trace[t]=step_cache; obs_trace[t]=[step_cache[a][4] for a in design.AGENTS]
    return dict(messages=messages,actions=actions,rewards=rewards,
                team_return=rewards.mean(axis=(1,2)),cache=cache_trace,obs=obs_trace,
                probs_msg=probs_msg,probs_act=probs_act,final_inventory=remain,
                episodes=episodes,message_mode=message_mode,memory=memory,
                information=information,channel=channel)


def oracle_team_return(episodes):
    """Finite-horizon optimum when the scout request is fully visible."""
    site=np.asarray(episodes['site_type'],dtype=np.int8)
    sender=np.asarray(episodes['sender'],dtype=np.int8)
    goals=np.asarray(episodes['goal'],dtype=np.int8)
    cap=np.asarray(episodes['capacity'],dtype=np.int16)
    require(site.ndim==2 and site.shape[1]==2,'bad site_type')
    require(goals.shape==(len(site),2,design.HORIZON),'bad goals')
    require(np.all(cap==cap[0]),'capacity must be constant')
    c=int(cap[0]); n=len(site)
    dp=np.zeros((n,c+1,c+1),dtype=np.float64)
    for t in range(design.HORIZON-1,-1,-1):
        nxt=np.full_like(dp,-np.inf)
        for r0 in range(c+1):
            for r1 in range(c+1):
                best=np.full(n,-np.inf)
                for act in ([0] if t<design.ACTION_START else range(design.ACTION_COUNT)):
                    if act==0:
                        cand=dp[:,r0,r1]
                    else:
                        s=act-1; avail=(r0,r1)[s]
                        rr0=r0-(1 if s==0 and avail>0 else 0)
                        rr1=r1-(1 if s==1 and avail>0 else 0)
                        correct=site[:,s]==goals[np.arange(n),sender,t]
                        val=np.where((avail>0)&correct,design.CORRECT_REWARD,design.WRONG_REWARD)
                        cand=val+dp[:,rr0,rr1]
                    best=np.maximum(best,cand)
                nxt[:,r0,r1]=best
        dp=nxt
    return dp[:,c,c]/float(design.HORIZON)
