"""Multi-round private-resource environment and causal evaluation controls."""
from __future__ import annotations
import numpy as np
from . import design, model


def require(ok,msg):
    if not ok: raise ValueError(msg)


def run_episode_batch(nets, episodes, memory, information, channel, *,
                      sample=True, controls=None, collect=True):
    """Roll out a batch; returns traces and team rewards.

    `controls` can set `message_mode` to `natural`, `closed`, or `permuted`.
    Policies never receive researcher-only goal/site fields in PI.
    """
    site_types = np.asarray(episodes.get('site_type'))
    require(site_types.ndim == 2 and site_types.shape[1] == 2,
            'episodes must provide independent site types')
    n=site_types.shape[0]; H=design.HORIZON
    message_mode=(controls or {}).get('message_mode','natural')
    require(message_mode in ('natural','closed','permuted'),'Unknown message mode')
    cap=episodes['capacity'].astype(np.int16).copy()
    remain=np.stack([cap,cap],axis=1)  # site x batch; material type is separate
    last=np.zeros((n,2),dtype=np.int64)
    h=[np.zeros((n,design.HIDDEN),dtype=np.float64) for _ in design.AGENTS]
    prev_recv=[np.zeros((n,design.RECV_DIM),dtype=np.float64) for _ in design.AGENTS]
    # null is the last receive slot
    for q in prev_recv: q[:,-1]=1.
    messages=np.full((n,H,2),design.NULL_MESSAGE,dtype=np.int64)
    actions=np.zeros((n,H,2),dtype=np.int64)
    rewards=np.zeros((n,H,2),dtype=np.float64)
    obs_trace=[[[] for _ in design.AGENTS] for _ in range(H)]
    cache_trace=[[[] for _ in design.AGENTS] for _ in range(H)]
    probs_msg=[[None,None] for _ in range(H)]; probs_act=[[None,None] for _ in range(H)]
    # A sees site0 and B sees site1.  The other patch is hidden in PI and
    # visible in FI; agents are not told that the two patch types are related.
    for t in range(H):
      step_msg=[]; step_act=[]; step_cache=[]
      for a in design.AGENTS:
        local_site=a; remote_site=1-a
        x=np.zeros((n,design.OBS_DIM),dtype=np.float64)
        for i in range(n):
          partner_goal=int(episodes['goal'][i,1-a,t])
          x[i]=design.encode_observation(
            own_goal=int(episodes['goal'][i,a,t]), local_type=int(site_types[i,local_site]),
            local_inventory=int(remain[i,local_site]), last_result=int(last[i,a]), time=t,
            partner_goal=partner_goal, remote_type=int(site_types[i,remote_site]),
            remote_inventory=int(remain[i,remote_site]), information=information)
        hh,ml,al,cache=model.forward(nets[a],x,prev_recv[a],h[a],memory)
        pm=model.softmax(ml)
        pa=model.softmax(al)
        mu=episodes['message_uniforms'][:,t,a]
        au=episodes['action_uniforms'][:,t,a]
        m=model.onehot_sample(pm,mu) if sample and t in design.MESSAGE_ROUNDS else (pm.argmax(axis=-1) if t in design.MESSAGE_ROUNDS else np.full(n,design.NULL_MESSAGE,dtype=np.int64))
        # action always samples; greedy controls use argmax and ignore action uniforms.
        act=model.onehot_sample(pa,au) if sample else pa.argmax(axis=-1)
        step_msg.append(pm);step_act.append(pa);step_cache.append((cache,hh,ml,al,x))
        messages[:,t,a]=m; actions[:,t,a]=act; h[a]=hh
      if channel=='silent': messages[:,t,:]=design.NULL_MESSAGE
      if message_mode=='closed': messages[:,t,:]=design.NULL_MESSAGE
      elif message_mode=='permuted' and t in design.MESSAGE_ROUNDS:
        # deterministic within-batch permutation preserves token marginals.
        messages[:,t,0]=messages[np.roll(np.arange(n),1),t,0]
        messages[:,t,1]=messages[np.roll(np.arange(n),-1),t,1]
      # Rebuild receive one-hots for the next step; current-step actions were
      # generated before these tokens, so a message affects the next round.
      for a in design.AGENTS:
        prev_recv[a].fill(0.); incoming=messages[:,t,1-a]; prev_recv[a][np.arange(n),incoming]=1.
      # Resolve simultaneous actions at each site.
      for i in range(n):
        chosen=actions[i,t]
        for site in (0,1):
          actors=np.flatnonzero(chosen==site+1)
          if len(actors)==0: continue
          avail=int(remain[i,site]); winners=actors if avail>=len(actors) else actors[:max(avail,0)]
          remain[i,site]-=len(winners)
          for a in actors:
            goal=int(episodes['goal'][i,a,t]); typ=int(site_types[i,site])
            if a in winners and typ==goal: rewards[i,t,a]=1.; last[i,a]=1
            elif a in winners: rewards[i,t,a]=-0.25; last[i,a]=2
            else: rewards[i,t,a]=-0.25; last[i,a]=2
        # wait leaves result neutral
        for a in design.AGENTS:
          if chosen[a]==0: last[i,a]=0
      for a in design.AGENTS:
        probs_msg[t][a]=step_msg[a]; probs_act[t][a]=step_act[a]
      if collect:
        cache_trace[t]=step_cache
        obs_trace[t]=[step_cache[a][4] for a in design.AGENTS]
    team_return=rewards.mean(axis=(1,2))
    return dict(messages=messages,actions=actions,rewards=rewards,team_return=team_return,
                cache=cache_trace,obs=obs_trace,probs_msg=probs_msg,probs_act=probs_act,
                final_inventory=remain,episodes=episodes,message_mode=message_mode,
                memory=memory,information=information,channel=channel)


def oracle_team_return(episodes):
    """Compute the finite-horizon centralized optimum for each episode.

    The oracle sees both private goal sequences and both patch types, while
    respecting the same simultaneous-action and persistent-inventory rules as
    the learned agents.  It is used only as an evaluation denominator; no
    oracle field enters a policy observation or a training gradient.
    """
    site_types = np.asarray(episodes['site_type'], dtype=np.int8)
    goals = np.asarray(episodes['goal'], dtype=np.int8)
    cap = np.asarray(episodes['capacity'], dtype=np.int16)
    require(site_types.ndim == 2 and site_types.shape[1] == 2, 'bad site_type')
    require(goals.shape == (len(site_types), 2, design.HORIZON), 'bad goals')
    require(np.all(cap == cap[0]), 'capacity must be constant within a batch')
    c = int(cap[0]); n = len(site_types)
    # dp[i, r0, r1] stores the best future sum of the two agents' rewards
    # from the current round onward, before normalization by 2*HORIZON.
    dp = np.zeros((n, c + 1, c + 1), dtype=np.float64)
    actions = range(design.ACTION_COUNT)
    for t in range(design.HORIZON - 1, -1, -1):
        nxt = np.full_like(dp, -np.inf)
        for r0 in range(c + 1):
            for r1 in range(c + 1):
                best = np.full(n, -np.inf, dtype=np.float64)
                for a0 in actions:
                    for a1 in actions:
                        chosen = (a0, a1)
                        wins = [False, False]
                        used = [0, 0]
                        for site in (0, 1):
                            actors = [a for a, act in enumerate(chosen) if act == site + 1]
                            avail = (r0, r1)[site]
                            if avail >= len(actors):
                                winners = actors
                            else:
                                winners = actors[:max(avail, 0)]
                            for a in winners:
                                wins[a] = True
                            used[site] = len(winners)
                        reward = np.zeros(n, dtype=np.float64)
                        for a, act in enumerate(chosen):
                            if act == 0:
                                continue
                            site = act - 1
                            correct = site_types[:, site] == goals[:, a, t]
                            reward += np.where(wins[a] & correct, 1.0, -0.25)
                        rr0, rr1 = r0 - used[0], r1 - used[1]
                        best = np.maximum(best, reward + dp[:, rr0, rr1])
                nxt[:, r0, r1] = best
        dp = nxt
    return dp[:, c, c] / float(2 * design.HORIZON)
