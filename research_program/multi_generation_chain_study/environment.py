"""Staged environment with mixed joint-history and slot-local worker readout."""
from __future__ import annotations
import numpy as np
from research_program.action_dependent_signaling_study import design as action
from research_program.action_dependent_signaling_study import policy


def require(ok,msg):
    if not ok: raise ValueError(msg)

def _permute_sequences(sequences,partner_id):
    out=sequences.copy()
    for worker in range(action.WORKERS):
        idx=np.flatnonzero(partner_id==worker)
        if len(idx)>1: out[idx]=sequences[np.roll(idx,1)]
    return out

def _schedule(delivered):
    n,length=delivered.shape; messages=np.full((n,action.HORIZON),action.NULL_MESSAGE,dtype=np.int64)
    for slot,t in enumerate(action.message_arrival_times("staged","dual2")): messages[:,t]=delivered[:,slot]
    return messages

def rollout(p,ep,representation,channel,*,message_mode="natural",sample=True,partner_filter=None,message_override=None):
    require(representation in ("joint_history","slot_local"),"bad representation")
    require(channel in action.CHANNELS and message_mode in ("natural","closed","permuted"),"bad channel")
    n=len(ep["goal"]); active=np.ones(n,dtype=bool) if partner_filter is None else (ep["partner_id"]==int(partner_filter))
    k=action.alphabet_size("dual2"); length=action.message_length("dual2")
    remain=ep["capacity"].astype(np.int16).copy(); actions=np.zeros((n,action.HORIZON),dtype=np.int64); rewards=np.zeros((n,action.HORIZON),dtype=np.float64)
    action_probs=[None]*action.HORIZON; states=[None]*action.HORIZON; locals_=[None]*action.HORIZON; inventories=[None]*action.HORIZON; message_probs=[None]*length
    goal_idx=action.goal_index(ep["goal"]); selected=np.zeros((n,length),dtype=np.int64)
    for slot in range(length):
        sp=policy.softmax(p["sender_logits_hidden"][goal_idx,slot]); message_probs[slot]=sp
        selected[:,slot]=policy.sample(sp,ep["message_uniforms"][:,slot]) if sample else sp.argmax(axis=-1)
    delivered=selected.copy()
    if channel=="silent" or message_mode=="closed": delivered=np.full((n,length),action.NULL_MESSAGE,dtype=np.int64)
    elif message_mode=="permuted": delivered=_permute_sequences(selected,ep["partner_id"])
    delivered=np.where(active[:,None],delivered,action.NULL_MESSAGE)
    messages=_schedule(delivered); worker_idx=ep["partner_id"].astype(np.int64)
    history=np.zeros(n,dtype=np.int64); received=np.zeros(n,dtype=np.int8); slot_memory=np.full((n,length),action.NULL_MESSAGE,dtype=np.int64)
    arrivals=action.message_arrival_times("staged","dual2")
    for t in range(action.HORIZON):
        incoming=messages[:,t-1] if t>0 else np.full(n,action.NULL_MESSAGE,dtype=np.int64); seen=incoming!=action.NULL_MESSAGE
        if np.any(seen):
            first=seen&(received==0); later=seen&~first
            history[first]=incoming[first]+1
            if np.any(later): history[later]=1+(history[later]-1)*k+incoming[later]
            received[seen]+=1
            for slot,arrival in enumerate(arrivals):
                if t-1==arrival: slot_memory[seen,slot]=incoming[seen]
        stage=min(max((t-action.ACTION_START)//action.STEPS_PER_SUBTASK,0),action.SUBTASKS-1)
        local=ep["site_type"][:,stage,0].astype(np.int64); inventory=remain[:,stage,:].sum(axis=1).astype(np.int64)
        local_state=slot_memory[:,stage]+1
        use_local=(representation=="slot_local")&(worker_idx==0)&(t>=action.ACTION_START)
        state=np.where(use_local,local_state,history).astype(np.int64)
        ap=policy.softmax(p["worker_logits"][worker_idx,state,t,local,inventory]); action_probs[t]=ap; states[t]=state.copy(); locals_[t]=local.copy(); inventories[t]=inventory.copy()
        act=policy.sample(ap,ep["action_uniforms"][:,t]) if sample else ap.argmax(axis=-1); act=np.where((t<action.ACTION_START)|(~active),0,act).astype(np.int64); actions[:,t]=act
        if t>=action.ACTION_START:
            for i in np.flatnonzero(active):
                a=int(act[i])
                if a==0: continue
                site=a-1; available=int(remain[i,stage,site])
                if available>0: remain[i,stage,site]-=1
                target=int(ep["target_bits"][i,stage]); rewards[i,t]=action.CORRECT_REWARD if available>0 and int(ep["site_type"][i,stage,site])==target else action.WRONG_REWARD
    return {"messages":messages,"selected_messages":selected,"actions":actions,"rewards":rewards,"team_return":rewards.mean(axis=1),"active":active,"message_probs":message_probs,"action_probs":action_probs,"state":states,"local":locals_,"inventory":inventories,"final_inventory":remain,"episodes":ep,"message_mode":message_mode,"representation":representation}
