"""Small NumPy policy and optimizer used by the temporal scarcity study."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from . import design


def require(ok, msg):
    if not ok: raise ValueError(msg)


def softmax(logits):
    logits=np.asarray(logits,dtype=np.float64); z=logits-logits.max(axis=-1,keepdims=True)
    e=np.exp(z); return e/e.sum(axis=-1,keepdims=True)


def logsoftmax(logits):
    z=logits-logits.max(axis=-1,keepdims=True)
    return z-np.log(np.exp(z).sum(axis=-1,keepdims=True))


def onehot_sample(probabilities, uniforms):
    c=np.cumsum(probabilities,axis=-1)
    return (uniforms[...,None] >= c).sum(axis=-1).astype(np.int64)


def make_network(seed, agent, memory):
    # Keep every non-memory parameter paired between stateless and recurrent
    # arms.  The only architectural intervention is whether W_h is active.
    rng=np.random.default_rng(np.random.SeedSequence([seed, agent, 681]))
    d=design.OBS_DIM; r=design.RECV_DIM; h=design.HIDDEN; k=design.ALPHABET_SIZE; a=design.ACTION_COUNT
    # x and received-message streams both feed the recurrent core.
    net={
      'W_obs':rng.normal(0,math.sqrt(2/(d+h)),(d,h)),
      'W_recv':rng.normal(0,math.sqrt(2/(r+h)),(r,h)),
      'b_h':np.zeros(h), 'W_h':rng.normal(0,math.sqrt(2/(h+h)),(h,h)),
      'W_msg':rng.normal(0,math.sqrt(2/(h+k)),(h,k)), 'b_msg':np.zeros(k),
      'W_act':rng.normal(0,math.sqrt(2/(h+a)),(h,a)), 'b_act':np.zeros(a),
    }
    if memory=='stateless': net['W_h'][:] = 0.
    return net


def make_networks(seed,memory): return [make_network(seed,a,memory) for a in design.AGENTS]


def clone_networks(nets): return [{k:v.copy() for k,v in n.items()} for n in nets]


def parameter_hash(nets):
    h=hashlib.sha256()
    for n in nets:
        for k in sorted(n): h.update(k.encode()); h.update(np.asarray(n[k],dtype=np.float64).tobytes())
    return h.hexdigest()


def forward(net,x,recv,hprev,memory):
    hprev_eff = hprev if memory == 'recurrent' else np.zeros_like(hprev)
    pre=(np.einsum('bi,ij->bj',x,net['W_obs'])
         +np.einsum('bi,ij->bj',recv,net['W_recv'])
         +np.einsum('bi,ij->bj',hprev_eff,net['W_h'])+net['b_h'])
    h=np.tanh(pre)
    return (h,np.einsum('bi,ij->bj',h,net['W_msg'])+net['b_msg'],
            np.einsum('bi,ij->bj',h,net['W_act'])+net['b_act'],
            (x,recv,hprev_eff,pre,h))


def zero_grads(nets): return [{k:np.zeros_like(v) for k,v in n.items()} for n in nets]


def adam_state(nets): return [{k:{'m':np.zeros_like(v),'v':np.zeros_like(v)} for k,v in n.items()} for n in nets]


def adam_step(nets,grads,state,update):
    norm=math.sqrt(sum(float((g*g).sum()) for q in grads for g in q.values()))
    require(math.isfinite(norm),'Nonfinite gradient')
    scale=min(1.,design.GRADIENT_CLIP/max(norm,1e-300))
    for n,g,s in zip(nets,grads,state):
      for k in n:
        q=g[k]*scale; s[k]['m']=design.BETA1*s[k]['m']+(1-design.BETA1)*q
        s[k]['v']=design.BETA2*s[k]['v']+(1-design.BETA2)*q*q
        n[k]-=design.LEARNING_RATE*(s[k]['m']/(1-design.BETA1**update))/(np.sqrt(s[k]['v']/(1-design.BETA2**update))+design.EPSILON)
        require(np.isfinite(n[k]).all(),'Nonfinite parameter')
    return norm,scale


def entropy_grad(prob):
    lp=np.log(np.maximum(prob,1e-300)); ent=-(prob*lp).sum(axis=-1,keepdims=True)
    return -prob*(lp+ent)


def network_sha(net):
    h=hashlib.sha256()
    for k in sorted(net): h.update(k.encode()); h.update(net[k].tobytes())
    return h.hexdigest()
