"""Independent fixed-scope analysis and bounded execution replay for v24.

The v23 independent metric implementation is reused and hash-bound. No v24
production metric, sampling, probability-enumeration or update helper is used.
"""
from __future__ import annotations
import argparse,copy,hashlib,itertools,json,math,shutil,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.23'))
import analyze_experience as prior
Checks=prior.Checks
CHUNK=256
ARMS=('mean','attention')

def read(p):return json.loads(Path(p).read_text())
def write(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return torch.load(p,weights_only=True,map_location='cpu')
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {key:z[key] for key in z.files}


def summarize(rows,seeds,times):
    """Mean directions within a pair, partitions within a source, then sources."""
    source=[]
    for seed,condition in itertools.product(seeds,ARMS):
        selected=[r for r in rows if r['seed']==seed and r['condition']==condition]
        curve=[dict(update=t,scores=prior.average([r['curve'][i]['scores'] for r in selected])) for i,t in enumerate(times)]
        source.append(dict(seed=seed,condition=condition,curve=curve,scores=curve[-1]['scores'],auc=prior.integrate(curve)))
    aggregate={}
    for condition in ARMS:
        selected=[r for r in source if r['condition']==condition]
        curve=[dict(update=t,scores=prior.average([r['curve'][i]['scores'] for r in selected])) for i,t in enumerate(times)]
        aggregate[condition]=dict(curve=curve,scores=curve[-1]['scores'],auc=prior.integrate(curve))
    primary=[]
    for seed in seeds:
        by_arm={r['condition']:r for r in source if r['seed']==seed}
        get=lambda arm,kind:by_arm[arm][kind]['new12']['pooled']['J']
        primary.append(dict(seed=seed,mean=get('mean','scores'),attention=get('attention','scores'),
            difference=get('attention','scores')-get('mean','scores'),
            auc_difference=get('attention','auc')-get('mean','auc')))
    return dict(rows=rows,seed_rows=source,aggregate=aggregate,primary=primary,
        primary_mean=math.fsum(r['difference'] for r in primary)/len(primary))


def check_attention(raw,check):
    a=raw['effective_attention']
    check.check(a.shape==(960,2,2) and a.dtype==np.float32 and np.isfinite(a).all(),'effective attention shape/dtype')
    check.check(np.all((a>=0)&(a<=1)) and np.max(np.abs(a.astype(np.float64).sum(-1)-1))<1e-6,'effective attention simplex')


class IndependentSender(nn.Module):
    """Same declared atomic architecture; production FocusSender is not called."""
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.encoder=nn.Linear(8,96);self.embedding=nn.Embedding(8,16)
        self.recur=nn.LSTMCell(16,96);self.bilinear=nn.Linear(96,96,bias=False)
        self.fusion=nn.Linear(192,96);self.out=nn.Linear(96,7)


def roles(agent):
    return dict(sender=list(agent.focus_sender.parameters()),receiver=[v for name in ('receive_embedding','actor') for v in getattr(agent,name).parameters()])


def independent_initialize(agent,mode,seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed));agent.focus_sender=IndependentSender(mode)
    agent.requires_grad_(False)
    for group in roles(agent).values():
        for value in group:value.requires_grad_(True)
    for value in agent.parameters():value.grad=None
    return agent


def sender_step(a,key,token,state):
    m=a.focus_sender
    h,c=m.recur(m.embedding(token.detach()),state)
    scores=(m.bilinear(h)[:,None,:]*key).sum(-1)
    weights=F.softmax(scores,-1);context=(weights[:,:,None]*key).sum(1)
    logits=m.out(torch.tanh(m.fusion(torch.cat((h,context),-1))))
    effective=weights.expand(-1,2)*.5 if m.mode=='mean' else weights
    return (h,c),logits,effective


def sender_start(a,x):
    key=F.gelu(a.focus_sender.encoder(x))
    if a.focus_sender.mode=='mean':key=key.mean(1,keepdim=True)
    zero=x.new_zeros(len(x),96)
    state,logits,weights=sender_step(a,key,torch.full((len(x),),7,dtype=torch.int64),(zero,zero))
    return key,state,logits,weights


def enumerate_sender(a,x):
    key,state,first,w0=sender_start(a,x)
    second=torch.stack([F.log_softmax(sender_step(a,key,torch.full((len(x),),t,dtype=torch.int64),state)[1],-1) for t in range(7)],1)
    lp=(F.log_softmax(first,-1)[:,:,None]+second).reshape(len(x),49)
    t0=first.argmax(-1).detach();_,last,w1=sender_step(a,key,t0,state)
    return lp,torch.stack((t0,last.argmax(-1).detach()),1),torch.stack((w0,w1),1)


def receive(a,messages):
    embedded=a.receive_embedding(messages.detach()).flatten(1)
    return a.actor(torch.cat((embedded,embedded.new_zeros(len(messages),20)),-1)).reshape(len(messages),2,6)


def independent_direction(sender,receiver,x,uniforms,positions,weight):
    def draw(logits,u):
        lp=F.log_softmax(logits,-1);prob=lp.exp()
        action=(prob.detach().cumsum(-1)<u).sum(-1).clamp(max=logits.shape[-1]-1)
        selected=lp.gather(1,action[:,None]).squeeze(1);entropy=-(prob*lp).sum(-1)
        return action.detach(),selected,entropy,prob
    key,state,first_logits,w0=sender_start(sender,x);u=torch.from_numpy(uniforms)
    first,l0,e0,p0=draw(first_logits,u[:,0:1])
    _,second_logits,w1=sender_step(sender,key,first,state)
    second,l1,e1,p1=draw(second_logits,u[:,1:2]);messages=torch.stack((first,second),1).detach()
    action_logits=receive(receiver,messages)
    food,lf,ef,pf=draw(action_logits[:,0],u[:,2:3]);water,lw,ew,pw=draw(action_logits[:,1],u[:,3:4])
    actions=torch.stack((food,water),1);success=(actions.numpy()==positions).astype(np.float32)
    reward=(.25*success.sum(1)+.5*success.prod(1)).astype(np.float32);advantage=torch.from_numpy(reward-np.float32(.5))
    slp=l0+l1;rlp=lf+lw;se=e0+e1;re=ef+ew
    sl=-(slp*advantage).mean()-float(weight)*se.mean();rl=-(rlp*advantage).mean()-float(weight)*re.mean()
    ar=lambda value:value.detach().numpy().copy()
    trace=dict(concepts=ar(x),uniforms=uniforms.copy(),messages=ar(messages),first_logits=ar(first_logits),second_logits=ar(second_logits),
        token_probabilities=ar(torch.stack((p0,p1),1)),effective_attention=ar(torch.stack((w0,w1),1)),action_logits=ar(action_logits),
        action_probabilities=ar(torch.stack((pf,pw),1)),actions=ar(actions),positions=positions.copy(),success=success,reward=reward,
        advantage=ar(advantage),sender_logp=ar(slp),receiver_logp=ar(rlp),sender_entropy=ar(se),receiver_entropy=ar(re),
        sender_loss=float(sl.detach()),receiver_loss=float(rl.detach()),entropy_weight=float(weight),baseline=.5)
    return sl,rl,trace
