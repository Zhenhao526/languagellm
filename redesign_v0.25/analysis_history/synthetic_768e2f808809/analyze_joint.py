"""Independent analysis and bounded replay of the zero-initialized joint readout.

Only the new joint condition is replayed. The frozen v24 mean/attention rows are
reused as explicitly labelled references, with their analysis and audit hashes.
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
sys.path.insert(0,str(PROJECT/'redesign_v0.24'))
import analyze_attention as prior
experience=prior.prior
Checks=prior.Checks;CHUNK=256

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return torch.load(p,weights_only=True,map_location='cpu')
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def summarize(rows,seeds,times):
    arms=('mean','joint','attention');source=[]
    for seed,arm in itertools.product(seeds,arms):
        r=[x for x in rows if x['seed']==seed and x['condition']==arm]
        curve=[dict(update=t,scores=experience.average([x['curve'][i]['scores'] for x in r])) for i,t in enumerate(times)]
        source.append(dict(seed=seed,condition=arm,curve=curve,scores=curve[-1]['scores'],auc=experience.integrate(curve)))
    aggregate={}
    for arm in arms:
        r=[x for x in source if x['condition']==arm]
        curve=[dict(update=t,scores=experience.average([x['curve'][i]['scores'] for x in r])) for i,t in enumerate(times)]
        aggregate[arm]=dict(curve=curve,scores=curve[-1]['scores'],auc=experience.integrate(curve))
    primary=[]
    for seed in seeds:
        r={x['condition']:x for x in source if x['seed']==seed}
        metric=lambda arm,kind,group:r[arm][kind][group]['pooled']['J']
        primary.append(dict(seed=seed,mean=metric('mean','scores','old'),joint=metric('joint','scores','old'),
            difference=metric('joint','scores','old')-metric('mean','scores','old'),
            auc_difference=metric('joint','auc','old')-metric('mean','auc','old'),
            new12_difference=metric('joint','scores','new12')-metric('mean','scores','new12')))
    return dict(rows=rows,seed_rows=source,aggregate=aggregate,primary=primary,
        primary_mean=math.fsum(x['difference'] for x in primary)/len(primary))


class IndependentJoint(nn.Module):
    def __init__(self,reference):
        super().__init__();self.mode='joint'
        self.encoder=copy.deepcopy(reference.encoder);self.embedding=copy.deepcopy(reference.embedding);self.recur=copy.deepcopy(reference.recur)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(25025);self.contrast=nn.Linear(96,96,bias=False);nn.init.zeros_(self.contrast.weight)
        self.fusion=copy.deepcopy(reference.fusion);self.out=copy.deepcopy(reference.out)


def independent_initialize(agent):
    agent.focus_sender=IndependentJoint(agent.focus_sender);agent.requires_grad_(False)
    for vs in prior.roles(agent).values():
        for value in vs:value.requires_grad_(True)
    for value in agent.parameters():value.grad=None
    return agent


def sender_step(a,key,token,state):
    m=a.focus_sender;h,c=m.recur(m.embedding(token.detach()),state)
    average=key.mean(1);difference=(key[:,0]-key[:,1])*.5
    activation=m.fusion(torch.cat((h,average),-1))+m.contrast(difference)
    return (h,c),m.out(torch.tanh(activation))


def sender_start(a,x):
    key=F.gelu(a.focus_sender.encoder(x));zero=x.new_zeros(len(x),96)
    state,first=sender_step(a,key,torch.full((len(x),),7,dtype=torch.int64),(zero,zero))
    return key,state,first


def enumerate_sender(a,x):
    key,state,first=sender_start(a,x)
    second=torch.stack([F.log_softmax(sender_step(a,key,torch.full((len(x),),i,dtype=torch.int64),state)[1],-1) for i in range(7)],1)
    probabilities=(F.log_softmax(first,-1)[:,:,None]+second).reshape(len(x),49)
    first_token=first.argmax(-1).detach();_,last=sender_step(a,key,first_token,state)
    return probabilities,torch.stack((first_token,last.argmax(-1).detach()),1)


def independent_direction(sender,receiver,x,uniforms,positions,weight):
    def draw(logits,u):
        lp=F.log_softmax(logits,-1);p=lp.exp();action=(p.detach().cumsum(-1)<u).sum(-1).clamp(max=logits.shape[-1]-1)
        chosen=lp.gather(1,action[:,None]).squeeze(1);entropy=-(p*lp).sum(-1)
        return action.detach(),chosen,entropy,p
    key,state,fl=sender_start(sender,x);u=torch.from_numpy(uniforms)
    t0,l0,e0,p0=draw(fl,u[:,0:1]);_,second=sender_step(sender,key,t0,state)
    t1,l1,e1,p1=draw(second,u[:,1:2]);messages=torch.stack((t0,t1),1).detach();al=prior.receive(receiver,messages)
    food,lf,ef,pf=draw(al[:,0],u[:,2:3]);water,lw,ew,pw=draw(al[:,1],u[:,3:4])
    actions=torch.stack((food,water),1);success=(actions.numpy()==positions).astype(np.float32)
    reward=(.25*success.sum(1)+.5*success.prod(1)).astype(np.float32);advantage=torch.from_numpy(reward-np.float32(.5))
    slp=l0+l1;rlp=lf+lw;se=e0+e1;re=ef+ew
    sl=-(slp*advantage).mean()-float(weight)*se.mean();rl=-(rlp*advantage).mean()-float(weight)*re.mean()
    ar=lambda value:value.detach().numpy().copy()
    trace=dict(concepts=ar(x),uniforms=uniforms.copy(),messages=ar(messages),first_logits=ar(fl),second_logits=ar(second),
        token_probabilities=ar(torch.stack((p0,p1),1)),action_logits=ar(al),action_probabilities=ar(torch.stack((pf,pw),1)),
        actions=ar(actions),positions=positions.copy(),success=success,reward=reward,advantage=ar(advantage),sender_logp=ar(slp),receiver_logp=ar(rlp),
        sender_entropy=ar(se),receiver_entropy=ar(re),sender_loss=float(sl.detach()),receiver_loss=float(rl.detach()),entropy_weight=float(weight),baseline=.5)
    return sl,rl,trace
