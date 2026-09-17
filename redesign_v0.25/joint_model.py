"""Ordered sum/difference readout, initially the identical v24 mean function."""
import copy,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'redesign_v0.21'))
import social_model as old

class JointSender(nn.Module):
    def __init__(self,reference):
        super().__init__();self.mode='joint'
        # Keep parameter/optimizer ordering for the shared tensors; contrast
        # occupies the place that bilinear held in the old parameter list.
        self.encoder=copy.deepcopy(reference.encoder)
        self.embedding=copy.deepcopy(reference.embedding)
        self.recur=copy.deepcopy(reference.recur)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(25025);self.contrast=nn.Linear(96,96,bias=False)
            nn.init.zeros_(self.contrast.weight)
        self.fusion=copy.deepcopy(reference.fusion);self.out=copy.deepcopy(reference.out)
    def keys(self,x):
        assert x.ndim==3 and x.shape[1:]==(2,8) and not x.requires_grad
        assert x.dtype==torch.float32 and bool(torch.isfinite(x).all())
        return F.gelu(self.encoder(x))
    def step(self,keys,token,state):
        h,c=self.recur(self.embedding(token.detach()),state)
        mean=keys.mean(1);difference=(keys[:,0]-keys[:,1])*.5
        fused=self.fusion(torch.cat((h,mean),-1))+self.contrast(difference)
        return (h,c),self.out(torch.tanh(fused))
    def start(self,x):
        keys=self.keys(x);zero=x.new_zeros(len(x),96)
        state,logits=self.step(keys,torch.full((len(x),),7,dtype=torch.int64),(zero,zero))
        return keys,state,logits

def initialize(agent):
    agent.focus_sender=JointSender(agent.focus_sender);agent.requires_grad_(False)
    for group in trainable_groups(agent).values():
        for v in group:v.requires_grad_(True)
    for v in agent.parameters():v.grad=None
    return dict(arm='joint',contrast_initialization='all zeros',counts={k:sum(v.numel() for v in g) for k,g in trainable_groups(agent).items()})

def trainable_groups(agent):
    return dict(sender=list(agent.focus_sender.parameters()),receiver=[v for name in old.RECEIVER_MODULES for v in getattr(agent,name).parameters()])

def message_log_probs(agent,x):
    keys,state,first=agent.focus_sender.start(x)
    second=torch.stack([F.log_softmax(agent.focus_sender.step(keys,torch.full((len(x),),t,dtype=torch.int64),state)[1],-1) for t in range(7)],1)
    return (F.log_softmax(first,-1)[:,:,None]+second).reshape(len(x),49)

def greedy_messages(agent,x):
    keys,state,first=agent.focus_sender.start(x);t0=first.argmax(-1).detach()
    _,second=agent.focus_sender.step(keys,t0,state)
    return torch.stack((t0,second.argmax(-1).detach()),1)

def direction_loss(sender,receiver,x,uniforms,positions,entropy_weight):
    assert sender is not receiver and not(set(map(id,sender.parameters()))&set(map(id,receiver.parameters())))
    assert uniforms.shape==(len(x),4) and uniforms.dtype==np.float32
    assert positions.shape==(len(x),2) and (positions[:,0]!=positions[:,1]).all()
    keys,state,first_logits=sender.focus_sender.start(x);u=torch.from_numpy(uniforms)
    first,lp0,e0,p0=old._draw(first_logits,u[:,0:1])
    _,second_logits=sender.focus_sender.step(keys,first,state)
    second,lp1,e1,p1=old._draw(second_logits,u[:,1:2])
    messages=torch.stack((first,second),1).detach();al=old.receive_messages(receiver,messages)
    food,lpf,ef,pf=old._draw(al[:,0],u[:,2:3]);water,lpw,ew,pw=old._draw(al[:,1],u[:,3:4])
    actions=torch.stack((food,water),1);success=(actions.numpy()==positions).astype(np.float32)
    reward=(.25*success.sum(1)+.5*success.prod(1)).astype(np.float32);adv=torch.from_numpy(reward-np.float32(.5))
    slp=lp0+lp1;rlp=lpf+lpw;se=e0+e1;re=ef+ew
    sl=-(slp*adv).mean()-float(entropy_weight)*se.mean();rl=-(rlp*adv).mean()-float(entropy_weight)*re.mean()
    n=lambda v:v.detach().numpy().copy()
    trace=dict(concepts=n(x),uniforms=uniforms.copy(),messages=n(messages),first_logits=n(first_logits),second_logits=n(second_logits),
        token_probabilities=n(torch.stack((p0,p1),1)),action_logits=n(al),action_probabilities=n(torch.stack((pf,pw),1)),
        actions=n(actions),positions=positions.copy(),success=success,reward=reward,advantage=n(adv),sender_logp=n(slp),receiver_logp=n(rlp),
        sender_entropy=n(se),receiver_entropy=n(re),sender_loss=float(sl.detach()),receiver_loss=float(rl.detach()),entropy_weight=float(entropy_weight),baseline=.5)
    return sl,rl,trace
