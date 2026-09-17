"""Independent visual policies with optional access to private episodic records."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

DIM = 64
HISTORY_LENGTH = 4
# role(2), own visual slots(4*64), visibility(4), message(4), own choice(4),
# shared success(1), record-valid flag(1). No hidden target or partner choice.
EVENT_DIM = 272


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),
            nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,32,3,2,1),nn.ReLU(),
            nn.Flatten(),nn.Linear(32*4*4,DIM))

    def forward(self,x):
        return F.normalize(self.net(x),dim=-1)


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=Encoder()
        self.visual_log_temperature=nn.Parameter(torch.tensor(3.0))
        self.history=nn.Sequential(nn.Linear(EVENT_DIM*HISTORY_LENGTH,64),nn.Tanh(),
                                   nn.Linear(64,64),nn.Tanh())
        self.sender=nn.Sequential(nn.Linear(128,64),nn.Tanh(),nn.Linear(64,4))
        self.symbol_embedding=nn.Embedding(4,64)
        self.listener=nn.Sequential(nn.Linear(128,64),nn.Tanh(),nn.Linear(64,64))
        self.listen_log_temperature=nn.Parameter(torch.tensor(2.5))

    def memory(self,h,enabled):
        # Exact same network is evaluated in both conditions.
        if not enabled: h=torch.zeros_like(h)
        return self.history(h.reshape(h.shape[0],-1))

    def send(self,target_features,h,enabled):
        return self.sender(torch.cat([target_features,self.memory(h,enabled)],dim=-1))

    def receive(self,candidate_features,symbol,h,enabled):
        query=F.normalize(self.listener(torch.cat([self.symbol_embedding(symbol),
                                                 self.memory(h,enabled)],dim=-1)),dim=-1)
        scale=self.listen_log_temperature.clamp(0,4.3).exp()
        return (candidate_features*query[:,None,:]).sum(-1)*scale

    def visual_match(self,target,candidates):
        target_z=self.encoder(target)
        candidate_z=self.encoder(candidates.flatten(0,1)).reshape(-1,4,DIM)
        return (candidate_z*target_z[:,None,:]).sum(-1)*self.visual_log_temperature.clamp(0,4.3).exp()

    def freeze_perception(self):
        for p in self.encoder.parameters(): p.requires_grad_(False)
        self.visual_log_temperature.requires_grad_(False)


def empty_history(device):
    return torch.zeros(HISTORY_LENGTH,EVENT_DIM,device=device)


def private_record(role,own_features,message,own_choice,reward):
    """Construct only the information available to this individual.

    role=0 sender: one privately seen image, no access to the partner's choice.
    role=1 receiver: four visible candidates and its own chosen position.
    Features use this individual's frozen visual encoder.
    """
    device=own_features.device
    event=torch.zeros(EVENT_DIM,device=device)
    event[role]=1
    if role==0:
        if own_choice is not None: raise ValueError('Sender cannot see receiver choice')
        event[2:2+DIM]=own_features.detach().reshape(DIM)
        event[258]=1
    elif role==1:
        event[2:258]=own_features.detach().reshape(4*DIM)
        event[258:262]=1
        event[266+int(own_choice)]=1
    else: raise ValueError('Unknown role')
    event[262+int(message)]=1
    event[270]=float(reward)
    event[271]=1
    return event


def append_record(history,event):
    return torch.cat([history[1:],event[None,:]],dim=0).detach()


def sample_policy(logits,uniform):
    """Separate policy randomness from the world's RNG; no differentiable message."""
    logp=F.log_softmax(logits,dim=-1)
    p=logp.exp()
    action=int((p.detach().cumsum(-1)<float(uniform)).sum().clamp(max=p.numel()-1).item())
    entropy=-(p*logp).sum()
    return action,logp[action],entropy
