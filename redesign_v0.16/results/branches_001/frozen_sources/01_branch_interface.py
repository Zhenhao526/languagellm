"""Fixed policy/value input scales with unchanged raw visual observation.

The only new state is send_context.h_scale and send_value.h_scale. Use
load_scaled_state before loading a saved branch-scaled checkpoint. Helpers
policy_h/value_h are recording views; pass raw observe output to agent.send.
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import sys
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'redesign_v0.8'))
from camp import CampAgent,remake_agents

WIDTH=96
BUFFER_KEYS=('send_context.h_scale','send_value.h_scale')
ACTIVE_PREFIXES=('send_','receive_embedding.','actor.','receive_value.')


def _scalar(value,reference):
    value=float(value)
    if not np.isfinite(value) or value<=0:raise ValueError('scale must be finite and positive')
    if reference.dtype!=torch.float32:raise ValueError('branch scale requires a float32 model')
    result=reference.new_tensor(value)
    if not bool(torch.isfinite(result)) or not bool(result>0):raise ValueError('scale is not finite positive in float32')
    return result


class PrefixScaledSequential(nn.Sequential):
    """Multiply only the first 96 input columns before the existing first layer."""
    def forward(self,local):
        if local.ndim!=2 or local.shape[1]!=WIDTH+4:raise ValueError('expected raw h96 plus four local features')
        value=torch.cat((local[:,:WIDTH]*self.h_scale,local[:,WIDTH:]),dim=1)
        return super().forward(value)


def attach_scales(agent,policy_scale,value_scale):
    """Attach fixed branch buffers without changing parameters, observe or RNG.

    The passed agent must be the original raw-observation CampAgent. A v15
    ScaledCampAgent is rejected, so a previous whole-h scale cannot be applied
    again. Reattaching the same values is idempotent; changing them is rejected.
    """
    if type(agent) is not CampAgent or agent.width!=WIDTH:raise TypeError('expected original raw CampAgent with width96')
    if hasattr(agent,'visual_scale'):raise ValueError('whole-observation scale must not be present')
    modules=(agent.send_context,agent.send_value)
    if any(type(m) not in (nn.Sequential,PrefixScaledSequential) for m in modules):raise TypeError('unexpected branch module type')
    if any(not isinstance(m[0],nn.Linear) or m[0].in_features!=WIDTH+4 for m in modules):raise ValueError('unexpected first-layer shape')
    values=(_scalar(policy_scale,modules[0][0].weight),_scalar(value_scale,modules[1][0].weight))
    installed=[isinstance(m,PrefixScaledSequential) for m in modules]
    if any(installed):
        if not all(installed):raise ValueError('partially wrapped state is not supported')
        if any(not torch.equal(m.h_scale,v) for m,v in zip(modules,values)):raise ValueError('fixed branch scales cannot be changed')
        return agent
    if any(hasattr(m,'h_scale') for m in modules):raise ValueError('h_scale already exists')
    for module,value in zip(modules,values):
        module.__class__=PrefixScaledSequential
        module.register_buffer('h_scale',value,persistent=True)
    return agent


def load_scaled_state(agent,state):
    """Register both stored buffers, then load the whole state strictly."""
    if 'visual_scale' in state:raise ValueError('v15 whole-h checkpoints must not load as branch checkpoints')
    for key in BUFFER_KEYS:
        if key not in state or state[key].ndim!=0 or state[key].dtype!=torch.float32:raise ValueError('checkpoint requires two scalar float32 branch buffers')
    attach_scales(agent,*(state[k].detach().cpu().item() for k in BUFFER_KEYS))
    agent.load_state_dict(state,strict=True)
    return agent


def remake_branch_agents(seed,prepared,states):
    agents=remake_agents(seed,prepared,7,2,'identity')
    if len(states)!=2:raise ValueError('expected two personal states')
    for agent,state in zip(agents,states):load_scaled_state(agent,state)
    return agents


def raw_h(agent,visual,delay=0,memory_mode='retain'):
    """Raw observation, explicitly named for protocol artifacts; no scaling."""
    if type(agent) is not CampAgent or hasattr(agent,'visual_scale'):raise TypeError('raw-observation agent required')
    return agent.observe(visual,delay,memory_mode)


def policy_h(agent,raw):
    """Effective policy view for recording only; never feed this into send again."""
    if raw.ndim!=2 or raw.shape[1]!=WIDTH:raise ValueError('expected raw h96')
    return raw*agent.send_context.h_scale


def value_h(agent,raw):
    """Effective value view for recording only; not an additional model input."""
    if raw.ndim!=2 or raw.shape[1]!=WIDTH:raise ValueError('expected raw h96')
    return raw*agent.send_value.h_scale


def _social_flags(agent):
    for key,p in agent.named_parameters():p.requires_grad_(key.startswith(ACTIVE_PREFIXES))


def self_test():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(99516)
        base=CampAgent(nn.Sequential(nn.Linear(8,64)),7,2);_social_flags(base)
        original=copy.deepcopy(base.state_dict());flags={k:p.requires_grad for k,p in base.named_parameters()}
        alpha=float(np.float32(5.23456));agents={}
        for key,scales in {'00':(1.,1.),'10':(alpha,1.),'01':(1.,alpha),'11':(alpha,alpha)}.items():
            a=copy.deepcopy(base);pointers={k:p.data_ptr() for k,p in a.named_parameters()};rng=torch.random.get_rng_state().clone()
            attach_scales(a,*scales)
            assert torch.equal(rng,torch.random.get_rng_state())
            assert pointers=={k:p.data_ptr() for k,p in a.named_parameters()}
            assert flags=={k:p.requires_grad for k,p in a.named_parameters()}
            assert set(a.state_dict())==set(original)|set(BUFFER_KEYS)
            assert all(torch.equal(v,a.state_dict()[k]) for k,v in original.items())
            assert not any(k in dict(a.named_parameters()) for k in BUFFER_KEYS)
            agents[key]=a
        visual=torch.randn(19,390);goal=torch.eye(2)[torch.arange(19)%2];inventory=torch.rand(19,2)*2
        with torch.no_grad():
            for a in agents.values():
                for mode in ('retain','reset','replay'):
                    for delay in (0,2):assert torch.equal(raw_h(a,visual,delay,mode),base.observe(visual,delay,mode))
                assert not policy_h(a,a.observe(visual,memory_mode='reset')).any()
                assert not value_h(a,a.observe(visual,memory_mode='reset')).any()
            for greedy in (False,True):
                ref0=base.send(base.observe(visual),goal,inventory,np.random.default_rng(16),greedy)
                ref1=base.send(base.observe(visual)*np.float32(alpha),goal,inventory,np.random.default_rng(16),greedy)
                result={k:a.send(a.observe(visual),goal,inventory,np.random.default_rng(16),greedy) for k,a in agents.items()}
                assert all(torch.equal(x,y) for x,y in zip(result['00'],ref0))
                assert all(torch.equal(x,y) for x,y in zip(result['11'],ref1))
                assert all(torch.equal(x,y) for x,y in zip(result['10'][:3],ref1[:3])) and torch.equal(result['10'][3],ref0[3])
                assert all(torch.equal(x,y) for x,y in zip(result['01'][:3],ref0[:3])) and torch.equal(result['01'][3],ref1[3])
            raw=base.observe(visual);local=torch.cat((raw,goal,inventory/2),1)
            for key,a in agents.items():
                expected_p=torch.cat((policy_h(a,raw),goal,inventory/2),1)
                expected_v=torch.cat((value_h(a,raw),goal,inventory/2),1)
                assert torch.equal(a.send_context(local),base.send_context(expected_p))
                assert torch.equal(a.send_value(local),base.send_value(expected_v))
                fresh=copy.deepcopy(base);load_scaled_state(fresh,a.state_dict())
                assert all(torch.equal(v,fresh.state_dict()[k]) for k,v in a.state_dict().items())
                assert torch.equal(fresh.send_context(local),a.send_context(local))
        # Artificial gradient equality only: no optimizer or training step.
        for key,scale in (('00',1.),('11',alpha)):
            lhs=copy.deepcopy(base);rhs=copy.deepcopy(agents[key]);target=torch.linspace(-1.,0.,len(visual))
            outputs=[]
            for a,hscale in ((lhs,scale),(rhs,1.)):
                h=a.observe(visual)*hscale
                _,lp,en,v=a.send(h,goal,inventory,np.random.default_rng(17),False)
                loss=-(lp*(target-v).detach()).mean()+.5*F.mse_loss(v,target)-.02*en.mean()
                loss.backward();outputs.append(loss.detach())
            assert torch.equal(*outputs)
            for (k,x),(l,y) in zip(lhs.named_parameters(),rhs.named_parameters()):
                assert k==l
                assert x.grad is None and y.grad is None or x.grad is not None and y.grad is not None and torch.equal(x.grad,y.grad),k
        for bad in (0.,-1.,float('nan'),float('inf'),1e-100,1e100):
            a=copy.deepcopy(base)
            try:attach_scales(a,1.,bad)
            except ValueError:assert set(a.state_dict())==set(original)
            else:raise AssertionError('invalid scale accepted')
    return dict(passed=True,checks=['Only two persistent buffers; original parameters/flags/storage/RNG unchanged',
        'Raw observe unchanged for all modes and delays','00/11 exactly reproduce unscaled/whole-h-scale send and artificial loss gradients',
        '10/01 policy and value outputs follow only their assigned factor','Only first96 local features scaled; direct protocol context path correct',
        'Checkpoint strict reload reproduces outputs','Invalid multiplier rejected before mutation'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true',required=True);p.parse_args()
    print(json.dumps(self_test()))
