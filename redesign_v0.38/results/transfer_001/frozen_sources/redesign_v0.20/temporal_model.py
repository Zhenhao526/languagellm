"""Two legal visual frames through the existing CampAgent recurrent interface.

'full' and 'detach' have identical forward state transitions at equal weights;
only credit assignment across the h0 boundary differs. No temporal/task/position
truth is sent to the private action head. Erasure is an evaluation information
ablation, not the detach training condition.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS=('full','detach')
LAYER_NORM_EPS=1e-5
WIDTH=96
VISUAL_WIDTH=390


def make_head(init_seed:int):
    """Fresh10476-parameter private head; caller owns recorded seed identity."""
    if not isinstance(init_seed,(int,np.integer)) or not 0<=init_seed<2**63:raise ValueError('expected explicit63bit initialization seed')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(init_seed))
        head=nn.Sequential(nn.Linear(96,96,dtype=torch.float32),nn.Tanh(),nn.Linear(96,12,dtype=torch.float32))
    assert len(list(head.parameters()))==4 and sum(p.numel() for p in head.parameters())==10476
    return head


def _sequence_states(agent,frames,view_bits,arm,erase_history=False):
    if arm not in ARMS:raise ValueError(f'unknown arm {arm!r}')
    if not isinstance(erase_history,bool):raise TypeError('erase_history must be bool')
    if erase_history and torch.is_grad_enabled():raise ValueError('history erasure is evaluation only; use no_grad')
    if not isinstance(frames,torch.Tensor) or frames.ndim!=3 or frames.shape[1:]!=(2,VISUAL_WIDTH) or frames.dtype!=torch.float32 or frames.device.type!='cpu':
        raise ValueError('expected CPU float32 frames[B,2,390] in original 384visual+6exists layout')
    if len(frames)==0 or not bool(torch.isfinite(frames).all()):raise ValueError('frames must be nonempty and finite')
    if agent.width!=WIDTH or agent.memory.input_size!=VISUAL_WIDTH+1 or agent.memory.hidden_size!=WIDTH:raise ValueError('expected original391-to96 CampAgent memory')
    if hasattr(agent,'visual_scale'):raise ValueError('use plain CampAgent; v20 has one final nonaffine LayerNorm, not v15 scaling')
    bits=torch.as_tensor(view_bits,dtype=frames.dtype,device=frames.device)
    if bits.shape!=(len(frames),2) or bits.requires_grad or not bool(torch.isfinite(bits).all()) or not bool(((bits==0)|(bits==1)).all()) or not bool((bits[:,0]==1).all()):
        raise ValueError('view bits must be fixed[B,2] with first full=1 and second full1/partial0')
    zero=frames.new_zeros(len(frames),WIDTH)
    encoded0=agent.encode_slots(frames[:,0])
    seen0=torch.cat((encoded0,bits[:,0,None]),dim=-1)
    h0=agent.memory(seen0,zero)
    previous=torch.zeros_like(h0) if erase_history else (h0.detach() if arm=='detach' else h0)
    encoded1=agent.encode_slots(frames[:,1])
    seen1=torch.cat((encoded1,bits[:,1,None]),dim=-1)
    h1=agent.memory(seen1,previous)
    final=F.layer_norm(h1,(WIDTH,),weight=None,bias=None,eps=LAYER_NORM_EPS)
    return final,h0,h1


def observe_sequence(agent,frames,view_bits,arm,erase_history=False):
    """Return normalized h1; no demand, coordinates or map IDs are accepted."""
    return _sequence_states(agent,frames,view_bits,arm,erase_history)[0]


def loss_terms(agent,head,frames,view_bits,goals,positions,uniform,arm,step):
    """Selected-goal REINFORCE loss; optimizer and parameter flags belong to caller.

    Frames are only projected legal observations. Positions are evaluator truth
    used to return binary action success, never to construct head input. No
    extra JSD or CE is computed. No .grad accumulation or parameter mutation.
    """
    if not isinstance(step,(int,np.integer)) or step<0:raise ValueError('step must be a nonnegative zero-based integer')
    n=len(frames);goals=np.asarray(goals);positions=np.asarray(positions);uniform=np.asarray(uniform)
    if goals.shape!=(n,) or not np.issubdtype(goals.dtype,np.integer) or not np.isin(goals,[0,1]).all():raise ValueError('goals must be integer[B] in0,1')
    if positions.shape!=(n,2) or not np.issubdtype(positions.dtype,np.integer) or not ((positions>=0)&(positions<6)).all() or (positions[:,0]==positions[:,1]).any():raise ValueError('positions must be distinct resource locations[B,2]')
    if uniform.shape!=(n,1) or uniform.dtype!=np.float32 or not ((uniform>=0)&(uniform<=1)).all():raise ValueError('uniform must be float32[B,1] in0..1')
    h,h0,h1=_sequence_states(agent,frames,view_bits,arm)
    full_logits=head(h).reshape(n,2,6)
    selected_logits=full_logits[torch.arange(n),torch.from_numpy(goals.astype(np.int64,copy=False))]
    log_probs=F.log_softmax(selected_logits,dim=-1)
    action=(log_probs.detach().exp().cumsum(-1)<torch.from_numpy(uniform)).sum(-1).clamp(max=5)
    places=action.detach().numpy();target=positions[np.arange(n),goals]
    reward=(places==target).astype(np.float32)
    logp=log_probs.gather(1,action[:,None]).squeeze(1)
    policy=-(logp*torch.from_numpy(reward-.5)).mean()
    entropy=-(log_probs.exp()*log_probs).sum(-1).mean()
    entropy_coefficient=.02 if step<2100 else 0.
    loss=policy-entropy_coefficient*entropy
    if not bool(torch.isfinite(loss)):raise ValueError('non-finite temporal loss')
    components=dict(loss=float(loss.detach()),policy_loss=float(policy.detach()),entropy=float(entropy.detach()),
        entropy_coefficient=entropy_coefficient,mean_reward=float(reward.mean()),arm=arm,rows=n)
    trace=dict(h0=h0.detach().numpy(),raw_h1=h1.detach().numpy(),hfinal=h.detach().numpy(),
        full_logits=full_logits.detach().numpy(),goalselected_logits=selected_logits.detach().numpy(),
        selected_probabilities=log_probs.detach().exp().numpy(),action_uniform=uniform.copy(),action=places.copy(),
        reward=reward,selected_target=target.copy())
    return loss,components,trace


def self_test():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'redesign_v0.8'))
    from camp import CampAgent
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(99520)
        agent=CampAgent(nn.Sequential(nn.Linear(8,64)),7,2)
        for name,p in agent.named_parameters():p.requires_grad_(name.startswith(('memory.','slot_phi.')))
        initial={k:v.clone() for k,v in agent.state_dict().items()}
        rng=torch.random.get_rng_state().clone();head=make_head(9920051);twin=make_head(9920051)
        assert torch.equal(rng,torch.random.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(head.parameters(),twin.parameters()))
        frames=torch.randn(5,2,390,requires_grad=True)
        bits=torch.tensor([[1.,0.],[1.,1.],[1.,0.],[1.,1.],[1.,0.]])
        weight=torch.linspace(-.7,1.1,96)[None,:]
        full,h0,h1=_sequence_states(agent,frames,bits,'full')
        detached,d0,d1=_sequence_states(agent,frames,bits,'detach')
        assert torch.equal(full,detached) and torch.equal(h0,d0) and torch.equal(h1,d1)
        lossf=(full*weight).sum();lossd=(detached*weight).sum()
        fg,hg=torch.autograd.grad(lossf,(frames,h0),retain_graph=True)
        dg,dhg=torch.autograd.grad(lossd,(frames,d0),retain_graph=True,allow_unused=True)
        assert bool((fg[:,0]!=0).any()) and bool((fg[:,1]!=0).any()) and bool((hg!=0).any())
        assert torch.count_nonzero(dg[:,0])==0 and bool((dg[:,1]!=0).any()) and dhg is None
        params=[p for name,p in agent.named_parameters() if name.startswith(('memory.','slot_phi.'))]
        names=[name for name,p in agent.named_parameters() if name.startswith(('memory.','slot_phi.'))]
        gfull=torch.autograd.grad(lossf,params,retain_graph=True)
        gdetach=torch.autograd.grad(lossd,params,retain_graph=True)
        # Full-gradient decomposition: current-step path + the h0 learning path.
        first_path=torch.autograd.grad(h0,params,grad_outputs=hg,retain_graph=True)
        max_decomposition_error=max(float((a-b-c).abs().max()) for a,b,c in zip(gfull,gdetach,first_path))
        assert all(torch.allclose(a,b+c,atol=3e-6,rtol=3e-5) for a,b,c in zip(gfull,gdetach,first_path)),max_decomposition_error
        for prefix in ('memory.','slot_phi.'):
            assert any(bool((g!=0).any()) for name,g in zip(names,gdetach) if name.startswith(prefix))
        with torch.no_grad():
            erased=observe_sequence(agent,frames,bits,'full',erase_history=True)
            changed=frames.detach().clone();changed[:,0]*=-2.
            assert torch.equal(erased,observe_sequence(agent,changed,bits,'detach',erase_history=True))
            assert not torch.equal(erased,observe_sequence(agent,frames,bits,'full'))
            direct=agent.memory(torch.cat((agent.encode_slots(frames[:,1]),bits[:,1,None]),-1),torch.zeros(5,96))
            assert torch.equal(erased,F.layer_norm(direct,(96,),eps=LAYER_NORM_EPS))
        for arm in ARMS:
            for step in (2099,2100):
                loss,components,trace=loss_terms(agent,head,frames.detach(),bits,np.asarray([0,1,0,1,0]),
                    np.asarray([[0,1],[1,2],[2,3],[3,4],[4,5]]),np.linspace(.1,.9,5,dtype=np.float32)[:,None],arm,step)
                assert components['entropy_coefficient']==(.02 if step==2099 else 0.)
                gradients=torch.autograd.grad(loss,params+list(head.parameters()))
                assert all(torch.isfinite(g).all() for g in gradients)
                assert trace['hfinal'].shape==(5,96) and trace['full_logits'].shape==(5,2,6)
        assert all(p.grad is None for p in list(agent.parameters())+list(head.parameters()))
        assert all(torch.equal(v,agent.state_dict()[k]) for k,v in initial.items())
        assert agent.input_transform.requires_grad is False and all(not p.requires_grad for p in agent.project.parameters())
    return dict(passed=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),training_run=False,
        memory_slot_trainable_parameters=sum(p.numel() for p in params),head_parameters=sum(p.numel() for p in head.parameters()),
        max_full_gradient_decomposition_error=max_decomposition_error,
        checks=['Full/detach forward h0/rawh1/normalized h identical at equal weights','First-frame and h0 gradient cut only by detach',
            'Second-frame and shared memory/slot parameters retain gradients','Full gradient equals current-step plus historical path',
            'History erasure is evaluation-only and invariant to first-frame changes','No goals/positions passed to head input',
            'One-goal reward loss and entropy boundary2099/2100','No parameter/grad-slot mutation; freshhead no external RNG mutation'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true',required=True);p.parse_args()
    print(json.dumps(self_test(),ensure_ascii=False))
