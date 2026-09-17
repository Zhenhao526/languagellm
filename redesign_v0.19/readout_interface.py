"""Fresh frozen-interface private readouts; no optimizer or encoder updates.

Only one sampled goal enters reward/CE and entropy. All four experimental arms
share initialization, paired worlds, and action uniforms, not sampled actions.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'redesign_v0.13'))
import private_preparation as private
NAMESPACE=19019
SIGNALS=('reward','ce')
WORLD_KEYS=('triple_id','source_map','target_map','permutation','positions','photo_ids','goals','action_uniform')

def seed_sequence(seed,partition,who,purpose,step=0):
    assert isinstance(seed,(int,np.integer)) and seed>=0
    assert partition in (1,2,3) and who in (0,1) and purpose in (0,1,2) and step>=0
    return np.random.SeedSequence([NAMESPACE,int(seed),int(partition),int(who),int(purpose),int(step)])

def head_seed(seed,partition,who):
    """Only Torch initialization needs the recorded 63-bit bridge integer."""
    return int(seed_sequence(seed,partition,who,0,0).generate_state(1,dtype=np.uint64)[0]>>np.uint64(1))

def make_head(seed,partition,who):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(head_seed(seed,partition,who))
        head=nn.Sequential(nn.Linear(96,96),nn.Tanh(),nn.Linear(96,12))
    assert len(list(head.parameters()))==4 and sum(p.numel() for p in head.parameters())==10476
    return head

def fixture(bank,seed,partition,who,step,batch):
    assert batch>0 and isinstance(batch,(int,np.integer))
    rng=np.random.default_rng(seed_sequence(seed,partition,who,1,step))
    triple_id=rng.integers(len(private.legal_triples(partition)),size=batch)
    triples=private.legal_triples(partition)[triple_id]
    permutation=private.PERMUTATIONS[triples[:,1]]
    positions=np.stack((private.MAPS[triples[:,0]],private.MAPS[triples[:,2]]),axis=1)
    photo_ids=private.photo_ids(bank,rng,batch,'train')
    goals=rng.integers(2,size=batch)
    action_rng=np.random.default_rng(seed_sequence(seed,partition,who,2,step))
    uniforms=action_rng.random((2*batch,1)).astype(np.float32)
    assert np.array_equal(permutation[np.arange(batch)[:,None],positions[:,0]],positions[:,1])
    return dict(triple_id=triple_id,source_map=triples[:,0],target_map=triples[:,2],permutation=permutation,
        positions=positions,photo_ids=photo_ids,goals=goals,action_uniform=uniforms)

def loss_terms(head,h,world,signal,step):
    """Pure forward loss: returns (differentiable loss, JSON scalars, NumPy trace).

    h has source half then target half, corresponding to concatenation of the
    two paired-world positions. True locations never enter head(h).
    """
    assert signal in SIGNALS,signal
    assert isinstance(step,(int,np.integer)) and step>=0
    n=len(world['goals']);positions=np.concatenate((world['positions'][:,0],world['positions'][:,1]))
    goals=np.tile(world['goals'],2)
    assert positions.shape==(2*n,2) and goals.shape==(2*n,) and np.isin(goals,[0,1]).all()
    assert isinstance(h,torch.Tensor) and h.shape==(2*n,96) and h.dtype==torch.float32 and h.device.type=='cpu' and not h.requires_grad and bool(torch.isfinite(h).all())
    uniforms=np.asarray(world['action_uniform'])
    assert uniforms.shape==(2*n,1) and uniforms.dtype==np.float32 and (uniforms>=0).all() and (uniforms<=1).all()
    assert all(p.device.type=='cpu' and p.dtype==torch.float32 and p.requires_grad for p in head.parameters())
    full_logits=head(h).reshape(2*n,2,6)
    selected_logits=full_logits[torch.arange(2*n),torch.from_numpy(goals)]
    selected_log_probs=F.log_softmax(selected_logits,dim=-1)
    action=(selected_log_probs.detach().exp().cumsum(-1)<torch.from_numpy(uniforms)).sum(-1).clamp(max=5)
    places=action.detach().numpy()
    selected_target=positions[np.arange(2*n),goals]
    reward=(places==selected_target).astype(np.float32)
    logp=selected_log_probs.gather(1,action[:,None]).squeeze(1)
    reward_loss=-(logp*torch.from_numpy(reward-.5)).mean()
    ce_loss=F.cross_entropy(selected_logits,torch.from_numpy(selected_target))
    entropy=-(selected_log_probs.exp()*selected_log_probs).sum(-1).mean()
    entropy_weight=.02 if step<2100 else 0.
    task_loss=reward_loss if signal=='reward' else ce_loss
    loss=task_loss-entropy_weight*entropy
    assert bool(torch.isfinite(loss))
    components=dict(loss=float(loss.detach()),task_loss=float(task_loss.detach()),reward_loss=float(reward_loss.detach()),
        ce_loss=float(ce_loss.detach()),entropy=float(entropy.detach()),entropy_coefficient=entropy_weight,
        mean_reward=float(reward.mean()),signal=signal,rows=2*n,batch_pairs=n)
    trace=dict(full_logits=full_logits.detach().numpy(),goalselected_logits=selected_logits.detach().numpy(),
        selected_probabilities=selected_log_probs.detach().exp().numpy(),action_uniform=uniforms.copy(),action=places.copy(),
        reward=reward,selected_target=selected_target.copy(),flat_goals=goals.copy(),flat_positions=positions.copy())
    return loss,components,trace

def self_test():
    torch.set_num_threads(1)
    class Bank:pools={('train',0):np.arange(22),('train',1):np.arange(22,44)}
    bank=Bank();rng=torch.random.get_rng_state().clone()
    heads=[make_head(99513,1,0) for _ in range(4)]
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(all(torch.equal(p,q) for p,q in zip(heads[0].parameters(),x.parameters())) for x in heads[1:])
    seeds=[head_seed(s,p,d) for s in (31101,31102,31103,31104) for p in (1,2,3) for d in (0,1)]
    assert len(set(seeds))==24 and all(0<=s<2**63 for s in seeds)
    w=fixture(bank,99513,1,0,0,32);repeat=fixture(bank,99513,1,0,0,32)
    assert all(np.array_equal(w[k],repeat[k]) for k in WORLD_KEYS)
    assert set(w['source_map'])<=set(private.partition_maps(1)['old']) and set(w['target_map'])<=set(private.partition_maps(1)['old'])
    h=torch.from_numpy(np.random.default_rng(91773).normal(size=(64,96)).astype(np.float32))
    before={k:v.clone() for k,v in heads[0].state_dict().items()};global_rng=torch.random.get_rng_state().clone()
    losses={}
    for signal in SIGNALS:
        for step in (0,2099,2100,2399):
            loss,comp,trace=loss_terms(heads[0],h,w,signal,step)
            assert comp['entropy_coefficient']==(.02 if step<2100 else 0.)
            expected=comp['reward_loss'] if signal=='reward' else comp['ce_loss']
            assert abs(comp['loss']-(expected-comp['entropy_coefficient']*comp['entropy']))<1e-6
            assert trace['full_logits'].shape==(64,2,6) and trace['goalselected_logits'].shape==(64,6)
            grad=torch.autograd.grad(loss,list(heads[0].parameters()))
            assert all(torch.isfinite(g).all() for g in grad);losses[signal,step]=(float(loss.detach()),[g.clone() for g in grad])
    changed=dict(w,action_uniform=1.-w['action_uniform'])
    loss,comp,trace=loss_terms(heads[0],h,changed,'ce',0)
    grads=torch.autograd.grad(loss,list(heads[0].parameters()))
    assert float(loss.detach())==losses['ce',0][0] and all(torch.equal(g,old) for g,old in zip(grads,losses['ce',0][1]))
    assert all(p.grad is None for p in heads[0].parameters()) and not h.requires_grad
    assert all(torch.equal(v,heads[0].state_dict()[k]) for k,v in before.items())
    assert torch.equal(global_rng,torch.random.get_rng_state())
    return dict(passed=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),formal_inference=False,
        head_parameters=10476,head_tensors=4,formal_initialization_seeds=seeds,
        checks=['Four arms same fresh head, no global RNG mutation','24 formal head seed bridges unique and full63bit',
        'Fixture deterministic and old support only; paired goal/photos; direct SeedSequence namespaces',
        'Single selected goal reward/CE and entropy boundary2099/2100','CE loss and gradients unchanged by action-uniform intervention',
        'Finite gradients, no optimizer/encoder/parameter/grad-slot mutation'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true',required=True);p.parse_args()
    print(json.dumps(self_test(),ensure_ascii=False))
