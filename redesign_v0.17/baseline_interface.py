"""Shuffle only the sender policy baseline within its current private batch.

This module adds no model parameters. Value regression and receiver losses keep
all original targets. The identity branch supports exact historical replay.
"""
from __future__ import annotations
import argparse,copy,hashlib,json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

NAMESPACE=17017
SENDER=('send_',)
RECEIVER=('receive_embedding.','actor.','receive_value.')
TRACE_KEYS=('baseline_permutation','baseline_original','baseline_used',
            'baseline_target','baseline_logp','baseline_entropy')


def permutation_seed(seed,partition,step,who):
    """Direct SeedSequence identity; no integer compression or global RNG."""
    values=(seed,partition,step,who)
    if any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) or v<0 for v in values):
        raise ValueError('RNG identity requires nonnegative integers')
    return np.random.SeedSequence([NAMESPACE,*map(int,values)])


def sender_permutation(seed,partition,step,who,n):
    """Uniform permutation, including ordinary fixed points; never redrawn."""
    if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<=0:
        raise ValueError('batch size must be a positive integer')
    return np.random.default_rng(permutation_seed(seed,partition,step,who)).permutation(n).astype(np.int64,copy=False)


def _array(tensor):return tensor.detach().cpu().numpy().copy()
def _sha(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def _sorted_sha(array):
    # Sort the float32 bit patterns: repeated values and signed zero retain
    # exact multiplicity. This is a canonical multiset hash, not numeric order.
    assert array.dtype==np.float32
    return _sha(np.sort(array.view(np.uint32)))


def prepare_baseline(value,*,seed,partition,step,who,shuffle):
    """Return detached indexed values, deterministic indices and JSON metadata."""
    if value.ndim!=1 or value.dtype!=torch.float32 or not bool(torch.isfinite(value).all()):
        raise ValueError('expected a finite one-dimensional float32 value vector')
    n=len(value)
    if n<=0:raise ValueError('empty sender batch')
    index=sender_permutation(seed,partition,step,who,n) if shuffle else np.arange(n,dtype=np.int64)
    assert np.array_equal(np.sort(index),np.arange(n))
    used=value.detach()[torch.as_tensor(index,device=value.device)]
    original_array=_array(value);used_array=_array(used)
    original_sorted=_sorted_sha(original_array);used_sorted=_sorted_sha(used_array)
    assert original_sorted==used_sorted
    difference=used_array.astype(np.float64)-original_array.astype(np.float64)
    info=dict(mode='shuffle' if shuffle else 'identity',namespace=NAMESPACE,step_zero_based=int(step),
        permutation_seed_entropy=permutation_seed(seed,partition,step,who).entropy,n=n,
        permutation_sha256=_sha(index),fixed_point_count=int((index==np.arange(n)).sum()),
        original_sha256=_sha(original_array),used_sha256=_sha(used_array),
        original_sorted_sha256=original_sorted,used_sorted_sha256=used_sorted,
        sorted_hash_semantics='sorted uint32 bit patterns of float32 values',
        multiset_preserved=True,unchanged_value_count=int((original_array.view(np.uint32)==used_array.view(np.uint32)).sum()),
        changed_value_count=int(np.count_nonzero(difference)),
        original_std=float(np.std(original_array.astype(np.float64))),
        used_minus_original_mean_abs=float(np.mean(np.abs(difference))),
        used_minus_original_max_abs=float(np.max(np.abs(difference))),
        used_minus_original_rms=float(np.sqrt(np.mean(difference*difference))))
    return used,index,info


def update_agents(agents,opts,learning,weight,*,seed,partition,step,shuffle=True,return_trace=False):
    """Original v13 update with one sender-baseline substitution.

    Returns the original list of per-agent info records, with a new
    sender_baseline record. With return_trace=True returns (info, arrays).
    Each trace array has shape [2,n]; formal n=256. No optimizer is stepped
    until both agents' gradients have been calculated, matching v13 exactly.
    """
    if len(agents)!=2 or len(opts)!=2 or len(learning)!=2:raise ValueError('expected exactly two agents')
    if not isinstance(shuffle,bool) or not isinstance(return_trace,bool):raise TypeError('flags must be bool')
    plans=[]
    for who,entry in enumerate(learning):
        if set(entry)!={'sender','receiver'}:raise ValueError('both roles required')
        for role in ('sender','receiver'):
            lp,en,value,target=entry[role]
            if any(t.ndim!=1 or t.shape!=value.shape for t in (lp,en,target)):
                raise ValueError('role tensors must be equal-length vectors')
        plans.append(prepare_baseline(entry['sender'][2],seed=seed,partition=partition,step=step,who=who,shuffle=shuffle))
    traces={k:[] for k in TRACE_KEYS};info=[]
    for who,(a,opt) in enumerate(zip(agents,opts)):
        parts=[];losses=[];used,index,metadata=plans[who]
        for role in ('sender','receiver'):
            lp,en,value,target=learning[who][role]
            if role=='sender' and shuffle:
                policy=-(lp*(target-used).detach()).mean()
            else:
                # Keep the legacy operation order for development identity replay.
                policy=-(lp*(target-value).detach()).mean()
            vl=.5*F.mse_loss(value,target)
            part=policy+vl-weight*en.mean();losses.append(part)
            parts.append(dict(role=role,n=len(target),policy_loss=float(policy.detach()),
                value_loss=float(vl.detach()),entropy=float(en.mean().detach()),
                total=float(part.detach()),target_mean=float(target.mean())))
            if role=='sender' and return_trace:
                for key,item in zip(TRACE_KEYS,(index,_array(value),_array(used),_array(target),_array(lp),_array(en))):
                    traces[key].append(item)
        loss=torch.stack(losses).mean();assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();norms={}
        for role,prefixes in [('sender',SENDER),('receiver',RECEIVER)]:
            params=[p for key,p in a.named_parameters() if key.startswith(prefixes)]
            assert params and all(p.requires_grad and p.grad is not None for p in params)
            norms[role]=float(torch.nn.utils.clip_grad_norm_(params,2.))
        assert all(np.isfinite(x) for x in norms.values())
        assert all(p.grad is None for p in a.parameters() if not p.requires_grad)
        info.append(dict(loss=float(loss.detach()),parts=parts,gradient_norm_by_role=norms,sender_baseline=metadata))
    for opt in opts:opt.step()
    if return_trace:return info,{k:np.stack(v) for k,v in traces.items()}
    return info


def self_test():
    """Synthetic loss/gradient/Adam tests only; no real-agent training or data."""
    class Toy(nn.Module):
        def __init__(self):
            super().__init__();self.send_policy=nn.Linear(3,1);self.send_value=nn.Linear(3,1)
            self.receive_embedding=nn.Linear(3,2);self.actor=nn.Linear(2,1);self.receive_value=nn.Linear(3,1)
            self.project=nn.Linear(3,3)
            for p in self.project.parameters():p.requires_grad_(False)
        def learning(self,x):
            lp=torch.tanh(self.send_policy(x)).squeeze(1);v=self.send_value(x).squeeze(1)
            rp=torch.tanh(self.actor(self.receive_embedding(x))).squeeze(1);rv=self.receive_value(x).squeeze(1)
            t=torch.linspace(-1,0,len(x))
            return dict(sender=(lp,torch.sigmoid(lp),v,t),receiver=(rp,torch.sigmoid(rp),rv,t.flip(0)))
    def legacy(agents,opts,learning,weight):
        result=[]
        for who,(agent,opt) in enumerate(zip(agents,opts)):
            losses=[];parts=[]
            for role in ('sender','receiver'):
                lp,en,value,target=learning[who][role]
                pol=-(lp*(target-value).detach()).mean();vl=.5*F.mse_loss(value,target);loss=pol+vl-weight*en.mean()
                losses.append(loss);parts.append(dict(role=role,n=len(target),policy_loss=float(pol.detach()),value_loss=float(vl.detach()),entropy=float(en.mean().detach()),total=float(loss.detach()),target_mean=float(target.mean())))
            loss=torch.stack(losses).mean();opt.zero_grad(set_to_none=True);loss.backward()
            norms={role:float(torch.nn.utils.clip_grad_norm_([p for k,p in agent.named_parameters() if k.startswith(prefix)],2.)) for role,prefix in [('sender',SENDER),('receiver',RECEIVER)]}
            result.append(dict(loss=float(loss.detach()),parts=parts,gradient_norm_by_role=norms))
        for opt in opts:opt.step()
        return result
    def equal_tree(a,b):
        if isinstance(a,torch.Tensor):assert torch.equal(a,b)
        elif isinstance(a,dict):
            assert set(a)==set(b)
            for k in a:equal_tree(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):equal_tree(x,y)
        else:assert a==b
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(99517);original=[Toy(),Toy()];x=torch.randn(13,3)*.1
        agents=copy.deepcopy(original);refs=copy.deepcopy(original)
        makeopts=lambda aa:[torch.optim.Adam([p for p in a.parameters() if p.requires_grad],lr=.0007) for a in aa]
        opts=makeopts(agents);refopts=makeopts(refs)
        for step,weight in ((0,.02),(2100,0.)):
            rr=legacy(refs,refopts,[a.learning(x) for a in refs],weight)
            result,trace=update_agents(agents,opts,[a.learning(x) for a in agents],weight,seed=99517,partition=1,step=step,shuffle=False,return_trace=True)
            equal_tree([{k:v for k,v in item.items() if k!='sender_baseline'} for item in result],rr)
            equal_tree([a.state_dict() for a in agents],[a.state_dict() for a in refs]);equal_tree([o.state_dict() for o in opts],[o.state_dict() for o in refopts])
            assert np.array_equal(trace['baseline_original'],trace['baseline_used'])
        refs=copy.deepcopy(original);agents=copy.deepcopy(original);refopts=makeopts(refs);opts=makeopts(agents)
        original_rng=torch.random.get_rng_state().clone();numpy_rng=np.random.get_state()
        result,trace=update_agents(agents,opts,[a.learning(x) for a in agents],.02,seed=99517,partition=1,step=0,shuffle=True,return_trace=True)
        assert torch.equal(original_rng,torch.random.get_rng_state());equal_tree(numpy_rng[0],np.random.get_state()[0]);assert np.array_equal(numpy_rng[1],np.random.get_state()[1])
        rr=legacy(refs,refopts,[a.learning(x) for a in refs],.02)
        assert all(q['gradient_norm_by_role']['sender']<2 for q in result+rr)
        for who,(a,b) in enumerate(zip(agents,refs)):
            assert not torch.equal(a.send_policy.weight.grad,b.send_policy.weight.grad)
            for k,p in a.named_parameters():
                other=dict(b.named_parameters())[k]
                if k.startswith(('send_value.',)+RECEIVER):assert torch.equal(p.grad,other.grad) and torch.equal(p,other),k
            idx=trace['baseline_permutation'][who]
            assert np.array_equal(trace['baseline_used'][who],trace['baseline_original'][who][idx])
        assert set(trace)==set(TRACE_KEYS) and all(v.shape==(2,13) for v in trace.values())
        seed0=sender_permutation(99517,1,0,0,256)
        assert np.array_equal(seed0,sender_permutation(99517,1,0,0,256))
        assert not np.array_equal(seed0,sender_permutation(99517,1,0,1,256))
        assert not np.array_equal(seed0,sender_permutation(99517,1,1,0,256))
        signed=torch.tensor([0.,-0.,1.,1.,-2.],dtype=torch.float32,requires_grad=True)
        used,_,record=prepare_baseline(signed,seed=1,partition=1,step=0,who=0,shuffle=True)
        assert not used.requires_grad and record['multiset_preserved']
    return dict(passed=True,namespace=NAMESPACE,trace_keys=list(TRACE_KEYS),checks=[
        'Synthetic identity two updates match legacy losses/gradients/parameters/Adam exactly',
        'Sender policy gradients change with shuffled baseline; value and receiver gradients stay exact at common inputs when clip inactive',
        'Original targets retained; trace records exact permutation of detached values',
        'Uniform permutation deterministic by personal update identity, not global RNG',
        'Float32 bit-multiset hash preserves duplicates and signed zero',
        'No real agents, images, sealed results or training runs used'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true',required=True);p.parse_args()
    print(json.dumps(self_test(),ensure_ascii=False))
