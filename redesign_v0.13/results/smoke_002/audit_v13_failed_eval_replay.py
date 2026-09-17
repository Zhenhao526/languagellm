"""Independent audit of private spatial-correspondence preparation and transfer.

This module does not import a training routine or reuse its loss implementation.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
from itertools import permutations,product
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
SEEDS=(31101,31102,31103,31104)
ARMS=('control','equivariant')
MAPS=np.array(list(permutations(range(6),2)),dtype=np.int64)
PERMS=np.array(list(permutations(range(6))),dtype=np.int64)
MATCHINGS=(((0,1),(2,3),(4,5)),((0,2),(1,4),(3,5)),((0,3),(1,5),(2,4)))


class Audit:
    def __init__(self):self.checks=Counter();self.failures=[];self.counts=Counter()
    def check(self,ok,name,context=None):
        self.checks[name]+=1
        if not bool(ok):self.failures.append(dict(check=name,context=context))


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def arrays(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}
def load(path):return torch.load(path,map_location='cpu',weights_only=True)
def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if type(a)!=type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b
def array_sha(*values):
    h=hashlib.sha256()
    for value in values:h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def groups(partition):
    def held(k):
        chosen={x for pair in MATCHINGS[k] for x in (pair,pair[::-1])}
        return np.array([i for i,x in enumerate(MAPS) if tuple(x) in chosen],dtype=np.int64)
    added=held(partition-1);sealed=held(partition%3)
    return dict(old=np.setdiff1d(np.arange(30),np.r_[added,sealed]),added=added,sealed=sealed)


def triples(partition):
    """Each row is source map index, lexicographic S6 index, target map index."""
    lookup={tuple(x):i for i,x in enumerate(MAPS)}
    old=groups(partition)['old'];allowed=set(old);out=[]
    for source in old:
        for g,perm in enumerate(PERMS):
            target=lookup[tuple(perm[MAPS[source]])]
            if target in allowed:out.append((source,g,target))
    return np.array(out,dtype=np.int64)


def logp(logits):
    x=np.asarray(logits,dtype=np.float64)
    return x-np.logaddexp.reduce(x,axis=-1,keepdims=True)


def pushed_probability(p,perms):
    return np.take_along_axis(p,np.argsort(perms,axis=-1)[:,None,:],axis=-1)


def jsd_probability(p,q):
    p=np.asarray(p,dtype=float);q=np.asarray(q,dtype=float)
    middle=(p+q)/2
    out=np.zeros_like(p)
    positive=p>0;out[positive]+=p[positive]*(np.log(p[positive])-np.log(middle[positive]))/2
    positive=q>0;out[positive]+=q[positive]*(np.log(q[positive])-np.log(middle[positive]))/2
    return out.sum(-1)


def jsd_logits(a,b,perms):
    return float(jsd_probability(np.exp(logp(b)),pushed_probability(np.exp(logp(a)),perms)).mean())


def mathematical_tests(audit):
    for p in (1,2,3):
        t=triples(p);old=groups(p)['old'];counts=Counter(map(tuple,t[:,[0,2]]))
        audit.check(t.shape==(7776,3) and len(set(map(tuple,t)))==7776,'exact_7776_legal_triples',p)
        audit.check(counts==Counter({(x,y):24 for x,y in product(old,repeat=2)}),'each_source_target_has_24_S6_maps',p)
        audit.check(np.array_equal(PERMS[t[:,1]][np.arange(len(t))[:,None],MAPS[t[:,0]]],MAPS[t[:,2]]),
                    'triples_are_forward_resource_site_maps',p)
        audit.check(not np.isin(t[:,[0,2]],np.r_[groups(p)['added'],groups(p)['sealed']]).any(),
                    'both_sides_exclude_added_sealed',p)
        for group,pool in groups(p).items():
            audit.check(all(np.array_equal(np.bincount(MAPS[pool,k],minlength=6),np.full(6,len(pool)//6)) for k in (0,1)),
                        'resource_location_marginals_balanced',[p,group])
    # Non-self-inverse permutation prevents a false-positive test of gather direction.
    perm=np.array([[1,3,5,0,2,4]])
    p=np.eye(6)[[2,3]][None]
    q=np.eye(6)[perm[0,[2,3]]][None]
    audit.check(np.array_equal(pushed_probability(p,perm),q),'noninvolutive_pushforward_uses_inverse_gather')
    audit.check(float(jsd_probability(q,pushed_probability(p,perm)).mean())==0.,'perfect_correspondence_JSD_zero')
    uniform=np.full((1,2,6),1/6)
    audit.check(float(jsd_probability(uniform,pushed_probability(uniform,perm)).mean())==0.,
                'uniform_noninformative_policy_also_has_zero_JSD')
    wrong=q[:,:,::-1].copy()
    audit.check(np.isclose(jsd_probability(q,wrong).max(),np.log(2)),'disjoint_point_masses_JSD_ln2')
    rng=np.random.default_rng(130130)
    for i in range(12):
        a=rng.normal(size=(5,2,6))*3;b=rng.normal(size=(5,2,6))*3;g=PERMS[rng.integers(720,size=5)]
        value=jsd_logits(a,b,g)
        audit.check(-1e-12<=value<=np.log(2)+1e-12,'random_policy_JSD_bounds',i)
        moved=pushed_probability(np.exp(logp(a)),g)
        audit.check(np.isclose(value,jsd_probability(moved,np.exp(logp(b))).mean()),'JSD_two_sides_symmetric',i)
        # Derivative of JSD wrt each logit, including the probability normalization.
        pa=np.exp(logp(a));pb=np.exp(logp(b));mp=pushed_probability(pa,g);mix=(mp+pb)/2
        db=.5*(np.log(pb)-np.log(mix));gradb=pb*(db-(db*pb).sum(-1,keepdims=True))/10
        da=.5*(np.log(mp)-np.log(mix));gradm=mp*(da-(da*mp).sum(-1,keepdims=True))/10
        grada=np.take_along_axis(gradm,g[:,None,:],axis=-1)
        loc=(i%5,(i//5)%2,i%6);eps=1e-5
        aa=a.copy();aa[loc]+=eps;ab=a.copy();ab[loc]-=eps
        ba=b.copy();ba[loc]+=eps;bb=b.copy();bb[loc]-=eps
        audit.check(np.isclose((jsd_logits(aa,b,g)-jsd_logits(ab,b,g))/(2*eps),grada[loc],atol=2e-10,rtol=2e-5),
                    'JSD_source_gradient_through_permutation',i)
        audit.check(np.isclose((jsd_logits(a,ba,g)-jsd_logits(a,bb,g))/(2*eps),gradb[loc],atol=2e-10,rtol=2e-5),
                    'JSD_target_gradient_not_detached',i)


PRIVATE=('memory.','slot_phi.')
SOCIAL=('send_','receive_embedding.','actor.','receive_value.')
PKEYS=('triple_id','source_map','target_map','permutation','positions','photo_ids','goals')
SKEYS=('positions','photo_ids','goals','menu')

def named_arrays_sha(d,keys=None):
    h=hashlib.sha256()
    for k in sorted(d) if keys is None else keys:
        x=np.ascontiguousarray(d[k]);h.update(k.encode());h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
    return h.hexdigest()

def seed_number(*identity):return int(np.random.SeedSequence(identity).generate_state(1,np.uint64)[0])//2

def private_world(bank,seed,p,who,step,n,table):
    rng=np.random.default_rng(seed_number(13013,seed,p,who,1,step));ix=rng.integers(len(table),size=n);rows=table[ix]
    return dict(triple_id=ix,source_map=rows[:,0],target_map=rows[:,2],permutation=PERMS[rows[:,1]],
        positions=np.stack((MAPS[rows[:,0]],MAPS[rows[:,2]]),axis=1),
        photo_ids=np.column_stack([rng.choice(bank.pools['train',k],n) for k in (0,1)]),goals=rng.integers(2,size=n))

def social_worlds(bank,seed,p,step,n,training):
    out=[]
    for d in (0,1):
        rng=np.random.default_rng(seed_number(13014,seed,p,1 if training else 90,step,d));m=n//2
        if training:pos=MAPS[rng.choice(groups(p)['old'],m)].copy();goal=rng.integers(2,size=m)
        else:
            ix=np.tile(np.array(list(product(range(30),range(2)))),(m//60,1));ix=ix[rng.permutation(m)];pos=MAPS[ix[:,0]].copy();goal=ix[:,1]
        out.append(dict(positions=pos,inventory=np.zeros((m,2),np.int64),photo_ids=np.column_stack([rng.choice(bank.pools['train' if training else 'test',k],m) for k in (0,1)]),goals=np.column_stack((goal,1-goal)),menu=np.argsort(rng.random((m,2,6)),axis=2),history=np.zeros((m,18),np.float32)))
    return out

def world_digest(worlds):return array_sha(*(w[k] for w in worlds for k in SKEYS))

def import_replay():
    import importlib.util
    path=PROJECT/'redesign_v0.11/audit_query_replay.py'
    spec=importlib.util.spec_from_file_location('v13_independent_replay',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def compare_tensors(audit,actual,expected,label,context):
    if isinstance(actual,dict):
        audit.check(actual.keys()==expected.keys(),label+'_keys',context)
        for key in actual:compare_tensors(audit,actual[key],expected[key],label,context+[str(key)])
    elif isinstance(actual,(tuple,list)):
        audit.check(len(actual)==len(expected),label+'_length',context)
        for i,(x,y) in enumerate(zip(actual,expected)):compare_tensors(audit,x,y,label,context+[i])
    else:audit.check(same(actual,expected),label,context)

def private_first(folder,bank,replay,audit,step=1):
    from torch.nn import functional as F
    cfg=read(folder/'config.json');seed=cfg['seed'];p=cfg['partition'];n=cfg['batch_pairs_per_person']
    prepared=load(cfg['prepared_source']['path']);initial=load(folder/('initial.pt' if step==1 else 'forensic_2100.pt'));agents=replay.remake_agents(seed,prepared,7,2,'identity');heads=[];opts=[]
    for who,a in enumerate(agents):
        a.load_state_dict(initial['agents'][who]);torch.manual_seed(seed*1000+881+who)
        head=torch.nn.Sequential(torch.nn.Linear(96,96),torch.nn.Tanh(),torch.nn.Linear(96,12));audit.check(step!=1 or same(head.state_dict(),initial['heads'][who]),'private_head_fresh_seed',[folder.name,who]);head.load_state_dict(initial['heads'][who]);heads.append(head)
        for key,param in a.named_parameters():param.requires_grad_(key.startswith(PRIVATE))
        opts.append(torch.optim.Adam(list(head.parameters())+[v for v in a.parameters() if v.requires_grad],lr=.0007))
    audit.check(step!=1 or same([o.state_dict() for o in opts],load(folder/'initial_optimizer.pt')),'private_fresh_Adam',folder.name)
    if step!=1:
        for o,state in zip(opts,load(folder/'forensic_optimizer_2100.pt')):o.load_state_dict(state)
    banks=replay.projected_banks(agents,bank);record=json.loads((folder/'training.jsonl').read_text().splitlines()[step-1]);weight=.02 if step<=2100 else 0.
    for who,(a,head,opt) in enumerate(zip(agents,heads,opts)):
        z=arrays(folder/f'train_{step:04d}_person{who}.npz');w=private_world(bank,seed,p,who,step,n,triples(p));rec=record['persons'][who]
        for key in PKEYS:audit.check(np.array_equal(w[key],z[key]),'private_first_world_reconstructed',[folder.name,who,key])
        positions=np.concatenate((w['positions'][:,0],w['positions'][:,1]));photos=np.tile(w['photo_ids'],(2,1));goals=torch.tensor(np.tile(w['goals'],2));logits=head(a.observe(replay.scene_visual(positions,photos,banks[who]))).reshape(2*n,2,6)
        audit.check(np.array_equal(logits.detach().numpy(),z['logits']),'private_first_logits_exact',[folder.name,who])
        lp=F.log_softmax(logits,dim=-1);selected=lp[torch.arange(2*n),goals]
        uniform=np.random.default_rng(seed_number(13013,seed,p,who,2,step)).random((2*n,1)).astype(np.float32)
        action=(selected.detach().exp().cumsum(-1)<torch.from_numpy(uniform)).sum(-1).clamp_max(5);reward=torch.tensor((action.numpy()==positions[np.arange(2*n),goals.numpy()]).astype(np.float32))
        audit.check(np.array_equal(uniform,z['action_uniform']) and np.array_equal(action.numpy(),z['action']) and np.array_equal(reward.numpy(),z['reward']),'private_first_sampling_and_reward_exact',[folder.name,who])
        chosen=selected.gather(1,action[:,None]).squeeze(1);entropy=-(selected.exp()*selected).sum(-1).mean();policy=-(chosen*(reward-.5)).mean()
        source=lp[:n];target=lp[n:];inverse=torch.tensor(np.argsort(w['permutation'],axis=1));moved=source.gather(2,inverse[:,None,:].expand(-1,2,-1));middle=torch.logaddexp(moved,target)-np.log(2.)
        jsd=(.5*((moved.exp()*(moved-middle)).sum(-1)+(target.exp()*(target-middle)).sum(-1))).mean()
        loss=policy-weight*entropy+cfg['coefficient']*jsd
        scalar=dict(loss=loss.item(),policy_loss=policy.item(),entropy=entropy.item(),jsd=jsd.item(),mean_reward=reward.mean().item())
        audit.check(all(abs(scalar[k]-rec['components'][k])<2e-7 for k in scalar),'private_first_loss_independent',[folder.name,who,scalar])
        audit.check(abs(jsd.item()-jsd_logits(z['logits'][:n],z['logits'][n:],w['permutation']))<1e-7,'private_JSD_numpy_oracle',[folder.name,who])
        opt.zero_grad(set_to_none=True);loss.backward();params=list(head.parameters())+[v for v in a.parameters() if v.requires_grad];norm=torch.nn.utils.clip_grad_norm_(params,2.)
        audit.check(abs(norm.item()-rec['gradient_norm'])<1e-7,'private_first_gradient_norm',[folder.name,who])
        audit.check(all(v.grad is None for v in a.parameters() if not v.requires_grad),'private_frozen_has_no_gradient',[folder.name,who]);opt.step()
    if step!=1:
        audit.counts['private_late_policy_gradient_replays']+=2;return
    compare_tensors(audit,dict(agents=[a.state_dict() for a in agents],heads=[h.state_dict() for h in heads]),load(folder/'after_first_update.pt'),'private_first_Adam_state_exact',[folder.name])
    compare_tensors(audit,[o.state_dict() for o in opts],load(folder/'after_first_optimizer.pt'),'private_first_Adam_moments_exact',[folder.name])

def social_first(folder,bank,replay,audit,step=1):
    from torch.nn import functional as F
    cfg=read(folder/'config.json');prepared=load(folder.parent/f"prepared_{cfg['seed']}.pt");agents=replay.remake_agents(cfg['seed'],prepared,7,2,'identity')
    for a,state in zip(agents,load(folder/('initial.pt' if step==1 else 'checkpoint_2100.pt'))):
        a.load_state_dict(state)
        for key,param in a.named_parameters():param.requires_grad_(key.startswith(SOCIAL))
    opts=[torch.optim.Adam([v for v in a.parameters() if v.requires_grad],lr=.0007) for a in agents]
    audit.check(step!=1 or same([o.state_dict() for o in opts],load(folder/'initial_optimizer.pt')),'social_fresh_Adam',folder.name)
    if step!=1:
        for o,state in zip(opts,load(folder/'optimizer_2100.pt')):o.load_state_dict(state)
    banks=replay.projected_banks(agents,bank);line=json.loads((folder/'training.jsonl').read_text().splitlines()[step-1]);weight=.02 if step<=2100 else 0.
    terms=replay.replay.__wrapped__(folder/f'train_{step:04d}.npz',agents,agents,banks,cfg,step-1,11,'stochastic',audit)
    for who,(a,opt) in enumerate(zip(agents,opts)):
        parts=[]
        for j,role in enumerate(('sender','receiver')):
            lp,en,value,target=terms[who][role];policy=-(lp*(target-value).detach()).mean();vl=.5*F.mse_loss(value,target);total=policy+vl-weight*en.mean();parts.append(total)
            scalars=dict(policy_loss=policy.item(),value_loss=vl.item(),entropy=en.mean().item(),total=total.item(),target_mean=target.mean().item())
            audit.check(all(abs(v-line['agents'][who]['parts'][j][k])<2e-6 for k,v in scalars.items()),'social_first_role_loss_replayed',[folder.name,who,role])
        loss=torch.stack(parts).mean();opt.zero_grad(set_to_none=True);loss.backward()
        for role,prefix in (('sender',('send_',)),('receiver',SOCIAL[1:])):
            norm=torch.nn.utils.clip_grad_norm_([v for k,v in a.named_parameters() if k.startswith(prefix)],2.)
            audit.check(abs(norm.item()-line['agents'][who]['gradient_norm_by_role'][role])<2e-6,'social_first_role_clip_replayed',[folder.name,who,role])
        audit.check(all(v.grad is None for v in a.parameters() if not v.requires_grad),'social_frozen_has_no_gradient',[folder.name,who])
    for opt in opts:opt.step()
    if step!=1:
        audit.counts['social_late_role_gradient_replays']+=4;return
    compare_tensors(audit,[a.state_dict() for a in agents],load(folder/'after_first_update.pt'),'social_first_Adam_state_exact',[folder.name])
    compare_tensors(audit,[o.state_dict() for o in opts],load(folder/'after_first_optimizer.pt'),'social_first_Adam_moments_exact',[folder.name])


def near_tree(audit,actual,expected,label,context,atol=2e-10):
    if isinstance(expected,dict):
        for k in expected:
            audit.check(k in actual,label+'_field',context+[k])
            if k in actual:near_tree(audit,actual[k],expected[k],label,context+[k],atol)
    elif isinstance(expected,str) or expected is None:audit.check(actual==expected,label,context)
    else:audit.check(np.allclose(actual,expected,atol=atol,rtol=2e-10),label,context)

def optimizer_checkpoint(audit,path,step,context):
    states=load(path)
    for who,state in enumerate(states):
        audit.check(len(state['param_groups'])==1 and state['param_groups'][0]['lr']==.0007,'Adam_fixed_learning_rate',context+[step,who])
        audit.check((len(state['state'])==0) if step==0 else all(float(s['step'])==step for s in state['state'].values()),'Adam_checkpoint_step_count',context+[step,who])
        audit.check(step==0 or len(state['state'])==len(state['param_groups'][0]['params']),'Adam_all_active_parameters_updated',context+[step,who])

def state_digest(states):
    h=hashlib.sha256()
    for i,s in enumerate(states):
        for k,v in sorted(s.items()):h.update(f'{i}/{k}'.encode());h.update(v.numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def private_capability(folder,bank,replay,audit):
    cfg=read(folder/'config.json');curve=read(folder/'curve.json');p=cfg['partition'];seed=cfg['seed'];prepared=load(cfg['prepared_source']['path']);agents=replay.remake_agents(seed,prepared,7,2,'identity');banks=replay.projected_banks(agents,bank)
    photo_pairs=np.array(list(product(*[bank.pools['test',k][:4] for k in (0,1)])));mids=np.repeat(np.arange(30),16);positions=MAPS[mids];photos=np.tile(photo_pairs,(30,1));pools=dict(groups(p),all=np.arange(30));lookup={tuple(v):i for i,v in enumerate(MAPS)}
    for row in curve:
        step=row['update'];state=load(folder/f'checkpoint_{step:04d}.pt');z=arrays(folder/f'evaluation_{step:04d}.npz')
        audit.check(np.array_equal(z['positions'],positions) and np.array_equal(z['map_ids'],mids) and np.array_equal(z['photo_ids'],photos),'private_evaluation_balanced_heldout_photos',[folder.name,step])
        audit.check(state_digest(state['agents'])==row['agent_sha256'],'private_checkpoint_state_fingerprint',[folder.name,step])
        optimizer_checkpoint(audit,folder/f'optimizer_{step:04d}.pt',step,[folder.name])
        for who,a in enumerate(agents):
            a.load_state_dict(state['agents'][who]);head=torch.nn.Sequential(torch.nn.Linear(96,96),torch.nn.Tanh(),torch.nn.Linear(96,12));head.load_state_dict(state['heads'][who])
            actual=head(a.observe(replay.scene_visual(positions,photos,banks[who]))).reshape(480,2,6).numpy();audit.check(np.array_equal(actual,z['logits'][who]),'private_evaluation_logits_exact_replay',[folder.name,step,who]);audit.counts['private_evaluation_worlds_replayed']+=480
            lp=logp(actual);prob=np.exp(lp);correct=actual.argmax(-1)==positions;pcorrect=np.take_along_axis(prob,positions[:,:,None],-1).squeeze(-1);entropy=-(prob*lp).sum(-1);ties=(actual==actual.max(-1,keepdims=True)).sum(-1)>1
            for group,pool in pools.items():
                mask=np.isin(mids,pool);good=correct[mask]
                values=dict(worlds=int(mask.sum()),correct_both=int(good.all(-1).sum()),correct_goals=int(good.sum()),J=float(good.all(-1).mean()),single=float(good.mean()),food=float(good[:,0].mean()),water=float(good[:,1].mean()),stochastic_J=float(pcorrect[mask].prod(-1).mean()),stochastic_single=float(pcorrect[mask].mean()),policy_entropy=float(entropy[mask].mean()),food_entropy=float(entropy[mask,0].mean()),water_entropy=float(entropy[mask,1].mean()),exact_max_tie_goal_count=int(ties[mask].sum()),exact_max_tie_rate=float(ties[mask].mean()),food_tie_rate=float(ties[mask,0].mean()),water_tie_rate=float(ties[mask,1].mean()))
                near_tree(audit,row['scores']['persons'][who]['groups'][group],values,'private_ability_independent',[folder.name,step,who,group])
            # Final all-S6 capability calculation is independent probability-space JSD.
            if step!=cfg['updates']:continue
            policy=prob.reshape(30,16,2,6);actions=actual.argmax(-1).reshape(30,16,2);srows=[]
            for source in range(30):
                target_ids=np.array([lookup[tuple(g[MAPS[source]])] for g in PERMS]);target=policy[target_ids]
                moved=np.take_along_axis(np.broadcast_to(policy[source],(720,16,2,6)),np.argsort(PERMS)[:,None,None,:],axis=-1)
                distance=jsd_probability(moved,target);consistent=actions[target_ids]==PERMS[:,actions[source]]
                record=dict(source_map=source,jsd=float(distance.mean()),food_jsd=float(distance[:,:,0].mean()),water_jsd=float(distance[:,:,1].mean()),greedy_single_consistency=float(consistent.mean()),greedy_both_consistency=float(consistent.all(-1).mean()));srows.append(record)
                near_tree(audit,row['scores']['persons'][who]['equivariance']['per_source_map'][source],record,'private_full720_independent',[folder.name,who,source])
            for group,pool in pools.items():
                values={k:float(np.mean([srows[i][k] for i in pool])) for k in srows[0] if k!='source_map'}
                near_tree(audit,row['scores']['persons'][who]['equivariance']['groups'][group],values,'private_full720_group_by_source',[folder.name,who,group])
            audit.counts['private_S6_goal_comparisons']+=30*720*16*2

def import_metric_reference():
    import importlib.util
    path=PROJECT/'redesign_v0.12/audit_receiver.py';spec=importlib.util.spec_from_file_location('v13_prior_independent_metrics',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

@torch.no_grad()
def social_protocol(folder,bank,replay,audit):
    cfg=read(folder/'config.json');seed=cfg['seed'];p=cfg['partition'];prepared=load(folder.parent/f'prepared_{seed}.pt');agents=replay.remake_agents(seed,prepared,7,2,'identity');banks=replay.projected_banks(agents,bank);curve=read(folder/'curve.json');reference=import_metric_reference()
    photos=np.tile(np.array(list(product(*[bank.pools['test',k][:4] for k in (0,1)]))),(30,1));mids=np.repeat(np.arange(30),16);positions=MAPS[mids];pools=dict(groups(p),common30=np.arange(30));messages=torch.tensor(list(product(range(7),repeat=2)))
    for row in curve:
        step=row['update'];states=load(folder/f'checkpoint_{step:04d}.pt');saved=read(folder/f'protocol_{step:04d}.json')
        audit.check(state_digest(states)==row['state_sha256'],'social_checkpoint_state_fingerprint',[folder.name,step]);optimizer_checkpoint(audit,folder/f'optimizer_{step:04d}.pt',step,[folder.name])
        for a,state in zip(agents,states):a.load_state_dict(state)
        for d in (0,1):
            raw=arrays(folder/f'protocol_{step:04d}_d{d}.npz');s=agents[d];receiver=agents[1-d]
            audit.check(np.array_equal(raw['positions'],positions) and np.array_equal(raw['map_ids'],mids) and np.array_equal(raw['photo_ids'],photos),'protocol_balanced_fixed_16photo_worlds',[folder.name,step,d])
            h=s.observe(replay.scene_visual(positions,photos,banks[d]));state=s.send_context(torch.cat((h,torch.zeros(480,4)),1));first=s.send_out(state);second=s.send_out(s.send_recur(s.send_embedding(torch.arange(7).repeat(480)),state.repeat_interleave(7,0))).reshape(480,7,7)
            for k,v in [('h',h),('first_logits',first),('second_logits',second)]:audit.check(np.array_equal(v.numpy(),raw[k]),'protocol_sender_native_logits_exact',[folder.name,step,d,k])
            f=first.argmax(-1).numpy();second_choice=second.numpy()[np.arange(480),f].argmax(-1);greedy=np.column_stack((f,second_choice));audit.check(np.array_equal(greedy,raw['greedy_message']),'protocol_sequential_greedy_distinct_from_joint_MAP',[folder.name,step,d])
            rl=[]
            for goal in (0,1):
                need=torch.zeros(49,2);need[:,goal]=1;out,_=receiver.receive(messages,need,torch.zeros(49,2),torch.zeros(49,18),torch.arange(6).repeat(49,1));rl.append(out.numpy())
            rlogits=np.stack(rl,1);audit.check(np.array_equal(rlogits,raw['receiver_logits']),'protocol_receiver_all49_logits_exact',[folder.name,step,d]);native=(np.exp(logp(raw['first_logits']))[:,:,None]*np.exp(logp(raw['second_logits']))).reshape(480,49)
            for mode,probs in [('native',native),('greedy',np.eye(49)[greedy@np.array([7,1])])]:
                for group,pool in pools.items():
                    mask=np.isin(mids,pool);c=np.zeros((49,6,6))
                    for world in np.flatnonzero(mask):c[:,positions[world,0],positions[world,1]]+=probs[world]/int(mask.sum())
                    values=reference.independent_metrics(c,rlogits);values.update(world_count=int(mask.sum()),defined=True)
                    near_tree(audit,saved[d]['metrics'][mode][group],values,'protocol_independent_probability_metrics',[folder.name,step,d,mode,group]);audit.counts['protocol_probability_tables']+=1
            audit.counts['protocol_sender_worlds_replayed']+=480


def pair_stats(reward,successes,goals):
    n=len(reward)
    if not n:return dict(n=0,decisions=0,reward_sum=0.,mean_reward=None,reward_variance=None,positive_rewards=0,single_correct=0,single_accuracy=None,both_correct=0,both_accuracy=None,food_correct=0,water_correct=0,outcome_counts={k:0 for k in ('00','01','10','11')})
    s=successes.astype(np.float64);by_resource=np.take_along_axis(s,np.argsort(goals,axis=1),axis=1);codes=2*by_resource[:,0]+by_resource[:,1];reward=reward.astype(np.float64)
    return dict(n=n,decisions=2*n,reward_sum=float(reward.sum()),mean_reward=float(reward.mean()),reward_variance=float(reward.var()),positive_rewards=int((reward>0).sum()),single_correct=int(s.sum()),single_accuracy=float(s.mean()),both_correct=int(s.prod(1).sum()),both_accuracy=float(s.prod(1).mean()),food_correct=int(by_resource[:,0].sum()),water_correct=int(by_resource[:,1].sum()),outcome_counts={k:int((codes==i).sum()) for i,k in enumerate(('00','01','10','11'))})

def final_statistics(folder,mode,audit):
    cfg=read(folder/'config.json');saved=read(folder/'result.json')['scores'][mode];z=arrays(folder/f'final_{mode}.npz');success=(z['place']==np.take_along_axis(z['positions'],z['goals'],1)).astype(np.float32);reward=.25*success.sum(1)+.5*success.prod(1)
    near_tree(audit,saved,pair_stats(reward,success,z['goals']),'final_statistics_independent',[folder.name,mode])
    mids=np.array([{tuple(v):i for i,v in enumerate(MAPS)}[tuple(x)] for x in z['positions']])
    for group,pool in groups(cfg['partition']).items():
        mask=np.isin(mids,pool);near_tree(audit,saved['map_groups'][group],pair_stats(reward[mask],success[mask],z['goals'][mask]),'final_group_statistics_independent',[folder.name,mode,group])
        for d in (0,1):
            mask=np.isin(mids,pool)&(z['scout']==d);near_tree(audit,saved['direction_groups'][d][group],pair_stats(reward[mask],success[mask],z['goals'][mask]),'final_direction_statistics_independent',[folder.name,mode,d,group])


def audit_batch(batch,audit,preflight):
    inv=read(batch/'invocation.json');cfg=inv['args'];hashes=inv['source_hashes'];complete=read(batch/'training_complete.json');audit.check(complete['status']=='complete','batch_complete')
    for i,(path,digest) in enumerate(hashes.items()):
        audit.check(sha(path)==digest,'source_current_equals_executed',path)
        archive=batch/'frozen_sources'/f'{i:02d}_{Path(path).name}'
        if archive.exists():audit.check(sha(archive)==digest,'archived_source_equals_executed',path)
        else:audit.check(Path(path).name=='features.npz','only_features_not_duplicated_in_source_archive',path)
    audit.check(complete['source_hashes']==hashes,'completion_source_manifest_unchanged')
    replay=import_replay();bank=replay.ImageBank();torch.set_num_threads(1)
    for seed,p in product(cfg['seeds'],cfg['partitions']):
        pair=[];spair=[];table=triples(p);prepared=load(batch/f'prepared_{seed}.pt');fresh=replay.remake_agents(seed,prepared,7,2,'identity')
        for arm in ARMS:
            folder=batch/f'private_s{seed}_p{p}_{arm}';c=read(folder/'config.json');initial=load(folder/'initial.pt');final=load(folder/'final.pt');transfer=load(folder/'transferred.pt');pair.append((initial,[]))
            audit.check(c['source_hashes']==hashes and sha(c['prepared_source']['path'])==c['prepared_source']['sha256'],'private_sources_bound',folder.name)
            audit.check(same(initial['agents'],[a.state_dict() for a in fresh]),'private_fresh_common_Camp_initialization',folder.name)
            z=arrays(folder/'legal_triples.npz');audit.check(np.array_equal(z['triples'],table) and np.array_equal(z['permutations'],PERMS),'saved_legal_triples_independently_enumerated',folder.name)
            audit.check(same(transfer,final['agents']),'only_final_Camp_and_no_private_head_transferred',folder.name)
            for checkpoint in c['checkpoints']:
                state=load(folder/f'checkpoint_{checkpoint:04d}.pt')
                for who in (0,1):
                    for k,v in state['agents'][who].items():
                        if not k.startswith(PRIVATE):audit.check(torch.equal(v,initial['agents'][who][k]),'private_unused_and_project_frozen_every_checkpoint',[folder.name,checkpoint,who,k])
                audit.counts['private_checkpoints']+=1
            lines=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()];audit.check(len(lines)==c['updates'],'private_budget_logged',folder.name)
            for step,line in enumerate(lines,1):
                audit.check(line['update']==step,'private_update_number',[folder.name,step]);pair[-1][1].append([])
                for who,person in enumerate(line['persons']):
                    w=private_world(bank,seed,p,who,step,c['batch_pairs_per_person'],table);dig=named_arrays_sha(w,PKEYS)
                    audit.check(person['world_sha256']==dig,'private_full_world_hash_reconstructed',[folder.name,step,who]);pair[-1][1][-1].append(dig)
                    audit.check(person['world_seed']==seed_number(13013,seed,p,who,1,step) and person['action_seed']==seed_number(13013,seed,p,who,2,step),'private_seed_identity',[folder.name,step,who])
                    audit.check(person['map_exposure']==dict(old=2*c['batch_pairs_per_person'],added=0,sealed=0),'private_old_only_exposure',[folder.name,step,who]);audit.counts['private_person_updates']+=1
            private_first(folder,bank,replay,audit)
            if c['updates']>=2101:private_first(folder,bank,replay,audit,2101)
            private_capability(folder,bank,replay,audit)
            social=batch/f'social_s{seed}_p{p}_{arm}';sc=read(social/'config.json');si=load(social/'initial.pt');spair.append((si,[]))
            audit.check(same(si,transfer) and sha(sc['source_checkpoint']['path'])==sc['source_checkpoint']['sha256'],'social_initial_equals_private_transfer',social.name)
            audit.check(sc['source_hashes']==hashes and sc['training_pool']==groups(p)['old'].tolist(),'social_sources_and_old_training_pool',social.name)
            for checkpoint in sc['checkpoints']:
                state=load(social/f'checkpoint_{checkpoint:04d}.pt')
                for who in (0,1):
                    for k,v in state[who].items():
                        if not k.startswith(SOCIAL):audit.check(torch.equal(v,si[who][k]),'social_visual_and_transform_frozen_every_checkpoint',[social.name,checkpoint,who,k])
                audit.counts['social_checkpoints']+=1
            lines=[json.loads(x) for x in (social/'training.jsonl').read_text().splitlines()];audit.check(len(lines)==sc['updates'],'social_budget_logged',social.name)
            for step,line in enumerate(lines):
                worlds=social_worlds(bank,seed,p,step,sc['batch'],True);digest=world_digest(worlds);spair[-1][1].append(digest)
                audit.check(line['stats']['world_sha256']==digest,'social_full_world_hash_reconstructed',[social.name,step])
                audit.check(line['stats']['map_groups']['old']['n']==sc['batch'] and line['stats']['map_groups']['added']['n']==line['stats']['map_groups']['sealed']['n']==0,'social_old_only_exposure',[social.name,step]);audit.counts['social_updates']+=1
            social_first(social,bank,replay,audit)
            if sc['updates']>=2101:social_first(social,bank,replay,audit,2101)
            social_protocol(social,bank,replay,audit)
            agents=replay.remake_agents(seed,prepared,7,2,'identity')
            for a,state in zip(agents,load(social/'final.pt')):a.load_state_dict(state);a.requires_grad_(False)
            banks=replay.projected_banks(agents,bank)
            for mode in ('normal','shuffle','blank','stochastic','erase_memory'):
                path=social/f'final_{mode}.npz';replay.replay(path,agents,agents,banks,sc,0,91,mode,audit);z=arrays(path);ew=social_worlds(bank,seed,p,0,sc['eval_n'],False)
                for d in (0,1):
                    ix=z['scout']==d
                    for k in ew[d]:audit.check(np.array_equal(z[k][ix],ew[d][k]),'final_worlds_balanced_reconstructed',[social.name,mode,d,k])
                final_statistics(social,mode,audit);audit.counts['final_modes']+=1
            audit.counts['private_runs']+=1;audit.counts['social_runs']+=1
        audit.check(same(pair[0][0],pair[1][0]),'arms_private_full_initial_identical',[seed,p]);audit.check(pair[0][1]==pair[1][1],'arms_private_all_worlds_identical',[seed,p]);audit.check(spair[0][1]==spair[1][1],'arms_social_all_worlds_identical',[seed,p])
        for who in (0,1):
            for k in spair[0][0][who]:
                if not k.startswith(PRIVATE):audit.check(torch.equal(spair[0][0][who][k],spair[1][0][who][k]),'arms_social_language_initial_identical',[seed,p,who,k])
    return inv

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT/'results/spatial_001');parser.add_argument('--preflight',action='store_true');args=parser.parse_args();batch=args.root.resolve();audit=Audit();mathematical_tests(audit);inv=audit_batch(batch,audit,args.preflight)
    rng_path=ROOT/'rng_preflight.json';rng=read(rng_path) if rng_path.exists() else None
    audit.check(rng is not None and rng['passed'] and rng['source_hashes']==inv['source_hashes'],'root_independent_RNG_identity_audit_available')
    result=dict(passed=not audit.failures,created_utc=datetime.now(timezone.utc).isoformat(),batch=str(batch),source_hashes=inv['source_hashes'],checks=dict(audit.checks),check_count=sum(audit.checks.values()),counts=dict(audit.counts),failures=audit.failures,script_sha256=sha(__file__),rng_audit=dict(path=str(rng_path),sha256=sha(rng_path)) if rng else None,scope='Independent legal-triple/JSD math; reconstructed all training worlds; frozen states and exact initialization/transfer; first private and social policy/loss/gradient/Adam replay; all final modes policy/world replay. RNG namespace enumeration separately conducted by root.',development_revision='smoke001 retained: private and social RNG namespace13013 collided; smoke002 executed socialworld namespace13014; no formal training had started before fix.')
    path=(ROOT/'preflight_qa.json') if args.preflight else batch/'audit_execution.json'
    if path.exists():path.rename(path.with_name(path.stem+'_previous_'+datetime.now().strftime('%Y%m%d%H%M%S')+path.suffix))
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(path=str(path),passed=result['passed'],checks=result['check_count'],counts=result['counts'],failures=result['failures'][:20]),ensure_ascii=False,indent=2));assert result['passed']

if __name__=='__main__':main()
