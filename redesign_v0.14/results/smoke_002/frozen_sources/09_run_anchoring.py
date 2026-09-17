"""Historical-partner anchoring and fixed-time release with matched role queries."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'redesign_v0.10'))
import run_generalization as v10
v9=v10.v9;v8=v10.v8
from camp import ImageBank,remake_agents,projected_banks,scene_visual,draw,write_json

SEEDS=[29101,29102,29103,29104]
ARMS=['current','anchor','release']
SOURCE=ROOT.parent/'redesign_v0.10/results/generalization_001'
PLAN=dict(v10.PLAN)
TIMES=[0,25,50,100,200,300,325,350,400,500,600]


def seed_for(seed,p,purpose,step=0,who=0):
    return int(np.random.SeedSequence([11011,seed,p,purpose,step,who]).generate_state(1,dtype=np.uint64)[0]>>np.uint64(1))


def sources():return [ROOT/'run_anchoring.py',ROOT/'固定执行方案.md']+v10.sources()


def worlds(bank,seed,p,step,pool,n,purpose,training=True):
    plan=dict(PLAN,map_pool=list(pool),allowed_sites=list(range(6)))
    return [v8.fixture(bank,np.random.default_rng(seed_for(seed,p,purpose,step,d)),n//2,plan,training,
        'train' if training else 'test') for d in (0,1)]


def hash_worlds(items):
    h=hashlib.sha256()
    for w in items:
        for key in v8.WORLD_KEYS:h.update(np.ascontiguousarray(w[key]).tobytes())
    return h.hexdigest()


def flatten(rows):return {k:np.concatenate([row[k] for row in rows]) for k in rows[0]}


def interact(senders,receivers,banks,items,seed,p,step,stream,*,sender_grad=False,receiver_grad=False,mode='normal'):
    learning=[{},{}];records=[];greedy=mode!='stochastic';world_hash=hash_worlds(items)
    for scout,w in enumerate(items):
        collector=1-scout;n=len(w['positions']);inv=torch.from_numpy(w['inventory'].astype(np.float32))
        sr=np.random.default_rng(seed_for(seed,p,stream,step,scout))
        rr=np.random.default_rng(seed_for(seed,p,stream+1,step,scout))
        with torch.set_grad_enabled(sender_grad):
            h=senders[scout].observe(scene_visual(w['positions'],w['photo_ids'],banks[scout]),
                memory_mode='reset' if mode=='erase_memory' else 'retain')
            sent,sl,se,sv=senders[scout].send(h,torch.zeros(n,2),inv,sr,greedy)
        assert sl.requires_grad==sender_grad
        delivered=sent.detach().numpy().copy()
        if mode=='blank':delivered[:]=0
        elif mode=='shuffle':
            ir=np.random.default_rng(seed_for(seed,p,stream+2,step,scout))
            for first in (0,1):
                ix=np.flatnonzero(w['goals'][:,0]==first)
                delivered[ix]=sent.detach().numpy()[ir.permutation(ix)]
        acts=[];places=[];lps=[];ents=[];values=[]
        with torch.set_grad_enabled(receiver_grad):
            for q in (0,1):
                goal=torch.from_numpy(np.eye(2,dtype=np.float32)[w['goals'][:,q]])
                logits,value=receivers[collector].receive(torch.from_numpy(delivered),goal,inv,
                    torch.from_numpy(w['history']),torch.from_numpy(w['menu'][:,q]))
                a,lp,en=draw(logits,rr,greedy);assert lp.requires_grad==receiver_grad
                acts.append(a.detach().numpy());places.append(w['menu'][np.arange(n),q,a.detach().numpy()])
                lps.append(lp);ents.append(en);values.append(value)
        action=np.column_stack(acts);place=np.column_stack(places)
        correct=(place==np.take_along_axis(w['positions'],w['goals'],axis=1)).astype(np.float32)
        reward=v8.utility(correct,.5);target=torch.from_numpy(reward-1)
        if sender_grad:learning[scout]['sender']=(sl,se,sv,target)
        if receiver_grad:learning[collector]['receiver']=(sum(lps),sum(ents),torch.stack(values).mean(0),target)
        records.append(dict(scout=np.full(n,scout),episode=np.arange(n),**w,sent=sent.detach().numpy(),
            delivered=delivered,action=action,place=place,successes=correct,reward=reward))
    arr=flatten(records)
    stats=v8.pair_metrics(arr['reward'],arr['successes'],arr['goals'])
    stats.update(world_sha256=world_hash,trace_sha256=hashlib.sha256(b''.join(np.ascontiguousarray(arr[k]).tobytes()
        for k in ('sent','delivered','action','place','successes','reward'))).hexdigest())
    return v10.grouped(stats,arr,p),learning,arr


def is_anchored(arm,step,release_after):return arm=='anchor' or (arm=='release' and step<release_after)


def training_queries(agents,old,banks,oldbanks,bank,seed,p,step,arm,release_after,contexts):
    group=v10.partition(p)
    ow=worlds(bank,seed,p,step,group['old'],contexts//2,1)
    nw=worlds(bank,seed,p,step,group['added'],contexts//2,2)
    anchored=is_anchored(arm,step,release_after)
    out={}
    out['old_s']=interact(agents,old if anchored else agents,banks,ow,seed,p,step,11,
        sender_grad=True,mode='stochastic')
    out['old_r']=interact(old if anchored else agents,agents,oldbanks if anchored else banks,ow,seed,p,step,21,
        receiver_grad=True,mode='stochastic')
    out['new']=interact(agents,agents,banks,nw,seed,p,step,31,sender_grad=True,receiver_grad=True,mode='stochastic')
    assert out['old_s'][0]['world_sha256']==out['old_r'][0]['world_sha256']
    assert all(value[0]['map_groups']['sealed']['n']==0 for value in out.values())
    return out,anchored


def update_agents(agents,optimizers,queries,weight):
    info=[]
    # No optimizer is stepped until all current agents' losses and gradients exist.
    for who,(a,opt) in enumerate(zip(agents,optimizers)):
        parts=[];losses=[]
        for query,role in (('old_s','sender'),('old_r','receiver'),('new','sender'),('new','receiver')):
            lp,en,value,target=queries[query][1][who][role]
            policy=-(lp*(target-value).detach()).mean();vl=.5*F.mse_loss(value,target)
            part=policy+vl-weight*en.mean();losses.append(part)
            parts.append(dict(query=query,role=role,n=len(target),policy_loss=float(policy.detach()),
                value_loss=float(vl.detach()),entropy=float(en.mean().detach()),total=float(part.detach()),
                target_mean=float(target.mean())))
        loss=torch.stack(losses).mean();assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();norms={}
        for role,prefixes in [('sender',v9.SENDER),('receiver',v9.RECEIVER)]:
            parameters=[param for key,param in a.named_parameters() if key.startswith(prefixes)]
            assert parameters and all(param.requires_grad and param.grad is not None for param in parameters)
            norms[role]=float(torch.nn.utils.clip_grad_norm_(parameters,2.))
        assert all(np.isfinite(x) for x in norms.values())
        assert all(param.grad is None for param in a.parameters() if not param.requires_grad)
        info.append(dict(loss=float(loss.detach()),parts=parts,gradient_norm_by_role=norms))
    for opt in optimizers:opt.step()
    return info


def evaluate(agents,old,banks,oldbanks,bank,seed,p,n,out=None,final=False):
    w=worlds(bank,seed,p,0,range(30),n,90,False);scores={}
    variants={'current':(agents,agents,banks),'current_to_old':(agents,old,banks),'old_to_current':(old,agents,oldbanks)}
    for name,(s,r,b) in variants.items():
        modes=('normal','shuffle','blank','stochastic','erase_memory') if final and name=='current' else ('normal',)
        scores[name]={}
        for mode in modes:
            stats,_,arrays=interact(s,r,b,w,seed,p,0,91,mode=mode)
            scores[name][mode]=stats
            if out is not None:np.savez_compressed(out/f'final_{name}_{mode}.npz',**arrays)
    return scores


def load_source(seed,p,source):
    folder=source/f's{seed}_p{p}_base';prepared=torch.load(source/f'prepared_{seed}.pt',weights_only=True)
    agents=remake_agents(seed,prepared,7,2,'identity')
    for a,state in zip(agents,torch.load(folder/'final.pt',weights_only=True)):a.load_state_dict(state)
    assert v8.state_sha(agents)==json.loads((folder/'result.json').read_text())['final_sha256']
    old=copy.deepcopy(agents)
    for a in old:a.requires_grad_(False)
    assert len({p.data_ptr() for a in agents+old for p in a.parameters()})==sum(len(list(a.parameters())) for a in agents+old)
    partition=v9.select_parameters(agents,'both')
    return agents,old,partition


def fit(seed,p,arm,bank,args):
    dest=args.out/f's{seed}_p{p}_{arm}';dest.mkdir(parents=True,exist_ok=False)
    agents,old,selected=load_source(seed,p,args.source_root)
    frozen=v9.state_subset(agents);initial=v8.state_sha(agents);assert initial==v8.state_sha(old)
    banks=projected_banks(agents,bank);oldbanks=projected_banks(old,bank)
    assert all(torch.equal(x,y) for x,y in zip(banks,oldbanks))
    checkpoints=sorted(set([0,args.updates,args.release_after,args.entropy_off_after]+[x for x in TIMES if x<args.updates]))
    checkpoints=[x for x in checkpoints if x<=args.updates]
    source_dir=args.source_root/f's{seed}_p{p}_base'
    cfg=dict(seed=seed,partition=p,arm=arm,updates=args.updates,contexts_per_update=args.contexts,
        communications_per_update=args.contexts*3//2,actions_per_update=args.contexts*3,eval_n=args.eval_n,
        release_after=args.release_after,entropy_off_after=args.entropy_off_after,entropy_coefficient=.02,
        learning_rate=.0007,optimizer='fresh Adam',checkpoints=checkpoints,plan=PLAN,
        role_loss_weights={'old_s_sender':.25,'old_r_receiver':.25,'new_sender':.25,'new_receiver':.25},
        map_groups={k:v.tolist() for k,v in v10.partition(p).items()},parameter_partition=selected,
        initial_sha256=initial,reference_state_sha256=initial,
        source_checkpoint={'path':str(source_dir/'final.pt'),'sha256':v8.sha(source_dir/'final.pt')},
        source_config={'path':str(source_dir/'config.json'),'sha256':v8.sha(source_dir/'config.json')},
        prepared_source={'path':str(args.source_root/f'prepared_{seed}.pt'),'sha256':v8.sha(args.source_root/f'prepared_{seed}.pt')},
        source_hashes={str(f):v8.sha(f) for f in sources()},rng_namespace=11011,
        inherited_v10_sources=True,old_photos_development=True,sealed_never_trained=True)
    write_json(dest/'config.json',cfg);torch.save([a.state_dict() for a in agents],dest/'initial.pt')
    torch.save([a.state_dict() for a in old],dest/'reference.pt')
    opts=[torch.optim.Adam([param for param in a.parameters() if param.requires_grad],lr=.0007) for a in agents]
    torch.save([o.state_dict() for o in opts],dest/'initial_optimizer.pt')
    curve=[];started=time.monotonic()
    with (dest/'training.jsonl').open('w') as log:
        for step in range(args.updates+1):
            if step in checkpoints:
                assert v9.subset_verified(agents,frozen) and v8.state_sha(old)==initial
                scores=evaluate(agents,old,banks,oldbanks,bank,seed,p,args.eval_n)
                curve.append(dict(update=step,scores=scores,current_sha256=v8.state_sha(agents),reference_state_sha256=initial))
                write_json(dest/'curve.json',curve)
                torch.save([a.state_dict() for a in agents],dest/f'checkpoint_{step:04d}.pt')
                torch.save([o.state_dict() for o in opts],dest/f'optimizer_{step:04d}.pt')
                print(json.dumps(dict(run=dest.name,update=step,
                    old=scores['current']['normal']['map_groups']['old']['both_accuracy'],
                    added=scores['current']['normal']['map_groups']['added']['both_accuracy'],
                    current_to_old=scores['current_to_old']['normal']['map_groups']['old']['both_accuracy'],
                    old_to_current=scores['old_to_current']['normal']['map_groups']['old']['both_accuracy'])),flush=True)
            if step==args.updates:break
            queries,anchored=training_queries(agents,old,banks,oldbanks,bank,seed,p,step,arm,args.release_after,args.contexts)
            for tag,(_,learning,_) in queries.items():
                expected={'sender'} if tag=='old_s' else {'receiver'} if tag=='old_r' else {'sender','receiver'}
                assert all(set(row)==expected for row in learning)
            if step in (0,args.release_after,args.entropy_off_after):
                for tag,(_,_,a) in queries.items():np.savez_compressed(dest/f'train_{step+1:04d}_{tag}.npz',**a)
            weight=.02 if step<args.entropy_off_after else 0.
            info=update_agents(agents,opts,queries,weight)
            assert all(param.grad is None for a in old for param in a.parameters())
            if step==0:
                torch.save([a.state_dict() for a in agents],dest/'after_first_update.pt')
                torch.save([o.state_dict() for o in opts],dest/'after_first_optimizer.pt')
            stats={tag:r[0] for tag,r in queries.items()}
            log.write(json.dumps(dict(update=step+1,anchored=anchored,entropy_weight=weight,queries=stats,agents=info,
                raw_communication_mean_reward=sum(s['reward_sum'] for s in stats.values())/(args.contexts*3//2),
                role_weighted_mean_reward=.25*stats['old_s']['mean_reward']+.25*stats['old_r']['mean_reward']+.5*stats['new']['mean_reward']))+'\n')
            if (step+1)%100==0:log.flush()
    final=evaluate(agents,old,banks,oldbanks,bank,seed,p,args.eval_n,dest,True)
    assert all(final[name]['normal']==curve[-1]['scores'][name]['normal'] for name in final)
    assert v9.subset_verified(agents,frozen) and v8.state_sha(old)==initial
    torch.save([a.state_dict() for a in agents],dest/'final.pt');torch.save([o.state_dict() for o in opts],dest/'final_optimizer.pt')
    write_json(dest/'result.json',dict(seed=seed,partition=p,arm=arm,updates=args.updates,scores=final,
        initial_sha256=initial,final_sha256=v8.state_sha(agents),reference_state_sha256=v8.state_sha(old),
        seconds=time.monotonic()-started,frozen_modules_verified=True,sealed_never_trained=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'results/anchoring_001')
    parser.add_argument('--source-root',type=Path,default=SOURCE);parser.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    parser.add_argument('--partitions',type=int,nargs='+',default=[1,2,3]);parser.add_argument('--updates',type=int,default=600)
    parser.add_argument('--release-after',type=int,default=300);parser.add_argument('--entropy-off-after',type=int,default=500)
    parser.add_argument('--contexts',type=int,default=512);parser.add_argument('--eval-n',type=int,default=9600)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args();torch.set_num_threads(1)
    args.out=args.out.resolve();args.source_root=args.source_root.resolve();formal=not args.smoke
    assert args.contexts%4==0 and args.eval_n%120==0 and 0<args.release_after<args.entropy_off_after<args.updates
    if formal:
        assert (args.seeds,args.partitions,args.updates,args.release_after,args.entropy_off_after,args.contexts,args.eval_n)==(SEEDS,[1,2,3],600,300,500,512,9600)
        assert args.source_root==SOURCE.resolve()
        qa=json.loads((ROOT/'preflight_qa.json').read_text());assert qa['passed'] and qa['runner_sha256']==v8.sha(__file__)
        oldqa=json.loads((SOURCE/'audit_execution.json').read_text());assert oldqa['passed'] and oldqa['completed_social_runs']==60
        for path,digest in oldqa['source_hashes'].items():assert v8.sha(path)==digest
    purpose_list=(1,2,11,12,21,22,31,32)
    ids=[seed_for(s,p,k,u,d) for s,p,k,u,d in product(args.seeds,args.partitions,purpose_list,range(args.updates),(0,1))]
    ids += [seed_for(s,p,k,0,d) for s,p,k,d in product(args.seeds,args.partitions,(90,91,92,93),(0,1))]
    assert len(ids)==len(set(ids))
    args.out.mkdir(parents=True,exist_ok=False)
    snapshot=args.out/'frozen_sources';snapshot.mkdir()
    for i,path in enumerate(sources()):
        if path.suffix in ('.py','.md'):shutil.copy2(path,snapshot/f'{i:02d}_{path.name}')
    if formal:shutil.copy2(ROOT/'preflight_qa.json',snapshot/'preflight_qa.json')
    write_json(args.out/'invocation.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),formal=formal,
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},arms=ARMS,
        rng_identity_count=len(ids),python=platform.python_version(),torch=str(torch.__version__),device='cpu',
        source_hashes={str(p):v8.sha(p) for p in sources()}))
    bank=ImageBank()
    for seed,p,arm in product(args.seeds,args.partitions,ARMS):fit(seed,p,arm,bank,args)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
