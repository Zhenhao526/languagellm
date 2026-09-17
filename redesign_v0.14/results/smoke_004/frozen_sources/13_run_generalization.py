"""Balanced 18 -> 24 social support expansion, with six never-trained maps."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
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
sys.path.insert(0,str(ROOT.parent/'redesign_v0.9'))
import run_adaptation as v9
v8=v9.v8
import run_controls as controls8
from camp import MAPS,ImageBank,remake_agents,projected_banks,split_maps,write_json

SEEDS=[29101,29102,29103,29104]
ARMS=['expand_both','stay_old_both','expand_sender','expand_receiver']
PLAN=dict(v8.CONDITIONS['full_mixed'])


def partition(p):
    assert p in (1,2,3)
    added=split_maps(p)[1]
    sealed=split_maps(p%3+1)[1]
    old=np.setdiff1d(np.arange(30),np.union1d(added,sealed))
    assert len(old)==18 and len(added)==len(sealed)==6 and not set(added)&set(sealed)
    for pool in (old,added,sealed):
        assert all(np.array_equal(np.bincount(MAPS[pool,k],minlength=6),np.full(6,len(pool)//6)) for k in (0,1))
    return {'old':old,'added':added,'sealed':sealed}


def seed_for(seed,p,purpose,step=0):
    return int(np.random.SeedSequence([10010,seed,p,purpose,step]).generate_state(1,dtype=np.uint64)[0] >> np.uint64(1))


def sources():
    return [ROOT/'run_generalization.py',ROOT/'固定执行方案.md',Path(v9.__file__),
        ROOT.parent/'redesign_v0.8/camp.py',ROOT.parent/'redesign_v0.8/run_experiment.py',
        ROOT.parent/'redesign_v0.8/run_controls.py',ROOT.parent/'redesign_v0.8/固定执行方案.md',
        ROOT.parent/'redesign_v0.4/agents.py',ROOT.parent/'redesign_v0.4/run_pilot.py',
        ROOT.parent/'redesign_v0.4/resource_env.py',ROOT.parent/'redesign_v0.4/data/manifest.json',
        ROOT.parent/'redesign_v0.4/data/features.npz']


def grouped(stats,arrays,p):
    mids=arrays['positions'][:,0]*5+arrays['positions'][:,1]-(arrays['positions'][:,1]>arrays['positions'][:,0])
    groups=partition(p)
    def score(mask):return v8.pair_metrics(arrays['reward'][mask],arrays['successes'][mask],arrays['goals'][mask])
    stats['map_groups']={name:score(np.isin(mids,pool)) for name,pool in groups.items()}
    stats['direction_groups']=[{name:score(np.isin(mids,pool)&(arrays['scout']==who)) for name,pool in groups.items()} for who in (0,1)]
    stats['sealed_maps_never_in_this_social_training']=True
    return stats


@torch.no_grad()
def evaluate(agents,banks,bank,p,seed,n,out=None,modes=('normal',)):
    result={}
    for mode in modes:
        stats,_,rows=v8.rollout(agents,banks,bank,PLAN,seed,n,greedy=mode!='stochastic',mode=mode,trace=True,split='test')
        arrays={k:np.concatenate([row[k] for row in rows]) for k in rows[0]}
        result[mode]=grouped(stats,arrays,p)
        if out is not None:np.savez_compressed(out/f'final_{mode}.npz',**arrays)
    return result


def fit(seed,p,arm,prepared,bank,out,updates,batch,eval_n,source=None):
    # base establishes the common convention; every adaptation clones it exactly.
    assert arm=='base' or arm in ARMS
    dest=out/f's{seed}_p{p}_{arm}'
    hashes={str(f):v8.sha(f) for f in sources()}
    if (dest/'result.json').exists():
        cfg=json.loads((dest/'config.json').read_text())
        assert cfg['source_hashes']==hashes and (cfg['updates'],cfg['batch'],cfg['eval_n'])==(updates,batch,eval_n)
        if source:assert cfg['source_checkpoint']['sha256']==v8.sha(source)
        print(f'SKIP {dest.name}',flush=True);return
    dest.mkdir(parents=True,exist_ok=False)
    agents=remake_agents(seed,prepared,7,2,'identity')
    if source:
        for a,state in zip(agents,torch.load(source,weights_only=True)):a.load_state_dict(state)
        assert v8.state_sha(agents)==json.loads((source.parent/'result.json').read_text())['final_sha256']
    group=partition(p)
    pool=group['old'] if arm in ('base','stay_old_both') else np.sort(np.r_[group['old'],group['added']])
    assert not set(pool)&set(group['sealed'])
    plasticity={'base':'both','expand_both':'both','stay_old_both':'both','expand_sender':'sender_only','expand_receiver':'receiver_only'}[arm]
    selected=v9.select_parameters(agents,plasticity)
    frozen=v9.state_subset(agents)
    initial_hash=v8.state_sha(agents)
    banks=projected_banks(agents,bank)
    boundary=updates-min(300 if arm=='base' else 100,max(1,updates//(8 if arm=='base' else 6)))
    proposed=(0,100,300,600,1200,1800,2100,2400) if arm=='base' else (0,25,50,100,200,400,500,600)
    checkpoints=sorted(set([0,updates]+[x for x in proposed if x<updates]))
    purpose=1 if arm=='base' else 2
    evaluation_seed=seed_for(seed,p,3)
    train_plan=dict(PLAN,map_pool=pool.tolist(),allowed_sites=list(range(6)))
    cfg=dict(seed=seed,partition=p,arm=arm,plasticity=plasticity,plan=PLAN,
        map_groups={name:ids.tolist() for name,ids in group.items()},training_pool=pool.tolist(),training_plan=train_plan,
        updates=updates,batch=batch,eval_n=eval_n,checkpoints=checkpoints,
        learning_rate=.0007,entropy_coefficient=.02,entropy_off_after=boundary,
        optimizer='fresh Adam',gradient_clip='norm 2 separately per active sender/receiver module',role_loss_reduction='mean of both roles, including frozen constants',
        rng_namespace=10010,training_purpose=purpose,evaluation_seed=evaluation_seed,
        partition_parameters=selected,initial_sha256=initial_hash,source_hashes=hashes,
        source_checkpoint=dict(path=str(source),sha256=v8.sha(source)) if source else None,
        prepared_source=dict(path=str(out/f'prepared_{seed}.pt'),sha256=v8.sha(out/f'prepared_{seed}.pt')),
        sealed_never_trained=True,development_photos=True)
    write_json(dest/'config.json',cfg)
    torch.save([a.state_dict() for a in agents],dest/'initial.pt')
    optimizers=[torch.optim.Adam([p for p in a.parameters() if p.requires_grad],lr=.0007) for a in agents]
    torch.save([o.state_dict() for o in optimizers],dest/'initial_optimizer.pt')
    curve=[];start=time.monotonic()
    with (dest/'training.jsonl').open('w') as log:
        for update in range(updates+1):
            if update in checkpoints:
                assert v9.subset_verified(agents,frozen)
                scores=evaluate(agents,banks,bank,p,evaluation_seed,eval_n)
                curve.append(dict(update=update,scores=scores,frozen_modules_verified=True))
                write_json(dest/'curve.json',curve)
                torch.save([a.state_dict() for a in agents],dest/f'checkpoint_{update:04d}.pt')
                print(json.dumps(dict(run=dest.name,update=update,
                    old=scores['normal']['map_groups']['old']['both_accuracy'],
                    added=scores['normal']['map_groups']['added']['both_accuracy'])),flush=True)
            if update==updates:break
            rng=seed_for(seed,p,purpose,update)
            stats,learning,rows=v8.rollout(agents,banks,bank,train_plan,rng,batch,training=True,greedy=False,trace=True,split='train')
            positions=np.concatenate([row['positions'] for row in rows])
            mids=positions[:,0]*5+positions[:,1]-(positions[:,1]>positions[:,0])
            exposure={name:int(np.isin(mids,ids).sum()) for name,ids in group.items()}
            assert exposure['sealed']==0 and np.isin(mids,pool).all()
            weight=.02 if update<boundary else 0.;info=[]
            for who in (0,1):
                losses=[];components=[]
                for lp,ent,value,target in learning[who]:
                    policy=-(lp*(target-value).detach()).mean();vloss=.5*F.mse_loss(value,target)
                    losses.append(policy+vloss-weight*ent.mean())
                    components.append(dict(policy_loss=float(policy.detach()),value_loss=float(vloss.detach()),entropy=float(ent.mean().detach()),trainable=bool(lp.requires_grad)))
                loss=torch.stack(losses).mean();assert torch.isfinite(loss)
                optimizers[who].zero_grad(set_to_none=True);loss.backward()
                assert all(param.grad is None for param in agents[who].parameters() if not param.requires_grad)
                norms={}
                for role,prefixes in [('sender',v9.SENDER),('receiver',v9.RECEIVER)]:
                    params=[param for key,param in agents[who].named_parameters() if param.requires_grad and key.startswith(prefixes)]
                    if params:norms[role]=float(torch.nn.utils.clip_grad_norm_(params,2.))
                assert all(np.isfinite(value) for value in norms.values())
                optimizers[who].step()
                info.append(dict(loss=float(loss.detach()),components=components,gradient_norm_by_role=norms))
            log.write(json.dumps(dict(update=update+1,rng_seed=rng,world_sha256=stats['world_sha256'],exposure=exposure,
                mean_reward=stats['mean_reward'],both_accuracy=stats['both_accuracy'],single_accuracy=stats['single_accuracy'],
                entropy_weight=weight,agents=info))+'\n')
            if (update+1)%100==0:log.flush()
    final=evaluate(agents,banks,bank,p,evaluation_seed,eval_n,dest,modes=('normal','shuffle','blank','stochastic','erase_memory'))
    assert final['normal']==curve[-1]['scores']['normal'] and v9.subset_verified(agents,frozen)
    torch.save([a.state_dict() for a in agents],dest/'final.pt')
    torch.save([o.state_dict() for o in optimizers],dest/'final_optimizer.pt')
    write_json(dest/'result.json',dict(seed=seed,partition=p,arm=arm,updates=updates,batch=batch,scores=final,
        initial_sha256=initial_hash,final_sha256=v8.state_sha(agents),source_checkpoint=cfg['source_checkpoint'],
        frozen_modules_verified=True,sealed_never_trained=True,seconds=time.monotonic()-start))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'results/generalization_001')
    parser.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    parser.add_argument('--partitions',type=int,nargs='+',default=[1,2,3])
    parser.add_argument('--pre-updates',type=int,default=2400)
    parser.add_argument('--updates',type=int,default=600)
    parser.add_argument('--batch',type=int,default=512)
    parser.add_argument('--eval-n',type=int,default=9600)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    args.out.mkdir(parents=True,exist_ok=True)
    formal=not args.smoke
    if formal:
        assert (args.seeds,args.partitions,args.pre_updates,args.updates,args.batch,args.eval_n)==(SEEDS,[1,2,3],2400,600,512,9600)
        qa=json.loads((ROOT/'preflight_qa.json').read_text())
        assert qa['passed'] and qa['runner_sha256']==v8.sha(__file__)
    all_seeds=[seed_for(s,p,k,u) for s in args.seeds for p in args.partitions for k,n in ((1,args.pre_updates),(2,args.updates),(3,1)) for u in range(n)]
    assert len(all_seeds)==len(set(all_seeds))
    files=sources();hashes={str(f):v8.sha(f) for f in files}
    snapshot=args.out/('sources_'+v8.sha(__file__)[:12]);snapshot.mkdir(exist_ok=True)
    for i,f in enumerate(files):
        if f.suffix in ('.py','.md'):shutil.copy2(f,snapshot/f'{i:02d}_{f.name}')
    write_json(args.out/f'invocation_{time.time_ns()}.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),formal=formal,
        seeds=args.seeds,partitions=args.partitions,arms=ARMS,pre_updates=args.pre_updates,updates=args.updates,batch=args.batch,eval_n=args.eval_n,
        source_hashes=hashes,unique_rng_count=len(all_seeds),python=platform.python_version(),torch=str(torch.__version__),device='cpu'))
    bank=ImageBank();prepared={s:v8.prepare(s,bank,args.out) for s in args.seeds}
    if formal:
        control_results=[]
        for seed in args.seeds:
            path=args.out/f'individual_controls/s{seed}_identity/result.json'
            result=json.loads(path.read_text()) if path.exists() else controls8.run(seed,'identity',prepared[seed],bank,args.out)
            assert result['complete'] and result['passed'], 'Personal capability diagnostic failed; retain seed and investigate'
            control_results.append(result)
        write_json(args.out/'individual_controls/summary.json',dict(complete=True,passed=True,runs=len(control_results),results=control_results,
            diagnostic_weights_never_transferred=True))
    for seed in args.seeds:
        for p in args.partitions:
            fit(seed,p,'base',prepared[seed],bank,args.out,args.pre_updates,args.batch,args.eval_n)
            source=args.out/f's{seed}_p{p}_base/final.pt'
            for arm in ARMS:fit(seed,p,arm,prepared[seed],bank,args.out,args.updates,args.batch,args.eval_n,source)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
