"""Frozen nonlinguistic spatial preparation followed by fresh social learning."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'redesign_v0.11'))
import run_anchoring as v11
v10=v11.v10; v9=v11.v9; v8=v11.v8
from camp import ImageBank, remake_agents, projected_banks, write_json
sys.path.insert(0, str(ROOT.parent/'redesign_v0.12'))
import run_receiver as probe
from receiver_metrics import evaluate_groups
sys.path.insert(0, str(ROOT))
import private_preparation as private

SEEDS=[31101,31102,31103,31104]
ARMS=['control','equivariant']
TIMES=[0,100,300,600,1200,1800,2100,2400]
SENDER=('send_',)
RECEIVER=('receive_embedding.','actor.','receive_value.')


def sources():
    files=[Path(__file__),ROOT/'private_preparation.py',ROOT/'固定执行方案.md',
        ROOT/'前置审查.md',Path(v11.__file__),Path(probe.__file__),
        ROOT.parent/'redesign_v0.12/receiver_metrics.py',
        ROOT.parent/'redesign_v0.4/data/encoder_report.json']+v10.sources()
    return list(dict.fromkeys(p.resolve() for p in files))


def seed_for(seed,p,purpose,step=0,who=0):
    return int(np.random.SeedSequence([13013,seed,p,purpose,step,who]).generate_state(1,dtype=np.uint64)[0]>>np.uint64(1))


def worlds(bank,seed,p,step,n,training):
    plan=dict(v10.PLAN,map_pool=v10.partition(p)['old'].tolist(),allowed_sites=list(range(6)))
    return [v8.fixture(bank,np.random.default_rng(seed_for(seed,p,1 if training else 90,step,d)),
        n//2,plan,training,'train' if training else 'test') for d in (0,1)]


def select_parameters(agents):
    rows=[]
    for a in agents:
        row={'active':[],'frozen':[]}
        for key,param in a.named_parameters():
            active=key.startswith(SENDER+RECEIVER)
            assert active or key.startswith(('project.','memory.','slot_phi.')),key
            param.requires_grad_(active)
            row['active' if active else 'frozen'].append(key)
        rows.append(row)
    return rows


def update_agents(agents,opts,learning,weight):
    info=[]
    for who,(a,opt) in enumerate(zip(agents,opts)):
        parts=[]; losses=[]
        assert set(learning[who])=={'sender','receiver'}
        for role in ('sender','receiver'):
            lp,en,value,target=learning[who][role]
            policy=-(lp*(target-value).detach()).mean()
            vl=.5*F.mse_loss(value,target)
            part=policy+vl-weight*en.mean();losses.append(part)
            parts.append(dict(role=role,n=len(target),policy_loss=float(policy.detach()),
                value_loss=float(vl.detach()),entropy=float(en.mean().detach()),
                total=float(part.detach()),target_mean=float(target.mean())))
        loss=torch.stack(losses).mean();assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();norms={}
        for role,prefixes in [('sender',SENDER),('receiver',RECEIVER)]:
            params=[p for key,p in a.named_parameters() if key.startswith(prefixes)]
            assert params and all(p.requires_grad and p.grad is not None for p in params)
            norms[role]=float(torch.nn.utils.clip_grad_norm_(params,2.))
        assert all(np.isfinite(x) for x in norms.values())
        assert all(p.grad is None for p in a.parameters() if not p.requires_grad)
        info.append(dict(loss=float(loss.detach()),parts=parts,gradient_norm_by_role=norms))
    for opt in opts:opt.step()
    return info


@torch.no_grad()
def evaluate(agents,banks,bank,seed,p,n,dest=None,final=False):
    items=worlds(bank,seed,p,0,n,False);scores={}
    for mode in (('normal','shuffle','blank','stochastic','erase_memory') if final else ('normal',)):
        # v11 native action sampler is retained, with its separate namespace11011.
        stats,_,arr=v11.interact(agents,agents,banks,items,seed,p,0,91,mode=mode)
        scores[mode]=stats
        if dest is not None:np.savez_compressed(dest/f'final_{mode}.npz',**arr)
    return scores


@torch.no_grad()
def protocol(agents,banks,bank,p,dest,step):
    rows=[bank.pools['test',k][:4].tolist() for k in (0,1)]
    output=[]
    for direction in (0,1):
        context=probe.make_context(agents[direction],banks[direction],rows)
        logits=probe.receiver_logits(agents[1-direction])
        masks={k:np.isin(context['map_ids'],v) for k,v in v10.partition(p).items()}
        masks['common30']=np.ones(len(context['map_ids']),dtype=bool)
        metrics={mode:evaluate_groups(probe.probabilities(context,mode),context['positions'],logits,masks)
            for mode in ('native','greedy')}
        np.savez_compressed(dest/f'protocol_{step:04d}_d{direction}.npz',**context,receiver_logits=logits)
        output.append(dict(direction=direction,metrics=metrics))
    write_json(dest/f'protocol_{step:04d}.json',output)
    return output


def fit(seed,p,arm,prepared,bank,args,hashes):
    dest=args.out/f'social_s{seed}_p{p}_{arm}';dest.mkdir(exist_ok=False)
    source=args.out/f'private_s{seed}_p{p}_{arm}/transferred.pt'
    agents=remake_agents(seed,prepared,7,2,'identity')
    reference=[{k:v.clone() for k,v in a.state_dict().items()} for a in agents]
    for a,state,ref in zip(agents,torch.load(source,weights_only=True),reference):
        assert all(torch.equal(v,ref[k]) for k,v in state.items() if not k.startswith(('memory.','slot_phi.')))
        a.load_state_dict(state)
    selected=select_parameters(agents);frozen=v9.state_subset(agents)
    assert len({p.data_ptr() for a in agents for p in a.parameters()})==sum(len(list(a.parameters())) for a in agents)
    banks=projected_banks(agents,bank);initial=v8.state_sha(agents)
    times=sorted(set([0,args.social_updates]+[u for u in TIMES if u<args.social_updates]))
    cfg=dict(seed=seed,partition=p,arm=arm,updates=args.social_updates,batch=args.batch,eval_n=args.eval_n,
        checkpoints=times,entropy_off_after=2100,entropy_coefficient=.02,learning_rate=.0007,
        optimizer='fresh Adam',gradient_clip='norm2 separately per sender/receiver',
        role_loss_weights={'sender':.5,'receiver':.5},communications_per_update=args.batch,
        actions_per_update=args.batch*2,parameter_partition=selected,source_hashes=hashes,
        map_groups={k:v.tolist() for k,v in v10.partition(p).items()},training_pool=v10.partition(p)['old'].tolist(),
        source_checkpoint=dict(path=str(source),sha256=v8.sha(source)),initial_sha256=initial,
        rng_world_namespace=13013,rng_policy_namespace=11011,training_policy_stream=11,evaluation_policy_stream=91,
        sealed_never_trained=True,development_photos=True,private_outcomes_never_filter_runs=True)
    write_json(dest/'config.json',cfg)
    torch.save([a.state_dict() for a in agents],dest/'initial.pt')
    opts=[torch.optim.Adam([p for p in a.parameters() if p.requires_grad],lr=.0007) for a in agents]
    torch.save([o.state_dict() for o in opts],dest/'initial_optimizer.pt')
    curve=[];started=time.monotonic()
    with (dest/'training.jsonl').open('w') as log:
        for step in range(args.social_updates+1):
            if step in times:
                assert v9.subset_verified(agents,frozen)
                scores=evaluate(agents,banks,bank,seed,p,args.eval_n)
                protocol(agents,banks,bank,p,dest,step)
                curve.append(dict(update=step,scores=scores,state_sha256=v8.state_sha(agents)))
                write_json(dest/'curve.json',curve)
                torch.save([a.state_dict() for a in agents],dest/f'checkpoint_{step:04d}.pt')
                torch.save([o.state_dict() for o in opts],dest/f'optimizer_{step:04d}.pt')
                print(json.dumps(dict(run=dest.name,update=step,checkpoint_saved=True)),flush=True)
            if step==args.social_updates:break
            items=worlds(bank,seed,p,step,args.batch,True)
            stats,learning,arr=v11.interact(agents,agents,banks,items,seed,p,step,11,
                sender_grad=True,receiver_grad=True,mode='stochastic')
            assert stats['map_groups']['added']['n']==stats['map_groups']['sealed']['n']==0
            if step in (0,2100):np.savez_compressed(dest/f'train_{step+1:04d}.npz',**arr)
            weight=.02 if step<2100 else 0.
            info=update_agents(agents,opts,learning,weight)
            if step==0:
                torch.save([a.state_dict() for a in agents],dest/'after_first_update.pt')
                torch.save([o.state_dict() for o in opts],dest/'after_first_optimizer.pt')
            log.write(json.dumps(dict(update=step+1,entropy_weight=weight,stats=stats,agents=info))+'\n')
            if (step+1)%100==0:log.flush()
    final=evaluate(agents,banks,bank,seed,p,args.eval_n,dest,True)
    assert final['normal']==curve[-1]['scores']['normal'] and v9.subset_verified(agents,frozen)
    torch.save([a.state_dict() for a in agents],dest/'final.pt')
    torch.save([o.state_dict() for o in opts],dest/'final_optimizer.pt')
    write_json(dest/'result.json',dict(seed=seed,partition=p,arm=arm,updates=args.social_updates,scores=final,
        initial_sha256=initial,final_sha256=v8.state_sha(agents),seconds=time.monotonic()-started,
        frozen_modules_verified=True,sealed_never_trained=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/spatial_001')
    parser.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    parser.add_argument('--partitions',type=int,nargs='+',default=[1,2,3])
    parser.add_argument('--private-updates',type=int,default=2400)
    parser.add_argument('--social-updates',type=int,default=2400)
    parser.add_argument('--batch',type=int,default=512)
    parser.add_argument('--eval-n',type=int,default=9600)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);args.out=args.out.resolve()
    assert args.batch%2==0 and args.eval_n%120==0
    hashes={str(p):v8.sha(p) for p in sources()}
    if not args.smoke:
        assert (args.seeds,args.partitions,args.private_updates,args.social_updates,args.batch,args.eval_n)==(SEEDS,[1,2,3],2400,2400,512,9600)
        qa=json.loads((ROOT/'preflight_qa.json').read_text())
        assert qa['passed'] and qa['source_hashes']==hashes
    report=json.loads((ROOT.parent/'redesign_v0.4/data/encoder_report.json').read_text())
    assert report['manifest_sha256']==v8.sha(ROOT.parent/'redesign_v0.4/data/manifest.json')
    assert report['features_sha256']==v8.sha(ROOT.parent/'redesign_v0.4/data/features.npz')
    args.out.mkdir(parents=True,exist_ok=False);snap=args.out/'frozen_sources';snap.mkdir()
    for i,path in enumerate(sources()):
        if path.suffix in ('.py','.md','.json'):shutil.copy2(path,snap/f'{i:02d}_{path.name}')
    if not args.smoke:shutil.copy2(ROOT/'preflight_qa.json',snap/'preflight_qa.json')
    write_json(args.out/'invocation.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),formal=not args.smoke,
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},arms=ARMS,
        source_hashes=hashes,python=platform.python_version(),torch=str(torch.__version__),device='cpu'))
    bank=ImageBank();prepared={s:v8.prepare(s,bank,args.out) for s in args.seeds}
    for s,p,a in product(args.seeds,args.partitions,ARMS):
        private.run(s,p,a,prepared[s],bank,args.out,args.private_updates,args.batch,hashes)
    assert all((args.out/f'private_s{s}_p{p}_{a}/result.json').exists() for s,p,a in product(args.seeds,args.partitions,ARMS))
    for s,p,a in product(args.seeds,args.partitions,ARMS):fit(s,p,a,prepared[s],bank,args,hashes)
    write_json(args.out/'training_complete.json',dict(status='complete',private_runs=len(args.seeds)*len(args.partitions)*2,
        social_runs=len(args.seeds)*len(args.partitions)*2,finished_utc=datetime.now(timezone.utc).isoformat(),source_hashes=hashes))
    print('TRAINING COMPLETE',flush=True)


if __name__=='__main__':main()
