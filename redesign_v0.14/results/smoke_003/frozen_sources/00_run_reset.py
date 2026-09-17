"""Paired reset of privately prepared visual interfaces; unchanged v13 social learning."""
from __future__ import annotations
import argparse,copy,json,platform,shutil,sys,time
from datetime import datetime,timezone
from itertools import product
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.13'))
import run_spatial as v13
v11=v13.v11;v10=v13.v10;v9=v13.v9;v8=v13.v8
ImageBank=v13.ImageBank;remake_agents=v13.remake_agents
projected_banks=v13.projected_banks;write_json=v13.write_json
select_parameters=v13.select_parameters;update_agents=v13.update_agents
worlds=v13.worlds;evaluate=v13.evaluate;protocol=v13.protocol
TIMES=v13.TIMES;SEEDS=v13.SEEDS;PREFIXES=('memory.','slot_phi.')
DEFAULT_SOURCE=PROJECT/'redesign_v0.13/results/spatial_001'

def sources():
    return list(dict.fromkeys([Path(__file__).resolve(),ROOT/'固定执行方案.md',ROOT/'前置科学审查.md',ROOT/'前置审查.md',ROOT/'check_manipulation.py',*v13.sources()]))

def source_receipt(args):
    inv=json.loads((args.source_batch/'invocation.json').read_text())
    done=json.loads((args.source_batch/'training_complete.json').read_text())
    assert done['status']=='complete'
    paths=[args.source_batch/'invocation.json',args.source_batch/'training_complete.json']
    for s in args.seeds:
        paths.extend([args.source_batch/f'prepared_{s}.pt',args.source_batch/f'preparation_{s}.json'])
        for p in args.partitions:
            priv=args.source_batch/f'private_s{s}_p{p}_control'
            paths.extend(priv/f for f in ('initial.pt','final.pt','transferred.pt','config.json','result.json'))
            paths.extend(x for x in (args.source_batch/f'social_s{s}_p{p}_control').iterdir() if x.is_file())
    if not args.smoke:
        paths.extend(args.source_batch/f for f in ('audit_execution.json','independent_recount.json','私人空间能力与共同符号形成研究报告.md'))
        manifest_path=args.source_batch/'completion_manifest.json'
        manifest=json.loads(manifest_path.read_text());assert manifest['status']=='complete'
        for path in paths:assert v8.sha(path)==manifest['artifacts'][str(path.relative_to(PROJECT))],str(path)
        audit=json.loads((args.source_batch/'audit_execution.json').read_text());assert audit['passed']
        paths.append(manifest_path)
    for path,digest in inv['source_hashes'].items():assert v8.sha(path)==digest,path
    return dict(source_batch=str(args.source_batch),formal_source=inv['formal'],
        inherited_control_runs=len(args.seeds)*len(args.partitions),
        files={str(p):v8.sha(p) for p in paths},source_training_hashes=inv['source_hashes'],
        scope='All consumed private states and retained social artifacts; control is previously observed development evidence')

def fit(seed,p,arm,prepared,bank,args,hashes):
    dest=args.out/f'social_s{seed}_p{p}_{arm}';dest.mkdir(exist_ok=False)
    original=args.source_batch/f'private_s{seed}_p{p}_control/initial.pt'
    retained=args.source_batch/f'social_s{seed}_p{p}_control/initial.pt'
    before=torch.load(original,weights_only=True)['agents']
    state=copy.deepcopy(torch.load(retained,weights_only=True))
    assert arm in ('reset','retained_replay')
    agents=remake_agents(seed,prepared,7,2,'identity')
    for who,a in enumerate(agents):
        for key,value in a.state_dict().items():
            assert torch.equal(value,before[who][key]), key
            if key.startswith(PREFIXES):
                if arm=='reset':state[who][key]=before[who][key].clone()
            else:assert torch.equal(value,state[who][key]), key
        a.load_state_dict(state[who])
    source=dest/'source_state.pt';torch.save(state,source)
    source_inputs=dict(private_initial=dict(path=str(original),sha256=v8.sha(original)),
        retained_social_initial=dict(path=str(retained),sha256=v8.sha(retained)),
        reset_prefixes=list(PREFIXES),intervention='initial values restored and frozen' if arm=='reset' else 'smoke-only retained replay')
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
        source_checkpoint=dict(path=str(source),sha256=v8.sha(source)),source_inputs=source_inputs,initial_sha256=initial,
        rng_world_namespace=13014,rng_policy_namespace=11011,training_policy_stream=11,evaluation_policy_stream=91,
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
            if step==2100:
                torch.save([a.state_dict() for a in agents],dest/'after_2101_update.pt')
                torch.save([o.state_dict() for o in opts],dest/'after_2101_optimizer.pt')
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
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'results/reset_001')
    ap.add_argument('--source-batch',type=Path,default=DEFAULT_SOURCE)
    ap.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    ap.add_argument('--partitions',type=int,nargs='+',default=[1,2,3])
    ap.add_argument('--social-updates',type=int,default=2400)
    ap.add_argument('--batch',type=int,default=512)
    ap.add_argument('--eval-n',type=int,default=9600)
    ap.add_argument('--arms',nargs='+',default=['reset'])
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1);args.out=args.out.resolve();args.source_batch=args.source_batch.resolve()
    assert args.batch%2==0 and args.eval_n%120==0 and args.social_updates>0
    assert len(set(args.arms))==len(args.arms) and set(args.arms)<= {'reset','retained_replay'}
    hashes={str(p):v8.sha(p) for p in sources()};receipt=source_receipt(args)
    if not args.smoke:
        assert (args.source_batch,args.seeds,args.partitions,args.social_updates,args.batch,args.eval_n,args.arms)==(DEFAULT_SOURCE,SEEDS,[1,2,3],2400,512,9600,['reset'])
        qa=json.loads((ROOT/'preflight_qa.json').read_text());assert qa['passed'] and qa['source_hashes']==hashes
        assert qa['source_receipt_sha256']==v8.sha(ROOT/'source_receipt.json')
        assert receipt==json.loads((ROOT/'source_receipt.json').read_text())
    args.out.mkdir(parents=True,exist_ok=False);snap=args.out/'frozen_sources';snap.mkdir()
    for i,path in enumerate(sources()):
        if path.suffix in ('.py','.md','.json'):shutil.copy2(path,snap/f'{i:02d}_{path.name}')
    if not args.smoke:shutil.copy2(ROOT/'preflight_qa.json',snap/'preflight_qa.json')
    write_json(args.out/'source_receipt.json',receipt)
    write_json(args.out/'invocation.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),formal=not args.smoke,
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},arms=args.arms,
        source_hashes=hashes,python=platform.python_version(),torch=str(torch.__version__),device='cpu'))
    bank=ImageBank()
    for s in args.seeds:
        source=args.source_batch/f'prepared_{s}.pt';shutil.copy2(source,args.out/source.name)
        prepared=torch.load(source,weights_only=True)
        for p,arm in product(args.partitions,args.arms):fit(s,p,arm,prepared,bank,args,hashes)
    write_json(args.out/'training_complete.json',dict(status='complete',new_private_runs=0,
        new_social_runs=len(args.seeds)*len(args.partitions)*len(args.arms),finished_utc=datetime.now(timezone.utc).isoformat(),source_hashes=hashes))
    print('TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
