"""Fixed 2x2 policy/value input scales on the frozen reset encoder."""
from __future__ import annotations
import argparse,json,platform,shutil,sys,time
from datetime import datetime,timezone
from itertools import product
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.15'))
import run_scaled as v15
sys.path.insert(0,str(ROOT))
import branch_interface as branches
v14=v15.v14;v13=v15.v13;v11=v15.v11;v10=v15.v10;v9=v15.v9;v8=v15.v8
ImageBank=v15.ImageBank;remake_agents=v15.remake_agents;projected_banks=v15.projected_banks;write_json=v15.write_json
select_parameters=v15.select_parameters;update_agents=v15.update_agents;worlds=v15.worlds;evaluate=v15.evaluate
TIMES=v15.TIMES;SEEDS=v15.SEEDS
DEFAULT_SOURCE=PROJECT/'redesign_v0.13/results/spatial_001'
DEFAULT_RESET=PROJECT/'redesign_v0.14/results/reset_001'
DEFAULT_SCALED=PROJECT/'redesign_v0.15/results/scaled_001'
DEFAULT_CALIBRATION=PROJECT/'redesign_v0.15/calibration_001'
FACTORS={'neither_replay':(0,0),'policy_only_scale':(1,0),'value_only_scale':(0,1),'both_replay':(1,1)}
ARMS=['policy_only_scale','value_only_scale']

def read(p):return json.loads(Path(p).read_text())
def sources():
    return list(dict.fromkeys([Path(__file__).resolve(),ROOT/'branch_interface.py',ROOT/'固定执行方案.md',ROOT/'前置科学审查.md',ROOT/'前置审查.md',*v15.sources()]))

def source_receipt(args):
    prior=v15.source_receipt(args)
    inv=read(args.scaled_batch/'invocation.json');done=read(args.scaled_batch/'training_complete.json');assert done['status']=='complete'
    paths=[args.scaled_batch/n for n in ('invocation.json','training_complete.json','source_receipt.json','calibration_receipt.json')]
    for seed in args.seeds:
        paths.append(args.scaled_batch/f'prepared_{seed}.pt')
        for p in args.partitions:paths.extend(x for x in (args.scaled_batch/f'social_s{seed}_p{p}_reset_scaled').iterdir() if x.is_file())
    if not args.smoke:
        paths.extend(args.scaled_batch/n for n in ('audit_execution.json','independent_recount.json','audit_report_review.json','视觉编码幅度与共同符号形成研究报告.md'))
        manifest=read(args.scaled_batch/'completion_manifest.json');assert manifest['status']=='complete'
        for path in paths:assert v8.sha(path)==manifest['artifacts'][str(path.relative_to(PROJECT))],str(path)
        assert read(args.scaled_batch/'audit_execution.json')['passed'];paths.append(args.scaled_batch/'completion_manifest.json')
    for path,digest in inv['source_hashes'].items():assert v8.sha(path)==digest,path
    return dict(**prior,scaled=dict(source_batch=str(args.scaled_batch),source_training_hashes=inv['source_hashes'],files={str(p):v8.sha(p) for p in paths}),
        factorial_references='00=v14 reset;11=v15 reset_scaled; all previously observed development sources')

def calibration_receipt(args):
    path=args.calibration/'calibration_complete.json';done=read(path)
    assert done['status']=='complete' and done['seeds']==args.seeds and done['partitions']==args.partitions
    assert done['train_only'] and done['training_updates']==0 and done['persons']==2*len(args.seeds)*len(args.partitions)
    for path0,digest in done['source_hashes'].items():assert v8.sha(path0)==digest,path0
    for name,digest in done['files'].items():assert v8.sha(args.calibration/name)==digest,name
    inherited=read(args.scaled_batch/'calibration_receipt.json')
    assert Path(inherited['path'])==path and inherited['sha256']==v8.sha(path) and inherited['files']==done['files']
    assert read(args.calibration/'source_receipt.json')==v15.source_receipt(args)
    return dict(path=str(path),sha256=v8.sha(path),files=done['files'],scope='Exact v15 train-only calibration reused; no calibration or outcome-dependent scale fitting')

@torch.no_grad()
def protocol(agents,banks,bank,p,dest,step):
    rows=[bank.pools['test',k][:4].tolist() for k in (0,1)];output=[]
    for direction in (0,1):
        sender=agents[direction]
        context=v13.probe.make_context(sender,banks[direction],rows)
        raw=torch.from_numpy(context['h'])
        context['raw_h']=context['h'].copy()
        context['policy_h']=branches.policy_h(sender,raw).numpy()
        context['value_h']=branches.value_h(sender,raw).numpy()
        logits=v13.probe.receiver_logits(agents[1-direction])
        masks={k:np.isin(context['map_ids'],v) for k,v in v10.partition(p).items()};masks['common30']=np.ones(len(context['map_ids']),dtype=bool)
        metrics={mode:v13.evaluate_groups(v13.probe.probabilities(context,mode),context['positions'],logits,masks) for mode in ('native','greedy')}
        np.savez_compressed(dest/f'protocol_{step:04d}_d{direction}.npz',**context,receiver_logits=logits)
        output.append(dict(direction=direction,metrics=metrics))
    write_json(dest/f'protocol_{step:04d}.json',output);return output

def fit(seed,p,arm,prepared,bank,args,hashes):
    dest=args.out/f'social_s{seed}_p{p}_{arm}';dest.mkdir(exist_ok=False)
    retained=args.source_batch/f'social_s{seed}_p{p}_control/initial.pt'
    reset=args.reset_batch/f'social_s{seed}_p{p}_reset/initial.pt'
    calibration=args.calibration/f's{seed}_p{p}/calibration.json'
    cal=read(calibration);base=torch.load(reset,weights_only=True)
    assert arm in FACTORS
    factors=FACTORS[arm]
    scales=[float(q['alpha_float32']) for q in cal['persons']]
    branch_scales=[{'policy':alpha if factors[0] else 1.,'value':alpha if factors[1] else 1.} for alpha in scales]
    agents=remake_agents(seed,prepared,7,2,'identity')
    for who,a in enumerate(agents):
        a.load_state_dict(base[who]);branches.attach_scales(a,branch_scales[who]['policy'],branch_scales[who]['value'])
    source=dest/'source_state.pt';torch.save([a.state_dict() for a in agents],source)
    source_inputs=dict(reset_initial=dict(path=str(reset),sha256=v8.sha(reset)),
        retained_social_initial=dict(path=str(retained),sha256=v8.sha(retained)),
        calibration=dict(path=str(calibration),sha256=v8.sha(calibration)),
        scaled_initial=dict(path=str(args.scaled_batch/f'social_s{seed}_p{p}_reset_scaled/initial.pt'),sha256=v8.sha(args.scaled_batch/f'social_s{seed}_p{p}_reset_scaled/initial.pt')),
        intervention='fixed alpha at policy and/or value input; raw observation and all base tensors equal reset')
    selected=select_parameters(agents);frozen=v9.state_subset(agents)
    for a,state in zip(agents,frozen):
        state['send_context.h_scale']=a.send_context.h_scale.clone()
        state['send_value.h_scale']=a.send_value.h_scale.clone()
    assert len({p.data_ptr() for a in agents for p in a.parameters()})==sum(len(list(a.parameters())) for a in agents)
    banks=projected_banks(agents,bank);initial=v8.state_sha(agents)
    times=sorted(set([0,args.social_updates]+[u for u in TIMES if u<args.social_updates]))
    cfg=dict(seed=seed,partition=p,arm=arm,updates=args.social_updates,batch=args.batch,eval_n=args.eval_n,
        checkpoints=times,entropy_off_after=2100,entropy_coefficient=.02,learning_rate=.0007,
        optimizer='fresh Adam',gradient_clip='norm2 separately per sender/receiver',
        role_loss_weights={'sender':.5,'receiver':.5},communications_per_update=args.batch,
        actions_per_update=args.batch*2,parameter_partition=selected,source_hashes=hashes,
        map_groups={k:v.tolist() for k,v in v10.partition(p).items()},training_pool=v10.partition(p)['old'].tolist(),
        source_checkpoint=dict(path=str(source),sha256=v8.sha(source)),source_inputs=source_inputs,calibration_scales_float32=scales,branch_scales_float32=branch_scales,factors=list(factors),initial_sha256=initial,
        protocol_h_semantics="h equals raw_h; policy_h/value_h are the actual scaled branch inputs",
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
    ap.add_argument('--out',type=Path,default=ROOT/'results/branches_001')
    ap.add_argument('--source-batch',type=Path,default=DEFAULT_SOURCE)
    ap.add_argument('--reset-batch',type=Path,default=DEFAULT_RESET)
    ap.add_argument('--scaled-batch',type=Path,default=DEFAULT_SCALED)
    ap.add_argument('--calibration',type=Path,default=DEFAULT_CALIBRATION)
    ap.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    ap.add_argument('--partitions',type=int,nargs='+',default=[1,2,3])
    ap.add_argument('--social-updates',type=int,default=2400)
    ap.add_argument('--batch',type=int,default=512);ap.add_argument('--eval-n',type=int,default=9600)
    ap.add_argument('--arms',nargs='+',default=ARMS);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1)
    for key in ('out','source_batch','reset_batch','scaled_batch','calibration'):setattr(args,key,getattr(args,key).resolve())
    assert args.batch%2==0 and args.eval_n%120==0 and args.social_updates>0
    assert len(set(args.arms))==len(args.arms) and set(args.arms)<=set(FACTORS)
    hashes={str(p):v8.sha(p) for p in sources()};receipt=source_receipt(args);creceipt=calibration_receipt(args)
    if not args.smoke:
        assert (args.source_batch,args.reset_batch,args.scaled_batch,args.calibration,args.seeds,args.partitions,args.social_updates,args.batch,args.eval_n,args.arms)==(DEFAULT_SOURCE,DEFAULT_RESET,DEFAULT_SCALED,DEFAULT_CALIBRATION,SEEDS,[1,2,3],2400,512,9600,ARMS)
        qa=read(ROOT/'preflight_qa.json');assert qa['passed'] and qa['source_hashes']==hashes
        assert qa['source_receipt_sha256']==v8.sha(ROOT/'source_receipt.json') and receipt==read(ROOT/'source_receipt.json')
        assert qa['calibration_complete_sha256']==creceipt['sha256']
    args.out.mkdir(parents=True,exist_ok=False);snap=args.out/'frozen_sources';snap.mkdir()
    for i,path in enumerate(sources()):
        if path.suffix in ('.py','.md','.json'):shutil.copy2(path,snap/f'{i:02d}_{path.name}')
    if not args.smoke:shutil.copy2(ROOT/'preflight_qa.json',snap/'preflight_qa.json')
    write_json(args.out/'source_receipt.json',receipt);write_json(args.out/'calibration_receipt.json',creceipt)
    write_json(args.out/'invocation.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),formal=not args.smoke,phase='social',
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},arms=args.arms,source_hashes=hashes,
        python=platform.python_version(),torch=str(torch.__version__),device='cpu'))
    bank=ImageBank()
    for seed in args.seeds:
        source=args.source_batch/f'prepared_{seed}.pt';shutil.copy2(source,args.out/source.name);prepared=torch.load(source,weights_only=True)
        for p,arm in product(args.partitions,args.arms):fit(seed,p,arm,prepared,bank,args,hashes)
    write_json(args.out/'training_complete.json',dict(status='complete',new_private_runs=0,new_social_runs=len(args.seeds)*len(args.partitions)*len(args.arms),
        source_hashes=hashes,finished_utc=datetime.now(timezone.utc).isoformat()))
    print('TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
