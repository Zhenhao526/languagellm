"""Matched fresh private readouts of two frozen DINO-based visual interfaces."""
from __future__ import annotations
import argparse, hashlib, itertools, json, platform, shutil, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.15'))
import run_scaled as v15
from camp import MAPS,scene_visual,CampAgent
sys.path.insert(0,str(ROOT))
import readout_interface as ri

SEEDS=[31101,31102,31103,31104]
TIMES=[0,100,300,600,1200,1800,2100,2400]
INTERFACES=['retained','reset_scaled']; SIGNALS=['reward','ce']
CHUNK=512

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def array_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def state_sha(module):
    h=hashlib.sha256()
    for k,v in module.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def arrays_sha(arrays):
    h=hashlib.sha256()
    for key in sorted(arrays):
        a=np.ascontiguousarray(arrays[key]);h.update(key.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def sources(scaled_source):
    inherited=read(scaled_source/'invocation.json')['source_hashes']
    for p,digest in inherited.items():assert sha(p)==digest,p
    own=[ROOT/n for n in ('run_readout.py','readout_interface.py','固定执行方案.md','前置审查.md')]
    return {**inherited,**{str(p):sha(p) for p in own}}

def inputs(scaled_source,seeds,partitions,formal):
    source=Path(read(scaled_source/'invocation.json')['args']['source_batch'])
    assert source.is_absolute()
    paths=[]
    for batch in (source,scaled_source):
        assert read(batch/'training_complete.json')['status']=='complete'
        local=[batch/n for n in ('invocation.json','training_complete.json')]
        if batch==scaled_source:local.append(batch/'calibration_receipt.json')
        for seed in seeds:
            local.append(batch/f'prepared_{seed}.pt')
            for p in partitions:
                folder=batch/(f'social_s{seed}_p{p}_control' if batch==source else f'social_s{seed}_p{p}_reset_scaled')
                local.extend(folder/n for n in ('initial.pt','config.json'))
                if batch==source:
                    private=batch/f'private_s{seed}_p{p}_control'
                    local.extend(private/n for n in ('initial.pt','final.pt','transferred.pt','config.json'))
        if formal:
            manifest=read(batch/'completion_manifest.json');assert manifest['status']=='complete'
            for p in local:assert sha(p)==manifest['artifacts'][str(p.relative_to(PROJECT))],p
            local.append(batch/'completion_manifest.json')
        paths.extend(local)
    cal=read(scaled_source/'calibration_receipt.json')
    assert sha(cal['path'])==cal['sha256']
    paths.append(Path(cal['path']))
    for seed,p in itertools.product(seeds,partitions):
        cp=Path(cal['path']).parent/f's{seed}_p{p}/calibration.json'
        assert sha(cp)==cal['files'][f's{seed}_p{p}/calibration.json']
        paths.append(cp)
    paths.extend(PROJECT/'redesign_v0.4/data'/n for n in ('manifest.json','features.npz','encoder_report.json'))
    encoder=read(paths[-1]);assert sha(paths[-3])==encoder['manifest_sha256'] and sha(paths[-2])==encoder['features_sha256']
    return source,{str(p):sha(p) for p in paths}

def world_table(bank,p,split):
    pools=[np.sort(bank.pools[split,k]) for k in (0,1)]
    maps=np.sort(v15.v13.private.partition_maps(p)['old']) if split=='train' else np.arange(30)
    pair=np.asarray(list(itertools.product(*pools)),np.int64)
    mids=np.repeat(maps,len(pair));ids=np.tile(pair,(len(maps),1))
    return dict(map_ids=mids,positions=MAPS[mids],photo_ids=ids),pools

@torch.no_grad()
def make_cache(seed,p,source,scaled_source,bank,out):
    folder=out/f'cache_s{seed}_p{p}';folder.mkdir()
    prepared=torch.load(source/f'prepared_{seed}.pt',weights_only=True)
    assert all(torch.equal(a[k],b[k]) for a,b in zip(prepared,torch.load(scaled_source/f'prepared_{seed}.pt',weights_only=True)) for k in a)
    paths={i:(source/f'social_s{seed}_p{p}_control/initial.pt' if i=='retained' else scaled_source/f'social_s{seed}_p{p}_reset_scaled/initial.pt') for i in INTERFACES}
    states={i:torch.load(path,weights_only=True) for i,path in paths.items()}
    old_initial=torch.load(source/f'private_s{seed}_p{p}_control/initial.pt',weights_only=True)['agents']
    cal=read(scaled_source/'calibration_receipt.json')
    c=read(Path(cal['path']).parent/f's{seed}_p{p}/calibration.json')
    records=[];caches={}
    for interface in INTERFACES:
        agents=v15.remake_agents(seed,prepared,7,2,'identity')
        for who,a in enumerate(agents):
            if interface=='retained':a.load_state_dict(states[interface][who],strict=True)
            else:
                v15.scaled.load_scaled_state(a,states[interface][who])
                assert a.visual_scale.item()==c['persons'][who]['alpha_float32']
                for k,value in old_initial[who].items():assert torch.equal(a.state_dict()[k],value),k
            a.requires_grad_(False);a.train()
            before=state_sha(a)
            projected=a.project(bank.features).detach()
            entries={}
            for split in ('train','test'):
                world,pools=world_table(bank,p,split)
                pieces=[];raw_pieces=[]
                for lo in range(0,len(world['map_ids']),CHUNK):
                    sl=slice(lo,lo+CHUNK)
                    visual=scene_visual(world['positions'][sl],world['photo_ids'][sl],projected)
                    effective=a.observe(visual);raw=CampAgent.observe(a,visual)
                    assert torch.equal(effective,raw if interface=='retained' else raw*a.visual_scale)
                    pieces.append(effective.numpy());raw_pieces.append(raw.numpy())
                h=np.concatenate(pieces);raw_h=np.concatenate(raw_pieces)
                assert h.shape==(len(world['map_ids']),96) and np.isfinite(h).all()
                path=folder/f'{interface}_d{who}_{split}.npz'
                np.savez_compressed(path,**world,h=h,raw_h=raw_h)
                entries[split]=dict(file=str(path.relative_to(out)),sha256=sha(path),worlds=len(h),h_sha256=array_sha(h),
                    mean_l2=float(np.linalg.norm(h.astype(np.float64),axis=1).mean()),photo_pools=[x.tolist() for x in pools])
                caches[interface,who,split]={**world,'h':torch.from_numpy(h)}
            assert state_sha(a)==before
            records.append(dict(interface=interface,direction=who,source_checkpoint=str(paths[interface]),source_sha256=sha(paths[interface]),
                state_sha256=before,scale=1. if interface=='retained' else a.visual_scale.item(),entries=entries))
    # Relative input scale is inherited, not retuned against the new scores.
    for who in (0,1):
        for k,v in states['retained'][who].items():
            if not k.startswith(('memory.','slot_phi.')):assert torch.equal(v,states['reset_scaled'][who][k]),k
    write(folder/'cache_receipt.json',dict(seed=seed,partition=p,records=records,chunk=CHUNK,dtype='float32',
        all_parameters_frozen=True,grad_enabled=False,model_training=True,new_dino_inferences=0,
        scope='Unified current-interface forward in 512-row chunks; full old/train and all30/test photo support.'))
    return caches,folder/'cache_receipt.json'

def metrics(logits,world,p):
    x=np.asarray(logits,np.float64);shift=x-x.max(-1,keepdims=True);e=np.exp(shift)
    logq=shift-np.log(e.sum(-1,keepdims=True));q=np.exp(logq)
    pos=world['positions'];n=len(pos);selected=q[np.arange(n)[:,None],np.arange(2)[None,:],pos]
    selected_logq=logq[np.arange(n)[:,None],np.arange(2)[None,:],pos]
    greedy=x.argmax(-1);good=greedy==pos
    entropy=-(q*logq).sum(-1)
    ties=(x==x.max(-1,keepdims=True)).sum(-1)>1
    groups={**v15.v13.private.partition_maps(p),'common30':np.arange(30)};result={}
    for group,pool in groups.items():
        take=np.isin(world['map_ids'],pool)
        result[group]=dict(n=int(take.sum()),J=float(good[take].all(-1).mean()),single=float(good[take].mean()),
            Q=float(selected[take].prod(-1).mean()),native_single=float(selected[take].mean()),
            entropy=float(entropy[take].mean()),nll=float(-selected_logq[take].mean()),
            exact_max_tie_goal_count=int(ties[take].sum()),exact_max_tie_rate=float(ties[take].mean()),
            correct_joint=int(good[take].all(-1).sum()),correct_goals=int(good[take].sum()))
    return result

@torch.no_grad()
def evaluate(head,cache,p,dest,step):
    logits=torch.cat([head(cache['h'][lo:lo+CHUNK]).reshape(-1,2,6) for lo in range(0,len(cache['h']),CHUNK)]).numpy()
    path=dest/f'evaluation_{step:04d}.npz'
    np.savez_compressed(path,map_ids=cache['map_ids'],positions=cache['positions'],photo_ids=cache['photo_ids'],logits=logits)
    return dict(update=step,head_sha256=state_sha(head),scores=metrics(logits,cache,p),raw_sha256=sha(path))

def cache_indices(cache,world,bank_size):
    lut=np.full((30,bank_size,bank_size),-1,np.int64)
    lut[cache['map_ids'],cache['photo_ids'][:,0],cache['photo_ids'][:,1]]=np.arange(len(cache['map_ids']))
    return lut

def fit(seed,p,who,interface,signal,cache,bank,out,args,receipt,hashes):
    folder=out/f's{seed}_p{p}_d{who}_{interface}_{signal}';folder.mkdir()
    head=ri.make_head(seed,p,who);initial=state_sha(head)
    assert sum(p.numel() for p in head.parameters())==10476
    torch.save(head.state_dict(),folder/'initial.pt')
    opt=torch.optim.Adam(head.parameters(),lr=.0007)
    torch.save(opt.state_dict(),folder/'initial_optimizer.pt')
    times=sorted(set([0,args.updates]+[t for t in TIMES if t<args.updates]))
    cfg=dict(seed=seed,partition=p,direction=who,interface=interface,signal=signal,updates=args.updates,batch_pairs=512,
        worlds_per_update=1024,learning_rate=.0007,gradient_clip=2.,entropy_coefficient=.02,entropy_off_after=2100,
        initial_sha256=initial,head_initialization_seed=ri.head_seed(seed,p,who),
        cache_receipt=str(receipt),cache_receipt_sha256=sha(receipt),source_hashes=hashes,
        checkpoints=times,only_head_trainable=True,correct_labels_available=signal=='ce',
        signal_scope='RL own binary reward only; CE correct selected-goal location label, both same entropy schedule',
        test_world_count=1920,training_pool=v15.v13.private.partition_maps(p)['old'].tolist(),new_communication_training=False)
    write(folder/'config.json',cfg)
    train=cache[interface,who,'train'];test=cache[interface,who,'test']
    lut=cache_indices(train,None,len(bank.entries))
    curve=[];started=time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for step in range(args.updates+1):
            if step in times:
                curve.append(evaluate(head,test,p,folder,step));write(folder/'curve.json',curve)
                torch.save(head.state_dict(),folder/f'checkpoint_{step:04d}.pt')
                torch.save(opt.state_dict(),folder/f'optimizer_{step:04d}.pt')
                print(json.dumps(dict(run=folder.name,update=step,seconds=round(time.monotonic()-started,2))),flush=True)
            if step==args.updates:break
            world=ri.fixture(bank,seed,p,who,step,512)
            ids=world['photo_ids']
            ix=np.concatenate([lut[world[k],ids[:,0],ids[:,1]] for k in ('source_map','target_map')])
            assert (ix>=0).all()
            h=train['h'][torch.from_numpy(ix)]
            opt.zero_grad(set_to_none=True)
            loss,components,trace=ri.loss_terms(head,h,world,signal,step)
            assert bool(torch.isfinite(loss));loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),2.))
            assert np.isfinite(norm)
            opt.step()
            if step in (0,2100):
                arrays={'world__'+k:np.asarray(v) for k,v in world.items()}
                arrays.update({'trace__'+k:np.asarray(v) for k,v in trace.items()})
                arrays['cache_indices']=ix
                np.savez_compressed(folder/f'train_{step+1:04d}.npz',**arrays)
                torch.save(head.state_dict(),folder/f'after_{step+1:04d}.pt')
                torch.save(opt.state_dict(),folder/f'after_{step+1:04d}_optimizer.pt')
            log.write(json.dumps(dict(update=step+1,world_sha256=arrays_sha(world),cache_indices_sha256=array_sha(ix),
                trace_sha256=arrays_sha(trace),components=components,gradient_norm=norm,
                clip_coefficient=min(1.,2./(norm+1e-6))))+'\n')
            if (step+1)%100==0:log.flush()
    torch.save(head.state_dict(),folder/'final.pt')
    write(folder/'result.json',dict(status='complete',seed=seed,partition=p,direction=who,interface=interface,signal=signal,
        updates=args.updates,initial_sha256=initial,final_sha256=state_sha(head),seconds=time.monotonic()-started,
        simulated_selected_goal_actions=args.updates*1024,ce_labels=args.updates*1024 if signal=='ce' else 0,
        scores=curve[-1]['scores'],only_head_trained=True,no_communication=True,cache_receipt_sha256=sha(receipt)))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--dev',action='store_true');ap.add_argument('--updates',type=int,default=2400)
    args=ap.parse_args();assert args.updates>0
    torch.set_num_threads(1)
    scaled_source=PROJECT/('redesign_v0.15/results/smoke_002' if args.dev else 'redesign_v0.15/results/scaled_001')
    seeds=[99513] if args.dev else SEEDS;parts=[1] if args.dev else [1,2,3]
    if not args.dev:assert args.updates==2400
    hashes=sources(scaled_source);source,receipt=inputs(scaled_source,seeds,parts,not args.dev)
    gate=None
    if not args.dev:
        gate_path=ROOT/'preflight_qa.json';g=read(gate_path)
        assert g['passed'] and g['source_hashes']==hashes
        gate=dict(path=str(gate_path),sha256=sha(gate_path))
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);snap=out/'frozen_sources';snap.mkdir()
    for p in hashes:
        target=snap/Path(p).relative_to(PROJECT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write(out/'invocation.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),formal=not args.dev,
        source=str(source),scaled_source=str(scaled_source),source_hashes=hashes,input_hashes=receipt,preflight=gate,
        seeds=seeds,partitions=parts,interfaces=INTERFACES,signals=SIGNALS,directions=[0,1],updates=args.updates,
        batch_pairs=512,worlds_per_update=1024,device='cpu',threads=1,torch_version=str(torch.__version__),
        numpy_version=np.__version__,platform=platform.platform(),new_dino_inferences=0,
        new_interface_training=0,new_communication_training=0,primary='reward retained minus reset_scaled sealed greedy joint J at2400'))
    bank=v15.ImageBank();started=time.monotonic();completed=[]
    for seed,p in itertools.product(seeds,parts):
        caches,cache_receipt=make_cache(seed,p,source,scaled_source,bank,out)
        for who,interface,signal in itertools.product((0,1),INTERFACES,SIGNALS):
            fit(seed,p,who,interface,signal,caches,bank,out,args,cache_receipt,hashes)
            completed.append(f's{seed}_p{p}_d{who}_{interface}_{signal}')
        del caches
    for path,digest in {**hashes,**receipt}.items():assert sha(path)==digest,path
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    write(out/'training_complete.json',dict(status='complete',formal=not args.dev,head_fits=len(completed),runs=completed,
        updates=args.updates,head_updates=args.updates*len(completed),simulated_selected_goal_actions=args.updates*1024*len(completed),
        supervised_labels=args.updates*1024*len(completed)//2,seconds=time.monotonic()-started,
        source_hashes=hashes,input_hashes=receipt,files=files,completed_utc=datetime.now(timezone.utc).isoformat(),
        no_interface_training=True,no_communication=True))
    print(json.dumps(dict(status='complete',head_fits=len(completed),seconds=time.monotonic()-started)),flush=True)

if __name__=='__main__':main()
