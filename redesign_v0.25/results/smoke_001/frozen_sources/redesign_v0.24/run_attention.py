"""Fixed two-arm experiment; private capabilities reused, no frontend training."""
import argparse,copy,hashlib,itertools,json,platform,shutil,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.23'))
import run_experience as previous
import world,metrics
from run_experience import sha,read,write,arrays_sha,state_sha,checkpoint_times
sys.path.insert(0,str(ROOT))
import focus_model as communication
from focus_model import old
SEEDS=[33101,33102,33103,33104];ARMS=('mean','attention');CHUNK=256

def source_hashes():
    files=[ROOT/n for n in ('focus_model.py','run_attention.py','固定执行方案.md','前置科学审查.md','近邻基线方法核查.md')]
    files += [PROJECT/n for n in ('redesign_v0.23/run_experience.py','redesign_v0.23/world.py','redesign_v0.23/metrics.py','redesign_v0.20/temporal_model.py','redesign_v0.20/temporal_world.py','redesign_v0.21/social_model.py','redesign_v0.8/camp.py','redesign_v0.4/run_pilot.py','redesign_v0.4/agents.py','redesign_v0.4/resource_env.py')]
    return {str(f):sha(f) for f in files}

def seed_value(seed,p,d):return int(np.random.SeedSequence([24024,seed,p,d,0]).generate_state(1,dtype=np.uint64)[0]>>np.uint64(1))

def restore(seed,p,mode,source):
    prepared=torch.load(source/f'prepared_{seed}.pt',weights_only=True)
    agents=previous.remake_agents(seed,prepared,7,2,'identity');resets=[]
    for d,a in enumerate(agents):
        blob=torch.load(source/'private'/f's{seed}_p{p}_d{d}_all'/'final.pt',weights_only=True);a.load_state_dict(blob['agent'])
        receiver_reset=old.reset_communication(a,world.seed_value(seed,p,d,1))
        focus_reset=communication.initialize(a,mode,seed_value(seed,p,d))
        assert focus_reset['counts']==dict(sender=73191,receiver=6364)
        resets.append(dict(receiver=receiver_reset,focus=focus_reset))
    return agents,resets

@torch.no_grad()
def concepts_from_head(head,h):
    prob=head(h).reshape(-1,2,6).softmax(-1)
    roles=torch.eye(2,dtype=torch.float32)[None].expand(len(h),-1,-1)
    return torch.cat((prob,roles),-1)

@torch.no_grad()
def make_cache(seed,p,source,out):
    folder=out/'cache'/f's{seed}_p{p}_all';folder.mkdir(parents=True);cache={};inputs={}
    for d in (0,1):
        fp=source/'private'/f's{seed}_p{p}_d{d}_all'/'final.pt';inputs[str(fp)]=sha(fp)
        blob=torch.load(fp,weights_only=True);head=previous.temporal.make_head(world.seed_value(seed,p,d,0));head.load_state_dict(blob['head']);head.requires_grad_(False)
        for split in ('train','test'):
            hp=source/'cache'/f's{seed}_p{p}_all'/f'{split}_d{d}.npy';inputs[str(hp)]=sha(hp);h=np.load(hp)
            x=np.concatenate([concepts_from_head(head,torch.from_numpy(h[lo:lo+CHUNK])).numpy() for lo in range(0,len(h),CHUNK)])
            np.save(folder/f'{split}_d{d}.npy',x);cache[split,d]=torch.from_numpy(x)
    write(folder/'manifest.json',dict(seed=seed,partition=p,private_head_used=True,gold_inputs=False,chunk=CHUNK,input_hashes=inputs,
        files={f.name:sha(f) for f in folder.iterdir() if f.is_file()}))
    return cache

@torch.no_grad()
def social_eval(agents,cache,w,p,folder,step):
    scores=[]
    for d in (0,1):
        x=cache['test',d];lp=[];tok=[];aw=[]
        for lo in range(0,len(x),CHUNK):
            block=x[lo:lo+CHUNK];lp.append(communication.message_log_probs(agents[d],block).numpy())
            t,a=communication.greedy_messages(agents[d],block);tok.append(t.numpy());aw.append(a.numpy())
        raw=dict(w,sender_log_probs=np.concatenate(lp),tokens=np.concatenate(tok),effective_attention=np.concatenate(aw),receiver_logits=old.receiver_logits(agents[1-d]).numpy())
        np.savez_compressed(folder/f'protocol_{step:04d}_d{d}.npz',**raw);scores.append(metrics.social(raw,p))
    return scores

def update_social(agents,opts,cache,tables,seed,p,step):
    losses=[{},{}];traces=[];fixtures={}
    for d in (0,1):
        f=world.fixture(seed,p,d,step,'old','social',tables['train']);idx=f['indices']
        sl,rl,trace=communication.direction_loss(agents[d],agents[1-d],cache['train',d][idx],f['uniforms'],tables['train']['positions'][idx],.02 if step<2100 else 0.)
        losses[d]['sender']=sl;losses[1-d]['receiver']=rl;traces.append(trace);fixtures.update({f'd{d}__{k}':v for k,v in f.items()})
    logs=[]
    for d,(a,opt) in enumerate(zip(agents,opts)):
        opt.zero_grad(set_to_none=True);loss=(losses[d]['sender']+losses[d]['receiver'])/2;loss.backward()
        norms={k:float(torch.nn.utils.clip_grad_norm_(v,2.)) for k,v in communication.trainable_groups(a).items()}
        assert all(np.isfinite(v) for v in norms.values()) and all(v.grad is None for v in a.parameters() if not v.requires_grad)
        logs.append(dict(loss=float(loss.detach()),norms=norms))
    for opt in opts:opt.step()
    return fixtures,{f'd{d}__{k}':np.asarray(v) for d,t in enumerate(traces) for k,v in t.items()},logs

def fit_social(seed,p,arm,cache,tables,source,out,args):
    folder=out/'social'/f's{seed}_p{p}_{arm}';folder.mkdir(parents=True);agents,resets=restore(seed,p,arm,source)
    opts=[torch.optim.Adam([v for g in communication.trainable_groups(a).values() for v in g],lr=.0007) for a in agents]
    prefixes=('focus_sender.',)+tuple(n+'.' for n in old.RECEIVER_MODULES)
    frozen=lambda a:state_sha({k:v for k,v in a.state_dict().items() if not k.startswith(prefixes)})
    frozen_hashes=[frozen(a) for a in agents];save=lambda:[copy.deepcopy(a.state_dict()) for a in agents]
    torch.save(save(),folder/'initial.pt');torch.save([o.state_dict() for o in opts],folder/'initial_optimizer.pt')
    write(folder/'config.json',dict(seed=seed,partition=p,arm=arm,updates=args.updates,reset=resets,checkpoints=checkpoint_times(args.updates),
        frozen_hashes=frozen_hashes,cache=str((out/'cache'/f's{seed}_p{p}_all').resolve()),batch_per_direction=256,source=str(source),
        source_private={str(source/'private'/f's{seed}_p{p}_d{d}_all'/'final.pt'):sha(source/'private'/f's{seed}_p{p}_d{d}_all'/'final.pt') for d in (0,1)}))
    curve=[];started=time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for step in range(args.updates+1):
            if step in checkpoint_times(args.updates):
                scores=social_eval(agents,cache,tables['test'],p,folder,step);curve.append(dict(update=step,scores=scores));write(folder/'curve.json',curve)
                torch.save(save(),folder/f'checkpoint_{step:04d}.pt');torch.save([o.state_dict() for o in opts],folder/f'optimizer_{step:04d}.pt')
                print(json.dumps(dict(phase='social',run=folder.name,update=step,seconds=round(time.monotonic()-started,2))),flush=True)
            if step==args.updates:break
            f,t,l=update_social(agents,opts,cache,tables,seed,p,step)
            if step in (0,2100):
                np.savez_compressed(folder/f'train_{step+1:04d}.npz',**{f'world__{k}':v for k,v in f.items()},**{f'trace__{k}':v for k,v in t.items()})
                torch.save(save(),folder/f'after_{step+1:04d}.pt');torch.save([o.state_dict() for o in opts],folder/f'after_{step+1:04d}_optimizer.pt')
            log.write(json.dumps(dict(update=step+1,world_sha256=arrays_sha(f),trace_sha256=arrays_sha(t),people=l))+'\n')
            if (step+1)%100==0:log.flush()
    assert frozen_hashes==[frozen(a) for a in agents]
    torch.save(save(),folder/'final.pt');write(folder/'result.json',dict(status='complete',scores=curve[-1]['scores'],updates=args.updates,
        seconds=time.monotonic()-started,messages=args.updates*512,actions=args.updates*1024,frozen_verified=True))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);ap.add_argument('--dev',action='store_true');ap.add_argument('--updates',type=int,default=2400);args=ap.parse_args()
    torch.set_num_threads(1);seeds=[99523] if args.dev else SEEDS;parts=[1] if args.dev else [1,2,3]
    source=PROJECT/'redesign_v0.23/results'/('smoke_001' if args.dev else 'experience_001');hashes=source_hashes()
    marker=source/('training_complete.json' if args.dev else 'completion_manifest.json')
    inputs={str(marker):sha(marker)}
    for seed,p in itertools.product(seeds,parts):
        for d in (0,1):
            fp=source/'private'/f's{seed}_p{p}_d{d}_all'/'final.pt';inputs[str(fp)]=sha(fp)
        fp=source/f'prepared_{seed}.pt';inputs[str(fp)]=sha(fp)
        for split,d in itertools.product(('train','test'),(0,1)):
            fp=source/'cache'/f's{seed}_p{p}_all'/f'{split}_d{d}.npy';inputs[str(fp)]=sha(fp)
    for split in ('train','test'):
        fp=source/f'{split}_worlds.npz';inputs[str(fp)]=sha(fp)
    preflight=None
    if args.dev:assert args.updates==40
    else:
        g=ROOT/'preflight_qa.json';gate=read(g);assert gate['passed'] and gate['source_hashes']==hashes and args.updates==2400
        preflight=dict(path=str(g),sha256=sha(g))
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    for fp in hashes:
        target=out/'frozen_sources'/Path(fp).relative_to(PROJECT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(fp,target)
    write(out/'invocation.json',dict(formal=not args.dev,seeds=seeds,partitions=parts,updates=args.updates,source=str(source),source_hashes=hashes,input_hashes=inputs,
        preflight=preflight,torch_version=str(torch.__version__),numpy_version=np.__version__,platform=platform.platform(),threads=1,new_dino_inferences=0,new_private_updates=0,
        social_arms=list(ARMS),started_utc=datetime.now(timezone.utc).isoformat(),primary='new12 pooled sequential-greedy joint success J: attention minus mean; source-level paired average'))
    tables={split:dict(np.load(source/f'{split}_worlds.npz')) for split in ('train','test')}
    for split,w in tables.items():np.savez_compressed(out/f'{split}_worlds.npz',**w)
    started=time.monotonic()
    for seed,p in itertools.product(seeds,parts):
        cache=make_cache(seed,p,source,out)
        for arm in ARMS:fit_social(seed,p,arm,cache,tables,source,out,args)
        del cache
    for fp,h in {**hashes,**inputs}.items():assert sha(fp)==h,fp
    ns=len(seeds)*len(parts)*2
    write(out/'training_complete.json',dict(status='complete',formal=not args.dev,social_runs=ns,pair_updates=ns*args.updates,messages=ns*args.updates*512,actions=ns*args.updates*1024,
        seconds=time.monotonic()-started,source_hashes=hashes,input_hashes=inputs,new_dino_inferences=0,new_private_updates=0,
        files={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()}))
    print(json.dumps(dict(status='complete',social_runs=ns,seconds=time.monotonic()-started)),flush=True)
if __name__=='__main__':main()
