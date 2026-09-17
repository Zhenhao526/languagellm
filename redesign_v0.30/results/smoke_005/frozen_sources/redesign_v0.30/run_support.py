"""Fresh communication under a fixed multi-partner target and matched supports."""
import argparse,copy,hashlib,itertools,json,platform,shutil,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.8'))
from camp import remake_agents
sys.path.insert(0,str(PROJECT/'redesign_v0.21'))
import social_model as communication
sys.path.insert(0,str(ROOT))
import support,metrics
SEEDS=[34101,34102,34103,34104];TIMES=[0,100,600,1200,2100,2400];CHUNK=256


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def arrays_sha(x):
    h=hashlib.sha256()
    for k,v in sorted(x.items()):
        a=np.ascontiguousarray(v);h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def state_sha(x):return arrays_sha({k:v.detach().numpy() for k,v in x.items()})
def checkpoint_times(updates):return sorted({0,updates,*[t for t in TIMES if t<updates]})
def source_hashes():
    files=[ROOT/n for n in ('run_support.py','support.py','metrics.py','code_relabelings.npy','support_design.json','固定执行方案.md')]
    files += [PROJECT/n for n in ('redesign_v0.28/world.py',
        'redesign_v0.20/temporal_model.py','redesign_v0.20/temporal_world.py','redesign_v0.21/social_model.py',
        'redesign_v0.8/camp.py','redesign_v0.4/run_pilot.py','redesign_v0.4/agents.py','redesign_v0.4/resource_env.py')]
    return {str(f):sha(f) for f in files}

def input_hashes(source,seeds,parts):
    files=[source/n for n in ('training_complete.json','train_worlds.npz','test_worlds.npz','private_applicability.json')]
    files += [PROJECT/'redesign_v0.28/data'/n for n in ('feature_cache.pt','selection.json','encoder_receipt.json')]
    for seed in seeds:
        files.append(source/f'prepared_{seed}.pt')
        for p in parts:
            files.append(source/'cache'/f's{seed}_p{p}_all'/'manifest.json')
            for d in (0,1):
                files += [source/'cache'/f's{seed}_p{p}_all'/f'{split}_d{d}.npy' for split in ('train','test')]
                files += [source/'private'/f's{seed}_p{p}_d{d}_all'/n for n in ('final.pt','result.json')]
    return {str(f):sha(f) for f in files}


def restore(seed,p,prepared,source):
    agents=remake_agents(seed,prepared,7,2,'identity');resets=[]
    for d,a in enumerate(agents):
        blob=torch.load(source/'private'/f's{seed}_p{p}_d{d}_all'/'final.pt',weights_only=True);a.load_state_dict(blob['agent'])
        resets.append(communication.reset_communication(a,support.seed_value(seed,p,d,1)))
    return agents,resets


@torch.no_grad()
def social_eval(agents,cache,w,p,arm,folder,step,endpoint):
    scores=[]
    for d in (0,1):
        h=cache['test',d]
        raw=dict(w,sender_log_probs=communication.message_log_probs(agents[d],h).numpy(),
            tokens=communication.greedy_messages(agents[d],h).numpy(),receiver_logits=communication.receiver_logits(agents[1-d]).numpy())
        np.savez_compressed(folder/f'protocol_{step:04d}_d{d}.npz',**raw);scores.append(metrics.social(raw,p,arm))
        if step==endpoint:metrics.save_null(raw,p,arm,folder/f'recombination_null_d{d}.npz')
    return scores


def update_social(agents,opts,cache,tables,seed,p,step,arm):
    losses=[{},{}];traces=[];fixtures={}
    for d in (0,1):
        f=support.fixture(seed,p,d,step,arm,tables['train']);idx=f['indices']
        sl,rl,trace=communication.direction_loss(agents[d],agents[1-d],cache['train',d][idx],f['uniforms'],tables['train']['positions'][idx],.02 if step<2100 else 0.)
        losses[d]['sender']=sl;losses[1-d]['receiver']=rl;traces.append(trace)
        fixtures.update({f'd{d}__{k}':v for k,v in f.items()})
    logs=[]
    for d,(a,opt) in enumerate(zip(agents,opts)):
        opt.zero_grad(set_to_none=True);loss=(losses[d]['sender']+losses[d]['receiver'])/2;loss.backward()
        norms={k:float(torch.nn.utils.clip_grad_norm_(v,2.)) for k,v in communication.trainable_groups(a).items()}
        assert all(np.isfinite(v) for v in norms.values()) and all(v.grad is None for v in a.parameters() if not v.requires_grad)
        logs.append(dict(loss=float(loss.detach()),norms=norms))
    for opt in opts:opt.step()
    return fixtures,{f'd{d}__{k}':np.asarray(v) for d,t in enumerate(traces) for k,v in t.items()},logs


def fit_social(seed,p,arm,prepared,cache,tables,source,out,args):
    folder=out/'social'/f's{seed}_p{p}_{arm}';folder.mkdir(parents=True)
    agents,resets=restore(seed,p,prepared,source)
    opts=[torch.optim.Adam([v for g in communication.trainable_groups(a).values() for v in g],lr=.0007) for a in agents]
    prefixes=communication.SENDER_MODULES+communication.RECEIVER_MODULES
    frozen=lambda a:state_sha({k:v for k,v in a.state_dict().items() if not k.startswith(prefixes)})
    frozen_hashes=[frozen(a) for a in agents]
    save=lambda:[copy.deepcopy(a.state_dict()) for a in agents]
    torch.save(save(),folder/'initial.pt');torch.save([o.state_dict() for o in opts],folder/'initial_optimizer.pt')
    write(folder/'config.json',dict(seed=seed,partition=p,arm=arm,updates=args.updates,reset=resets,checkpoints=checkpoint_times(args.updates),
        frozen_hashes=frozen_hashes,cache=str((source/'cache'/f's{seed}_p{p}_all').resolve()),source=str(source),batch_per_direction=240,
        groups={k:v.tolist() for k,v in support.groups(p,arm).items()},learning_rate=.0007,clip=2.,
        source_private={str(source/'private'/f's{seed}_p{p}_d{d}_all'/'final.pt'):sha(source/'private'/f's{seed}_p{p}_d{d}_all'/'final.pt') for d in (0,1)}))
    curve=[];started=time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for step in range(args.updates+1):
            if step in checkpoint_times(args.updates):
                scores=social_eval(agents,cache,tables['test'],p,arm,folder,step,args.updates)
                curve.append(dict(update=step,scores=scores));write(folder/'curve.json',curve)
                torch.save(save(),folder/f'checkpoint_{step:04d}.pt');torch.save([o.state_dict() for o in opts],folder/f'optimizer_{step:04d}.pt')
                print(json.dumps(dict(phase='social',run=folder.name,update=step,seconds=round(time.monotonic()-started,2))),flush=True)
            if step==args.updates:break
            f,t,l=update_social(agents,opts,cache,tables,seed,p,step,arm)
            if step in (0,2100,args.updates-1):
                np.savez_compressed(folder/f'train_{step+1:04d}.npz',**{f'world__{k}':v for k,v in f.items()},**{f'trace__{k}':v for k,v in t.items()})
                torch.save(save(),folder/f'after_{step+1:04d}.pt');torch.save([o.state_dict() for o in opts],folder/f'after_{step+1:04d}_optimizer.pt')
            log.write(json.dumps(dict(update=step+1,world_sha256=arrays_sha(f),trace_sha256=arrays_sha(t),people=l))+'\n')
            if (step+1)%100==0:log.flush()
    assert frozen_hashes==[frozen(a) for a in agents]
    torch.save(save(),folder/'final.pt');write(folder/'result.json',dict(status='complete',scores=curve[-1]['scores'],updates=args.updates,
        seconds=time.monotonic()-started,messages=args.updates*480,actions=args.updates*960,frozen_verified=True))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);ap.add_argument('--dev',action='store_true');ap.add_argument('--updates',type=int,default=2400);args=ap.parse_args()
    torch.set_num_threads(1);seeds=[99528] if args.dev else SEEDS;parts=[1] if args.dev else [1,2,3]
    source=PROJECT/'redesign_v0.28/results'/('smoke_001' if args.dev else 'formation_001')
    hashes=source_hashes();inputs=input_hashes(source,seeds,parts);preflight=None
    if not args.dev:
        gate=read(ROOT/'preflight_qa.json');assert gate['passed'] and gate['source_hashes']==hashes and gate['input_hashes']==inputs and args.updates==2400
        preflight=dict(path=str(ROOT/'preflight_qa.json'),sha256=sha(ROOT/'preflight_qa.json'))
    else:assert args.updates==40
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    for file in hashes:
        target=out/'frozen_sources'/Path(file).relative_to(PROJECT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
    write(out/'invocation.json',dict(formal=not args.dev,seeds=seeds,partitions=parts,updates=args.updates,source=str(source),source_hashes=hashes,input_hashes=inputs,
        preflight=preflight,torch_version=str(torch.__version__),numpy_version=np.__version__,platform=platform.platform(),threads=1,new_dino_inferences=0,
        social_arms=list(support.ARMS),started_utc=datetime.now(timezone.utc).isoformat(),batch_per_direction=240,
        primary='target12 pooled J at2400 aligned_paths1 minus sparse_paths; nested source / panels / directions / masks',
        study_scope='fixed_budget_multi_partner_target_support_contrast',new_private_fits=0,test_worlds=180,train_worlds=720))
    tables={split:dict(np.load(source/f'{split}_worlds.npz')) for split in ('train','test')}
    for split,w in tables.items():np.savez_compressed(out/f'{split}_worlds.npz',**w)
    started=time.monotonic()
    for seed,p in itertools.product(seeds,parts):
        prepared=torch.load(source/f'prepared_{seed}.pt',weights_only=True)
        cache={(split,d):torch.from_numpy(np.load(source/'cache'/f's{seed}_p{p}_all'/f'{split}_d{d}.npy')) for split,d in itertools.product(('train','test'),(0,1))}
        assert all(not h.requires_grad and torch.isfinite(h).all() and h.shape==({'train':720,'test':180}[split],96) for (split,d),h in cache.items())
        for arm in support.ARMS:fit_social(seed,p,arm,prepared,cache,tables,source,out,args)
    for file,h in {**hashes,**inputs}.items():assert sha(file)==h,file
    ns=len(seeds)*len(parts)*2
    write(out/'training_complete.json',dict(status='complete',formal=not args.dev,social_runs=ns,pair_updates=ns*args.updates,
        messages=ns*args.updates*480,actions=ns*args.updates*960,seconds=time.monotonic()-started,source_hashes=hashes,input_hashes=inputs,
        new_private_fits=0,new_dino_inferences=0,files={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()}))
    print(json.dumps(dict(status='complete',social_runs=ns,seconds=time.monotonic()-started)),flush=True)
if __name__=='__main__':main()
