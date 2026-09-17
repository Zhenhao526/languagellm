"""Matched private state exposure followed by newly learned discrete communication."""
import argparse,copy,hashlib,itertools,json,platform,shutil,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.8'))
from camp import ImageBank,make_agents,individual_practice,remake_agents
sys.path.insert(0,str(PROJECT/'redesign_v0.20'))
import temporal_model as temporal
sys.path.insert(0,str(PROJECT/'redesign_v0.21'))
import social_model as communication
sys.path.insert(0,str(ROOT))
import world,metrics
SEEDS=[33101,33102,33103,33104];TIMES=[0,100,600,1200,2100,2400];CHUNK=256
PRIVATE_PREFIXES=('memory.','slot_phi.');ARMS=('old_old','all_old','all_all')


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def arrays_sha(x):
    h=hashlib.sha256()
    for k,v in sorted(x.items()):
        a=np.ascontiguousarray(v);h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def state_sha(x):return arrays_sha({k:v.detach().numpy() for k,v in x.items()})
def snap(a,head):return dict(agent=copy.deepcopy(a.state_dict()),head=copy.deepcopy(head.state_dict()))
def source_hashes():
    files=[ROOT/n for n in ('run_experience.py','world.py','metrics.py','固定执行方案.md','前置科学审查.md')]
    files += [PROJECT/n for n in ('redesign_v0.20/temporal_model.py','redesign_v0.20/temporal_world.py','redesign_v0.21/social_model.py','redesign_v0.8/camp.py','redesign_v0.4/run_pilot.py','redesign_v0.4/agents.py','redesign_v0.4/resource_env.py')]
    return {str(f):sha(f) for f in files}
def checkpoint_times(updates):return sorted({0,updates,*[t for t in TIMES if t<updates]})


def prepare(seed,bank,out):
    agents=make_agents(seed);evidence=individual_practice(agents,bank,seed,updates=200,batch=64)
    states=[copy.deepcopy(a.state_dict()) for a in agents];torch.save(states,out/f'prepared_{seed}.pt')
    write(out/f'preparation_{seed}.json',dict(seed=seed,updates_per_person=200,batch=64,evidence=evidence,source_filter=False))
    return states


@torch.no_grad()
def encode(a,projected,w):
    hs=[]
    for lo in range(0,len(w['map_id']),CHUNK):
        f,b=world.frames(projected,world.subset(w,slice(lo,lo+CHUNK)))
        hs.append(temporal.observe_sequence(a,f,b,'full').numpy())
    return np.concatenate(hs)


@torch.no_grad()
def private_eval(a,head,projected,w,p,folder,step):
    h=encode(a,projected,w);logits=np.concatenate([head(torch.from_numpy(h[lo:lo+CHUNK])).reshape(-1,2,6).numpy() for lo in range(0,len(h),CHUNK)])
    raw=dict(w,h=h,logits=logits);np.savez_compressed(folder/f'evaluation_{step:04d}.npz',**raw)
    return metrics.private(raw,p)


def fit_private(seed,p,d,scope,prepared,bank,tables,out,args):
    folder=out/'private'/f's{seed}_p{p}_d{d}_{scope}';folder.mkdir(parents=True)
    a=remake_agents(seed,prepared,7,2,'identity')[d];head=temporal.make_head(world.seed_value(seed,p,d,0))
    for k,v in a.named_parameters():v.requires_grad_(k.startswith(PRIVATE_PREFIXES))
    params=[v for v in a.parameters() if v.requires_grad]+list(head.parameters());assert sum(v.numel() for v in params)==155598
    opt=torch.optim.Adam(params,lr=.0007);frozen=state_sha({k:v for k,v in a.state_dict().items() if not k.startswith(PRIVATE_PREFIXES)})
    with torch.no_grad():projected=a.project(bank.features).detach()
    torch.save(snap(a,head),folder/'initial.pt');torch.save(opt.state_dict(),folder/'initial_optimizer.pt')
    write(folder/'config.json',dict(seed=seed,partition=p,direction=d,scope=scope,updates=args.updates,initial_head_seed=world.seed_value(seed,p,d,0),
        trainable_parameters=155598,checkpoints=checkpoint_times(args.updates),frozen_sha256=frozen,prepared_sha256=sha(out/f'prepared_{seed}.pt'),
        trainable_names=[k for k,v in a.named_parameters() if v.requires_grad],new_communication=False,learning_rate=.0007,clip=2.))
    curve=[];started=time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for step in range(args.updates+1):
            if step in checkpoint_times(args.updates):
                scores=private_eval(a,head,projected,tables['test'],p,folder,step)
                torch.save(snap(a,head),folder/f'checkpoint_{step:04d}.pt');torch.save(opt.state_dict(),folder/f'optimizer_{step:04d}.pt')
                curve.append(dict(update=step,scores=scores));write(folder/'curve.json',curve)
                print(json.dumps(dict(phase='private',run=folder.name,update=step,seconds=round(time.monotonic()-started,2))),flush=True)
            if step==args.updates:break
            fixture=world.fixture(seed,p,d,step,scope,'private',tables['train']);w=world.subset(tables['train'],fixture['indices']);frames,bits=world.frames(projected,w)
            opt.zero_grad(set_to_none=True)
            loss,components,trace=temporal.loss_terms(a,head,frames,bits,fixture['goals'],w['positions'],fixture['uniforms'],'full',step)
            loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(params,2.));assert np.isfinite(norm)
            assert all(v.grad is None for v in a.parameters() if not v.requires_grad);opt.step()
            if step in (0,2100):
                np.savez_compressed(folder/f'train_{step+1:04d}.npz',**{f'world__{k}':v for k,v in fixture.items()},**{f'trace__{k}':np.asarray(v) for k,v in trace.items()})
                torch.save(snap(a,head),folder/f'after_{step+1:04d}.pt');torch.save(opt.state_dict(),folder/f'after_{step+1:04d}_optimizer.pt')
            log.write(json.dumps(dict(update=step+1,world_sha256=arrays_sha(fixture),trace_sha256=arrays_sha(trace),components=components,gradient_norm=norm))+'\n')
            if (step+1)%100==0:log.flush()
    assert frozen==state_sha({k:v for k,v in a.state_dict().items() if not k.startswith(PRIVATE_PREFIXES)})
    torch.save(snap(a,head),folder/'final.pt');write(folder/'result.json',dict(status='complete',scores=curve[-1]['scores'],updates=args.updates,
        seconds=time.monotonic()-started,selected_goal_actions=args.updates*512,frozen_verified=True))


def restore(seed,p,scope,prepared,out):
    agents=remake_agents(seed,prepared,7,2,'identity');resets=[]
    for d,a in enumerate(agents):
        blob=torch.load(out/'private'/f's{seed}_p{p}_d{d}_{scope}'/'final.pt',weights_only=True);a.load_state_dict(blob['agent'])
        resets.append(communication.reset_communication(a,world.seed_value(seed,p,d,1)))
    return agents,resets


def make_cache(seed,p,scope,prepared,bank,tables,out):
    folder=out/'cache'/f's{seed}_p{p}_{scope}';folder.mkdir(parents=True)
    agents,_=restore(seed,p,scope,prepared,out);cache={}
    for d,a in enumerate(agents):
        with torch.no_grad():projected=a.project(bank.features).detach()
        for split,w in tables.items():
            h=encode(a,projected,w);np.save(folder/f'{split}_d{d}.npy',h);cache[split,d]=torch.from_numpy(h)
    write(folder/'manifest.json',dict(seed=seed,partition=p,scope=scope,private_head_used=False,chunk=CHUNK,
        files={f.name:sha(f) for f in folder.iterdir() if f.is_file()}))
    return cache


@torch.no_grad()
def social_eval(agents,cache,w,p,folder,step):
    scores=[]
    for d in (0,1):
        h=cache['test',d];lp=[];tok=[]
        for lo in range(0,len(h),CHUNK):
            lp.append(communication.message_log_probs(agents[d],h[lo:lo+CHUNK]).numpy());tok.append(communication.greedy_messages(agents[d],h[lo:lo+CHUNK]).numpy())
        raw=dict(w,sender_log_probs=np.concatenate(lp),tokens=np.concatenate(tok),receiver_logits=communication.receiver_logits(agents[1-d]).numpy())
        np.savez_compressed(folder/f'protocol_{step:04d}_d{d}.npz',**raw);scores.append(metrics.social(raw,p))
    return scores


def update_social(agents,opts,cache,tables,seed,p,step,scope):
    losses=[{},{}];traces=[];fixtures={}
    for d in (0,1):
        f=world.fixture(seed,p,d,step,scope,'social',tables['train']);idx=f['indices']
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


def fit_social(seed,p,arm,prepared,cache,tables,out,args):
    private_scope,scope=arm.split('_');folder=out/'social'/f's{seed}_p{p}_{arm}';folder.mkdir(parents=True)
    agents,resets=restore(seed,p,private_scope,prepared,out)
    opts=[torch.optim.Adam([v for g in communication.trainable_groups(a).values() for v in g],lr=.0007) for a in agents]
    prefixes=communication.SENDER_MODULES+communication.RECEIVER_MODULES
    frozen=lambda a:state_sha({k:v for k,v in a.state_dict().items() if not k.startswith(prefixes)})
    frozen_hashes=[frozen(a) for a in agents]
    save=lambda:[copy.deepcopy(a.state_dict()) for a in agents]
    torch.save(save(),folder/'initial.pt');torch.save([o.state_dict() for o in opts],folder/'initial_optimizer.pt')
    write(folder/'config.json',dict(seed=seed,partition=p,arm=arm,updates=args.updates,reset=resets,checkpoints=checkpoint_times(args.updates),
        frozen_hashes=frozen_hashes,cache=str((out/'cache'/f's{seed}_p{p}_{private_scope}').resolve()),batch_per_direction=256,
        source_private={str(out/'private'/f's{seed}_p{p}_d{d}_{private_scope}'/'final.pt'):sha(out/'private'/f's{seed}_p{p}_d{d}_{private_scope}'/'final.pt') for d in (0,1)}))
    curve=[];started=time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for step in range(args.updates+1):
            if step in checkpoint_times(args.updates):
                scores=social_eval(agents,cache,tables['test'],p,folder,step);curve.append(dict(update=step,scores=scores));write(folder/'curve.json',curve)
                torch.save(save(),folder/f'checkpoint_{step:04d}.pt');torch.save([o.state_dict() for o in opts],folder/f'optimizer_{step:04d}.pt')
                print(json.dumps(dict(phase='social',run=folder.name,update=step,seconds=round(time.monotonic()-started,2))),flush=True)
            if step==args.updates:break
            f,t,l=update_social(agents,opts,cache,tables,seed,p,step,scope)
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
    assert args.updates>0;torch.set_num_threads(1);seeds=[99523] if args.dev else SEEDS;parts=[1] if args.dev else [1,2,3]
    hashes=source_hashes();inputs={str(PROJECT/'redesign_v0.4/data'/n):sha(PROJECT/'redesign_v0.4/data'/n) for n in ('features.npz','manifest.json','encoder_report.json')}
    old=PROJECT/'redesign_v0.22/results/visibility_001/completion_manifest.json';inputs[str(old)]=sha(old)
    preflight=None
    if not args.dev:
        g=ROOT/'preflight_qa.json';gate=read(g);assert gate['passed'] and gate['source_hashes']==hashes and args.updates==2400
        preflight=dict(path=str(g),sha256=sha(g))
    else:assert args.updates==40
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    for p in hashes:
        target=out/'frozen_sources'/Path(p).relative_to(PROJECT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write(out/'invocation.json',dict(formal=not args.dev,seeds=seeds,partitions=parts,updates=args.updates,source_hashes=hashes,input_hashes=inputs,
        preflight=preflight,torch_version=str(torch.__version__),numpy_version=np.__version__,platform=platform.platform(),threads=1,new_dino_inferences=0,
        private_scopes=['old','all'],social_arms=list(ARMS),started_utc=datetime.now(timezone.utc).isoformat(),
        primary='new12 pooled J at2400 all_old minus old_old; source nested partitions/directions/masks'))
    bank=ImageBank();tables={split:world.table(bank,split) for split in ('train','test')}
    for split,w in tables.items():np.savez_compressed(out/f'{split}_worlds.npz',**w)
    started=time.monotonic();prepared={}
    # Complete private matrix and descriptive capability check before all social fits.
    for seed in seeds:
        prepared[seed]=prepare(seed,bank,out)
        for p,d,scope in itertools.product(parts,(0,1),('old','all')):fit_private(seed,p,d,scope,prepared[seed],bank,tables,out,args)
    capability=[]
    for seed in seeds:
        mask_J={mask:float(np.mean([read(out/'private'/f's{seed}_p{p}_d{d}_all'/'result.json')['scores']['new12'][mask]['J'] for p,d in itertools.product(parts,(0,1))])) for mask in ('food_only','water_only')}
        capability.append(dict(seed=seed,mask_J=mask_J,passed=all(v>=.8 for v in mask_J.values())))
    write(out/'private_applicability.json',dict(passed=all(r['passed'] for r in capability),source_rows=capability,threshold=.8,filter_applied=False,social_matrix_proceeds_regardless=True))
    print(json.dumps(dict(private_phase_complete=True,capability=capability)),flush=True)
    for seed,p in itertools.product(seeds,parts):
        for private_scope in ('old','all'):
            cache=make_cache(seed,p,private_scope,prepared[seed],bank,tables,out)
            for arm in (('old_old',) if private_scope=='old' else ('all_old','all_all')):fit_social(seed,p,arm,prepared[seed],cache,tables,out,args)
            del cache
    for p,h in {**hashes,**inputs}.items():assert sha(p)==h,p
    npv=len(seeds)*len(parts)*4;ns=len(seeds)*len(parts)*3
    write(out/'training_complete.json',dict(status='complete',formal=not args.dev,private_fits=npv,social_runs=ns,private_updates=npv*args.updates,
        private_actions=npv*args.updates*512,pair_updates=ns*args.updates,messages=ns*args.updates*512,actions=ns*args.updates*1024,
        initial_preparation_updates=len(seeds)*400,initial_preparation_actions=len(seeds)*400*64,seconds=time.monotonic()-started,
        source_hashes=hashes,input_hashes=inputs,new_dino_inferences=0,files={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()}))
    print(json.dumps(dict(status='complete',private_fits=npv,social_runs=ns,seconds=time.monotonic()-started)),flush=True)
if __name__=='__main__':main()
