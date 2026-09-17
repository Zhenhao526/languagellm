"""New communication under matched full versus history-dependent observations."""
import argparse,copy,itertools,json,hashlib,platform,shutil,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.8'))
from camp import ImageBank,remake_agents
sys.path.insert(0,str(PROJECT/'redesign_v0.20'))
import temporal_model
sys.path.insert(0,str(ROOT))
import social_model as model
import social_world as world
from social_metrics import metrics
SEEDS=[32101,32102,32103,32104];TIMES=[0,100,300,600,1200,1800,2100,2400];CHUNK=256

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def arrays_sha(data):
    h=hashlib.sha256()
    for k in sorted(data):
        a=np.ascontiguousarray(data[k]);h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def state_sha(state):return arrays_sha({k:v.detach().cpu().numpy() for k,v in state.items()})
def snapshot(agents):return [copy.deepcopy(a.state_dict()) for a in agents]
def frozen(a):return {k:v for k,v in a.state_dict().items() if not k.startswith(model.SENDER_MODULES+model.RECEIVER_MODULES)}
def source_paths():
    local=['run_social.py','social_world.py','social_model.py','social_metrics.py','固定执行方案.md','前置审查.md']
    older=['redesign_v0.20/temporal_model.py','redesign_v0.20/temporal_world.py','redesign_v0.8/camp.py',
           'redesign_v0.4/run_pilot.py','redesign_v0.4/agents.py','redesign_v0.4/resource_env.py']
    return [ROOT/n for n in local]+[PROJECT/n for n in older]
def source_hashes():return {str(p):sha(p) for p in source_paths()}
def original(seed,p):
    base=PROJECT/'redesign_v0.20/results'/('smoke_002' if seed==99520 else 'temporal_001')
    return base,[base/f's{seed}_p{p}_d{d}_full/final.pt' for d in (0,1)]

def restore(seed,p):
    base,paths=original(seed,p)
    prepared=torch.load(base/f'prepared_{seed}.pt',weights_only=True)
    agents=remake_agents(seed,prepared,7,2,'identity')
    init=[]
    for who,(a,path) in enumerate(zip(agents,paths)):
        blob=torch.load(path,weights_only=True);a.load_state_dict(blob['agent'])
        init.append(model.reset_communication(a,world.comm_seed(seed,p,who)))
    return agents,init

@torch.no_grad()
def encode(a,projected,table,mode):
    blocks=[]
    for lo in range(0,len(table['target_map']),CHUNK):
        sub={k:v[lo:lo+CHUNK] for k,v in table.items()}
        frame,bits=world.temporal.frames(projected,sub,'delayed' if mode=='erase' else mode)
        blocks.append(temporal_model.observe_sequence(a,frame,bits,'full',erase_history=mode=='erase').numpy())
    return np.concatenate(blocks)

def caches(seed,p,bank,out):
    folder=out/f'cache_s{seed}_p{p}';folder.mkdir()
    agents,_=restore(seed,p);tables={'train':world.train_worlds(bank,p),'test':world.temporal.evaluation_worlds(bank,p)}
    cache={};base,paths=original(seed,p)
    for split,table in tables.items():np.savez_compressed(folder/f'{split}_worlds.npz',**table)
    for who,a in enumerate(agents):
        with torch.no_grad():projected=a.project(bank.features).detach()
        np.save(folder/f'projected_d{who}.npy',projected.numpy())
        for split,table in tables.items():
            for mode in (('immediate','delayed') if split=='train' else ('immediate','delayed','erase')):
                h=encode(a,projected,table,mode);np.save(folder/f'{split}_d{who}_{mode}.npy',h)
                cache[split,who,mode]=torch.from_numpy(h)
    write(folder/'cache_manifest.json',dict(seed=seed,partition=p,chunk=CHUNK,
        source_states={str(f):sha(f) for f in paths},source_prepared={str(base/f'prepared_{seed}.pt'):sha(base/f'prepared_{seed}.pt')},
        train_rows=len(tables['train']['target_map']),test_rows=len(tables['test']['target_map']),
        private_heads_used=False,new_dino_inferences=0,
        files={f.name:sha(f) for f in sorted(folder.iterdir()) if f.is_file()}))
    return tables,cache

@torch.no_grad()
def evaluate(agents,cache,table,p,mode,out,step,label):
    results=[]
    for d in (0,1):
        h=cache['test',d,mode];lp=[];token=[]
        for lo in range(0,len(h),CHUNK):
            sub=h[lo:lo+CHUNK];lp.append(model.message_log_probs(agents[d],sub).numpy())
            token.append(model.greedy_messages(agents[d],sub).numpy())
        lp=np.concatenate(lp);token=np.concatenate(token);rlogits=model.receiver_logits(agents[1-d]).numpy()
        np.savez_compressed(out/f'protocol_{step:04d}_{label}_d{d}.npz',**table,sender_log_probs=lp,receiver_logits=rlogits,tokens=token)
        results.append(metrics(lp,rlogits,token,table,p))
    return results

def perform_update(agents,opts,cache,table,seed,p,step,mode):
    losses=[{},{}];details=[];world_data={}
    weight=.02 if step<2100 else 0.
    for d in (0,1):
        indices,u=world.fixture(seed,p,d,step,len(table['target_map']))
        h=cache['train',d,mode][indices]
        sl,rl,trace=model.direction_loss(agents[d],agents[1-d],h,u,table['positions'][indices],weight)
        losses[d]['sender']=sl;losses[1-d]['receiver']=rl;details.append(trace)
        world_data[f'd{d}_indices']=indices;world_data[f'd{d}_uniforms']=u
    logs=[]
    for who,(a,opt) in enumerate(zip(agents,opts)):
        opt.zero_grad(set_to_none=True);loss=(losses[who]['sender']+losses[who]['receiver'])/2
        loss.backward();norms={role:float(torch.nn.utils.clip_grad_norm_(params,2.)) for role,params in model.trainable_groups(a).items()}
        assert all(np.isfinite(v) for v in norms.values())
        assert all(param.grad is None for param in a.parameters() if not param.requires_grad)
        logs.append(dict(loss=float(loss.detach()),norms=norms))
    for opt in opts:opt.step()
    return world_data,details,logs

def fit(seed,p,mode,tables,cache,out,args):
    folder=out/f's{seed}_p{p}_{mode}';folder.mkdir();agents,init=restore(seed,p)
    opts=[torch.optim.Adam([q for group in model.trainable_groups(a).values() for q in group],lr=.0007) for a in agents]
    frozen_hashes=[state_sha(frozen(a)) for a in agents];times=sorted({0,args.updates,*[t for t in TIMES if t<args.updates]})
    torch.save(snapshot(agents),folder/'initial.pt');torch.save([o.state_dict() for o in opts],folder/'initial_optimizer.pt')
    write(folder/'config.json',dict(seed=seed,partition=p,mode=mode,updates=args.updates,batch_per_direction=256,
        init=init,checkpoints=times,frozen_hashes=frozen_hashes,initial_hashes=[state_sha(a.state_dict()) for a in agents],
        source_states={str(f):sha(f) for f in original(seed,p)[1]},cache=str((out/f'cache_s{seed}_p{p}').resolve()),
        reward='.25*(c0+c1)+.5*c0*c1',constant_baseline=.5,learning_rate=.0007,clip_each_role=2.))
    curve=[];started=time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for step in range(args.updates+1):
            if step in times:
                scores=evaluate(agents,cache,tables['test'],p,mode,folder,step,'normal')
                torch.save(snapshot(agents),folder/f'checkpoint_{step:04d}.pt')
                torch.save([o.state_dict() for o in opts],folder/f'optimizer_{step:04d}.pt')
                curve.append(dict(update=step,scores=scores));write(folder/'curve.json',curve)
                print(json.dumps(dict(run=folder.name,update=step,seconds=round(time.monotonic()-started,2))),flush=True)
            if step==args.updates:break
            w,details,logs=perform_update(agents,opts,cache,tables['train'],seed,p,step,mode)
            trace={f'd{d}__{k}':np.asarray(v) for d,detail in enumerate(details) for k,v in detail.items()}
            if step in (0,2100):
                np.savez_compressed(folder/f'train_{step+1:04d}.npz',**w,**trace)
                torch.save(snapshot(agents),folder/f'after_{step+1:04d}.pt')
                torch.save([o.state_dict() for o in opts],folder/f'after_{step+1:04d}_optimizer.pt')
            log.write(json.dumps(dict(update=step+1,world_sha256=arrays_sha(w),trace_sha256=arrays_sha(trace),
                people=logs,reward_by_direction=[float(t['reward'].mean()) for t in details]))+'\n')
            if (step+1)%100==0:log.flush()
    assert [state_sha(frozen(a)) for a in agents]==frozen_hashes
    final_extra={label:evaluate(agents,cache,tables['test'],p,alt,folder,args.updates,label)
        for label,alt in [('cross_view','delayed' if mode=='immediate' else 'immediate'),('erase','erase')]}
    torch.save(snapshot(agents),folder/'final.pt');torch.save([o.state_dict() for o in opts],folder/'final_optimizer.pt')
    write(folder/'result.json',dict(status='complete',seed=seed,partition=p,mode=mode,scores=curve[-1]['scores'],extra=final_extra,
        updates=args.updates,messages=args.updates*512,actions=args.updates*1024,seconds=time.monotonic()-started,frozen_verified=True))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--dev',action='store_true');ap.add_argument('--updates',type=int,default=2400);args=ap.parse_args()
    assert args.updates>0;torch.set_num_threads(1);hashes=source_hashes()
    seeds=[99520] if args.dev else SEEDS;parts=[1] if args.dev else [1,2,3]
    preflight=None
    if not args.dev:
        g=ROOT/'preflight_qa.json';data=read(g);assert args.updates==2400 and data['passed'] and data['source_hashes']==hashes
        preflight=dict(path=str(g),sha256=sha(g))
    inputs={str(PROJECT/'redesign_v0.20/results/temporal_001/completion_manifest.json'):sha(PROJECT/'redesign_v0.20/results/temporal_001/completion_manifest.json')}
    for seed,p in itertools.product(seeds,parts):
        base,paths=original(seed,p)
        for f in paths+[base/f'prepared_{seed}.pt']:inputs[str(f)]=sha(f)
    for n in ['features.npz','manifest.json','encoder_report.json']:
        f=PROJECT/'redesign_v0.4/data'/n;inputs[str(f)]=sha(f)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    for f in source_paths():
        target=out/'frozen_sources'/f.relative_to(PROJECT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
    write(out/'invocation.json',dict(formal=not args.dev,started_utc=datetime.now(timezone.utc).isoformat(),seeds=seeds,partitions=parts,
        modes=['immediate','delayed'],updates=args.updates,source_hashes=hashes,input_hashes=inputs,preflight=preflight,
        torch_version=str(torch.__version__),numpy_version=np.__version__,platform=platform.platform(),threads=1,
        source_choice='All v20 full endpoints; failed v20 full-detach gate remains failed',new_dino_inferences=0,
        primary='common30 normal J at2400: delayed minus immediate; 4 outer source seeds'))
    bank=ImageBank();started=time.monotonic();completed=[]
    for seed,p in itertools.product(seeds,parts):
        tables,cache=caches(seed,p,bank,out)
        for mode in ('immediate','delayed'):
            fit(seed,p,mode,tables,cache,out,args);completed.append(f's{seed}_p{p}_{mode}')
        del cache
    for path,h in {**hashes,**inputs}.items():assert sha(path)==h,path
    files={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()}
    write(out/'training_complete.json',dict(status='complete',formal=not args.dev,runs=completed,social_runs=len(completed),
        pair_updates=args.updates*len(completed),messages=args.updates*len(completed)*512,actions=args.updates*len(completed)*1024,
        seconds=time.monotonic()-started,source_hashes=hashes,input_hashes=inputs,files=files,new_dino_inferences=0,new_private_training=0))
    print(json.dumps(dict(status='complete',social_runs=len(completed),seconds=time.monotonic()-started)),flush=True)
if __name__=='__main__':main()
