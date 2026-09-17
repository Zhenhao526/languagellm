"""One bounded new joint-readout arm, with frozen v24 mean/AT references."""
import argparse,copy,itertools,json,platform,shutil,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.24'))
import run_attention as previous
from run_attention import sha,read,write,arrays_sha,state_sha,checkpoint_times,world,metrics
sys.path.insert(0,str(ROOT));import joint_model as communication
from joint_model import old
SEEDS=previous.SEEDS;CHUNK=256

def source_hashes():
    result=previous.source_hashes()
    result.update({str(ROOT/n):sha(ROOT/n) for n in ('joint_model.py','run_joint.py','固定执行方案.md','前置科学审查.md')})
    return result

def restore(seed,p,reference):
    source=Path(read(reference/'invocation.json')['source'])
    prepared=torch.load(source/f'prepared_{seed}.pt',weights_only=True)
    agents=previous.previous.remake_agents(seed,prepared,7,2,'identity')
    origin=reference/'social'/f's{seed}_p{p}_mean'/'initial.pt';states=torch.load(origin,weights_only=True)
    for d,a in enumerate(agents):
        previous.communication.initialize(a,'mean',previous.seed_value(seed,p,d));a.load_state_dict(states[d])
        communication.initialize(a)
        assert {k:sum(v.numel() for v in g) for k,g in communication.trainable_groups(a).items()}==dict(sender=73191,receiver=6364)
    return agents

def load_cache(seed,p,reference):
    folder=reference/'cache'/f's{seed}_p{p}_all'
    return {(split,d):torch.from_numpy(np.load(folder/f'{split}_d{d}.npy')) for split,d in itertools.product(('train','test'),(0,1))}

@torch.no_grad()
def social_eval(agents,cache,w,p,folder,step):
    scores=[]
    for d in (0,1):
        x=cache['test',d];lp=[];tok=[]
        for lo in range(0,len(x),CHUNK):
            block=x[lo:lo+CHUNK];lp.append(communication.message_log_probs(agents[d],block).numpy());tok.append(communication.greedy_messages(agents[d],block).numpy())
        raw=dict(w,sender_log_probs=np.concatenate(lp),tokens=np.concatenate(tok),receiver_logits=old.receiver_logits(agents[1-d]).numpy())
        np.savez_compressed(folder/f'protocol_{step:04d}_d{d}.npz',**raw);scores.append(metrics.social(raw,p))
    return scores

def update_social(agents,opts,cache,tables,seed,p,step):
    losses=[{},{}];traces=[];fixtures={}
    for d in (0,1):
        f=world.fixture(seed,p,d,step,'old','social',tables['train']);idx=f['indices']
        sl,rl,tr=communication.direction_loss(agents[d],agents[1-d],cache['train',d][idx],f['uniforms'],tables['train']['positions'][idx],.02 if step<2100 else 0.)
        losses[d]['sender']=sl;losses[1-d]['receiver']=rl;traces.append(tr);fixtures.update({f'd{d}__{k}':v for k,v in f.items()})
    logs=[]
    for d,(a,opt) in enumerate(zip(agents,opts)):
        opt.zero_grad(set_to_none=True);loss=(losses[d]['sender']+losses[d]['receiver'])/2;loss.backward()
        contrast_norm=float(a.focus_sender.contrast.weight.grad.norm())
        norms={k:float(torch.nn.utils.clip_grad_norm_(v,2.)) for k,v in communication.trainable_groups(a).items()}
        assert all(np.isfinite(v) for v in norms.values()) and all(v.grad is None for v in a.parameters() if not v.requires_grad)
        logs.append(dict(loss=float(loss.detach()),norms=norms,contrast_gradient_norm_before_clip=contrast_norm))
    for opt in opts:opt.step()
    return fixtures,{f'd{d}__{k}':np.asarray(v) for d,t in enumerate(traces) for k,v in t.items()},logs

def fit_social(seed,p,cache,tables,reference,out,args):
    folder=out/'social'/f's{seed}_p{p}_joint';folder.mkdir(parents=True);agents=restore(seed,p,reference)
    opts=[torch.optim.Adam([v for g in communication.trainable_groups(a).values() for v in g],lr=.0007) for a in agents]
    prefixes=('focus_sender.',)+tuple(n+'.' for n in old.RECEIVER_MODULES)
    frozen=lambda a:state_sha({k:v for k,v in a.state_dict().items() if not k.startswith(prefixes)})
    frozen_hashes=[frozen(a) for a in agents];save=lambda:[copy.deepcopy(a.state_dict()) for a in agents]
    torch.save(save(),folder/'initial.pt');torch.save([o.state_dict() for o in opts],folder/'initial_optimizer.pt')
    origin=reference/'social'/f's{seed}_p{p}_mean'/'initial.pt'
    write(folder/'config.json',dict(seed=seed,partition=p,arm='joint',updates=args.updates,checkpoints=checkpoint_times(args.updates),
        frozen_hashes=frozen_hashes,cache=str((reference/'cache'/f's{seed}_p{p}_all').resolve()),batch_per_direction=256,
        reference=str(reference),source_initial=dict(path=str(origin),sha256=sha(origin)),trainable_parameters=dict(sender=73191,receiver=6364),
        contrast_initialization='all zeros',shared_initial_tensors='copied from v24 mean initial; bilinear replaced by same-size zero contrast'))
    curve=[];started=time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for step in range(args.updates+1):
            if step in checkpoint_times(args.updates):
                scores=social_eval(agents,cache,tables['test'],p,folder,step);curve.append(dict(update=step,scores=scores));write(folder/'curve.json',curve)
                torch.save(save(),folder/f'checkpoint_{step:04d}.pt');torch.save([o.state_dict() for o in opts],folder/f'optimizer_{step:04d}.pt')
                print(json.dumps(dict(run=folder.name,update=step,seconds=round(time.monotonic()-started,2))),flush=True)
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
    reference=PROJECT/'redesign_v0.24/results'/('smoke_001' if args.dev else 'attention_001');source=Path(read(reference/'invocation.json')['source']);hashes=source_hashes()
    files=[reference/n for n in ('invocation.json','analysis.json','audit_execution.json','training_complete.json','train_worlds.npz','test_worlds.npz')]
    if not args.dev:files.append(reference/'completion_manifest.json')
    for seed in seeds:files.append(source/f'prepared_{seed}.pt')
    for seed,p in itertools.product(seeds,parts):
        files.append(reference/'social'/f's{seed}_p{p}_mean'/'initial.pt')
        for split,d in itertools.product(('train','test'),(0,1)):files.append(reference/'cache'/f's{seed}_p{p}_all'/f'{split}_d{d}.npy')
    inputs={str(fp):sha(fp) for fp in files};preflight=None
    if args.dev:assert args.updates==40
    else:
        g=ROOT/'preflight_qa.json';gate=read(g);assert gate['passed'] and gate['source_hashes']==hashes and args.updates==2400
        preflight=dict(path=str(g),sha256=sha(g))
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    for fp in hashes:
        target=out/'frozen_sources'/Path(fp).relative_to(PROJECT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(fp,target)
    write(out/'invocation.json',dict(formal=not args.dev,seeds=seeds,partitions=parts,updates=args.updates,reference=str(reference),source=str(source),source_hashes=hashes,input_hashes=inputs,
        preflight=preflight,torch_version=str(torch.__version__),numpy_version=np.__version__,platform=platform.platform(),threads=1,
        new_dino_inferences=0,new_private_updates=0,new_cache_inferences=0,social_arms=['joint'],started_utc=datetime.now(timezone.utc).isoformat(),
        primary='old18 held-out-photo sequential-greedy joint success J: joint minus prior mean; four source-level paired means',
        secondary='new12 J/Q/AUC and fixed donor recombination; attention reference descriptive only'))
    tables={split:dict(np.load(reference/f'{split}_worlds.npz')) for split in ('train','test')}
    for split,w in tables.items():np.savez_compressed(out/f'{split}_worlds.npz',**w)
    started=time.monotonic()
    for seed,p in itertools.product(seeds,parts):fit_social(seed,p,load_cache(seed,p,reference),tables,reference,out,args)
    for fp,h in {**hashes,**inputs}.items():assert sha(fp)==h,fp
    ns=len(seeds)*len(parts)
    write(out/'training_complete.json',dict(status='complete',formal=not args.dev,social_runs=ns,pair_updates=ns*args.updates,messages=ns*args.updates*512,actions=ns*args.updates*1024,
        seconds=time.monotonic()-started,source_hashes=hashes,input_hashes=inputs,new_dino_inferences=0,new_private_updates=0,new_cache_inferences=0,
        reused_reference_pairs=ns*2,files={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()}))
    print(json.dumps(dict(status='complete',social_runs=ns,seconds=time.monotonic()-started)),flush=True)
if __name__=='__main__':main()
