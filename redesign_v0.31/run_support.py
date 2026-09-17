"""Train a four-agent population under fixed or rotating partner schedules."""
import argparse,copy,hashlib,itertools,json,platform,shutil,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.8'));from camp import remake_agents
sys.path.insert(0,str(PROJECT/'redesign_v0.21'));import social_model as communication
sys.path.insert(0,str(ROOT));import support,metrics

SEEDS=[34101,34102,34103,34104];PARTITIONS=[1,2,3];AGENTS=4;PRIVATE_TYPES=(0,1,0,1);CHUNK=256
CHECKPOINTS=(0,100,600,1200,2100,2400)

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays_sha(x):
    h=hashlib.sha256()
    for k,v in sorted(x.items()):
        a=np.ascontiguousarray(v);h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def state_sha(x):return arrays_sha({k:v.detach().numpy() for k,v in x.items()})
def checkpoint_times(updates):return sorted({0,updates,*[t for t in CHECKPOINTS if t<updates]})
def source_hashes():
    files=[ROOT/n for n in ('run_support.py','support.py','metrics.py','code_relabelings.npy','support_design.json','固定执行方案.md')]
    files += [PROJECT/n for n in ('redesign_v0.28/world.py','redesign_v0.20/temporal_model.py','redesign_v0.20/temporal_world.py','redesign_v0.21/social_model.py','redesign_v0.8/camp.py','redesign_v0.4/run_pilot.py','redesign_v0.4/agents.py','redesign_v0.4/resource_env.py')]
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

def restore_population(seed,p,prepared,source):
    templates=remake_agents(seed,prepared,7,2,'identity');agents=[];resets=[]
    for i,d in enumerate(PRIVATE_TYPES):
        a=copy.deepcopy(templates[d])
        blob=torch.load(source/'private'/f's{seed}_p{p}_d{d}_all'/'final.pt',weights_only=True);a.load_state_dict(blob['agent'])
        resets.append(communication.reset_communication(a,support.seed_value(seed,p,i,1)));agents.append(a)
    return agents,resets

@torch.no_grad()
def social_eval_population(agents,cache,w,p,condition,folder,step,endpoint):
    scores={};pair_ids=[(i,j) for i in range(AGENTS) for j in range(AGENTS) if i!=j and PRIVATE_TYPES[i]!=PRIVATE_TYPES[j]]
    for i,j in pair_ids:
        h=cache['test',PRIVATE_TYPES[i]]
        raw=dict(w,sender_log_probs=communication.message_log_probs(agents[i],h).numpy(),tokens=communication.greedy_messages(agents[i],h).numpy(),receiver_logits=communication.receiver_logits(agents[j]).numpy())
        key=f'i{i}_j{j}';np.savez_compressed(folder/f'protocol_{step:04d}_{key}.npz',**raw);scores[key]=metrics.social(raw,p,condition)
        if step==endpoint:metrics.save_null(raw,p,condition,folder/f'recombination_null_{key}.npz')
    return scores

def update_population(agents,opts,cache,table,seed,p,step,condition):
    losses=[{'sender':[],'receiver':[]} for _ in agents];fixtures={};traces={}
    for pair_no,(i,j) in enumerate(support.matching(condition,step)):
        for sender,receiver in ((i,j),(j,i)):
            d=PRIVATE_TYPES[sender];f=support.fixture(seed,p,d,step,condition,table['train']);idx=f['indices']
            sl,rl,trace=communication.direction_loss(agents[sender],agents[receiver],cache['train',d][idx],f['uniforms'],table['train']['positions'][idx],.02 if step<2100 else 0.)
            losses[sender]['sender'].append(sl);losses[receiver]['receiver'].append(rl)
            fixtures.update({f'a{sender}__{k}':v for k,v in f.items()});traces.update({f'a{sender}__{k}':np.asarray(v) for k,v in trace.items()})
    logs=[]
    for i,(a,opt) in enumerate(zip(agents,opts)):
        loss=(sum(losses[i]['sender'])+sum(losses[i]['receiver']))/2
        opt.zero_grad(set_to_none=True);loss.backward()
        norms={k:float(torch.nn.utils.clip_grad_norm_(v,2.)) for k,v in communication.trainable_groups(a).items()}
        assert all(np.isfinite(v) for v in norms.values()) and all(v.grad is None for v in a.parameters() if not v.requires_grad)
        logs.append(dict(loss=float(loss.detach()),norms=norms))
    for opt in opts:opt.step()
    return fixtures,traces,logs

def fit_population(seed,p,condition,prepared,cache,table,source,out,args):
    folder=out/'social'/f's{seed}_p{p}_{condition}';folder.mkdir(parents=True)
    agents,resets=restore_population(seed,p,prepared,source)
    opts=[torch.optim.Adam([v for g in communication.trainable_groups(a).values() for v in g],lr=.0007) for a in agents]
    prefixes=communication.SENDER_MODULES+communication.RECEIVER_MODULES
    frozen=lambda a:state_sha({k:v for k,v in a.state_dict().items() if not k.startswith(prefixes)})
    frozen_hashes=[frozen(a) for a in agents];save=lambda:[copy.deepcopy(a.state_dict()) for a in agents]
    torch.save(save(),folder/'initial.pt');torch.save([o.state_dict() for o in opts],folder/'initial_optimizer.pt')
    write(folder/'config.json',dict(seed=seed,partition=p,condition=condition,updates=args.updates,checkpoints=checkpoint_times(args.updates),reset=resets,
        frozen_hashes=frozen_hashes,cache=str((source/'cache'/f's{seed}_p{p}_all').resolve()),source=str(source),population_agents=AGENTS,private_types=list(PRIVATE_TYPES),matching_schedule=support._DESIGN['population'],
        groups={k:v.tolist() for k,v in support.groups(p,condition).items()},learning_rate=.0007,clip=2.,messages_per_population_update=960,actions_per_population_update=1920,
        source_private={str(source/'private'/f's{seed}_p{p}_d{d}_all'/'final.pt'):sha(source/'private'/f's{seed}_p{p}_d{d}_all'/'final.pt') for d in (0,1)}))
    curve=[];started=time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for step in range(args.updates+1):
            if step in checkpoint_times(args.updates):
                scores=social_eval_population(agents,cache,table['test'],p,condition,folder,step,args.updates);curve.append(dict(update=step,scores=scores));write(folder/'curve.json',curve)
                torch.save(save(),folder/f'checkpoint_{step:04d}.pt');torch.save([o.state_dict() for o in opts],folder/f'optimizer_{step:04d}.pt')
                print(json.dumps(dict(phase='social',run=folder.name,update=step,seconds=round(time.monotonic()-started,2))),flush=True)
            if step==args.updates:break
            f,t,l=update_population(agents,opts,cache,table,seed,p,step,condition)
            if step in (0,2100,args.updates-1):
                np.savez_compressed(folder/f'train_{step+1:04d}.npz',**{f'world__{k}':v for k,v in f.items()},**{f'trace__{k}':v for k,v in t.items()});torch.save(save(),folder/f'after_{step+1:04d}.pt');torch.save([o.state_dict() for o in opts],folder/f'after_{step+1:04d}_optimizer.pt')
            log.write(json.dumps(dict(update=step+1,world_sha256=arrays_sha(f),trace_sha256=arrays_sha(t),people=l))+'\n')
            if (step+1)%100==0:log.flush()
    assert frozen_hashes==[frozen(a) for a in agents]
    torch.save(save(),folder/'final.pt');write(folder/'result.json',dict(status='complete',scores=curve[-1]['scores'],updates=args.updates,seconds=time.monotonic()-started,messages=args.updates*960,actions=args.updates*1920,frozen_verified=True))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);ap.add_argument('--dev',action='store_true');ap.add_argument('--updates',type=int,default=2400);args=ap.parse_args()
    torch.set_num_threads(1);seeds=[99528] if args.dev else SEEDS;parts=[1] if args.dev else PARTITIONS;conditions=list(support.CONDITIONS);source=PROJECT/'redesign_v0.28/results'/('smoke_001' if args.dev else 'formation_001')
    hashes=source_hashes();inputs=input_hashes(source,seeds,parts);preflight=None
    if args.dev:assert args.updates==40
    else:
        gate=read(ROOT/'preflight_qa.json');assert gate['passed'] and gate['source_hashes']==hashes and gate['input_hashes']==inputs and args.updates==2400;preflight=dict(path=str(ROOT/'preflight_qa.json'),sha256=sha(ROOT/'preflight_qa.json'))
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    for file in hashes:
        target=out/'frozen_sources'/Path(file).relative_to(PROJECT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
    write(out/'invocation.json',dict(formal=not args.dev,seeds=seeds,partitions=parts,updates=args.updates,source=str(source),source_hashes=hashes,input_hashes=inputs,preflight=preflight,torch_version=str(torch.__version__),numpy_version=np.__version__,platform=platform.platform(),threads=1,new_dino_inferences=0,social_conditions=conditions,population_agents=AGENTS,private_types=list(PRIVATE_TYPES),messages_per_population_update=960,actions_per_population_update=1920,primary='population mean target12 J at2400 rotating_partners minus fixed_partners; source/panel/direction/pair nested',study_scope='partner_schedule_multi_agent_multi_partner_target',new_private_fits=0,test_worlds=180,train_worlds=720))
    tables={split:dict(np.load(source/f'{split}_worlds.npz')) for split in ('train','test')}
    for split,w in tables.items():np.savez_compressed(out/f'{split}_worlds.npz',**w)
    started=time.monotonic()
    for seed,p in itertools.product(seeds,parts):
        prepared=torch.load(source/f'prepared_{seed}.pt',weights_only=True);cache={(split,d):torch.from_numpy(np.load(source/'cache'/f's{seed}_p{p}_all'/f'{split}_d{d}.npy')) for split,d in itertools.product(('train','test'),(0,1))}
        assert all(not h.requires_grad and torch.isfinite(h).all() and h.shape==({'train':720,'test':180}[split],96) for (split,d),h in cache.items())
        for condition in conditions:fit_population(seed,p,condition,prepared,cache,tables,source,out,args)
    for file,h in {**hashes,**inputs}.items():assert sha(file)==h,file
    ns=len(seeds)*len(parts)*len(conditions);write(out/'training_complete.json',dict(status='complete',formal=not args.dev,social_runs=ns,pair_updates=ns*args.updates,messages=ns*args.updates*960,actions=ns*args.updates*1920,seconds=time.monotonic()-started,source_hashes=hashes,input_hashes=inputs,new_private_fits=0,new_dino_inferences=0,files={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()}));print(json.dumps(dict(status='complete',social_runs=ns,seconds=time.monotonic()-started)),flush=True)

if __name__=='__main__':main()
