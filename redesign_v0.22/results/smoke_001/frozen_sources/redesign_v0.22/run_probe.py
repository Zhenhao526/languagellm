"""Frozen no-movement probe with counterbalanced final-frame resource visibility."""
import argparse,copy,hashlib,itertools,json,shutil,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.8'))
from camp import ImageBank,remake_agents
sys.path.insert(0,str(PROJECT/'redesign_v0.20'))
import temporal_model as temporal
import temporal_world as oldworld
sys.path.insert(0,str(PROJECT/'redesign_v0.21'))
import social_model as social
SEEDS=[32101,32102,32103,32104];MASKS=('food_only','water_only');CHUNK=256

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def state_sha(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        a=np.ascontiguousarray(v.detach().numpy());h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def sources():
    paths=[ROOT/n for n in ['run_probe.py','固定执行方案.md','前置审查.md']]
    paths += [PROJECT/n for n in ['redesign_v0.20/temporal_model.py','redesign_v0.20/temporal_world.py','redesign_v0.21/social_model.py',
        'redesign_v0.8/camp.py','redesign_v0.4/run_pilot.py','redesign_v0.4/agents.py','redesign_v0.4/resource_env.py']]
    return {str(p):sha(p) for p in paths}
def locations(seed,p):
    private=PROJECT/'redesign_v0.20/results'/('smoke_002' if seed==99520 else 'temporal_001')
    communal=PROJECT/'redesign_v0.21/results'/('smoke_002' if seed==99520 else 'communication_001')
    return private,communal
def table(bank):
    photos=np.asarray(list(itertools.product(*[np.sort(bank.pools['test',k])[:4] for k in (0,1)])),np.int64)
    ids=np.repeat(np.arange(30,dtype=np.int64),16)
    return dict(map_id=ids,positions=oldworld.MAPS[ids],photo_ids=np.tile(photos,(30,1)))
def frames(projected,w,mask):
    kind=MASKS.index(mask);n=len(w['map_id']);rr=torch.arange(n);site=torch.from_numpy(w['positions'][:,kind])
    first=oldworld.full_frame(projected,w['positions'],w['photo_ids'])
    pix=projected.new_zeros(n,6,64);exists=projected.new_zeros(n,6)
    pix[rr,site]=projected[torch.from_numpy(w['photo_ids'][:,kind])];exists[rr,site]=1.
    second=torch.cat((pix.flatten(1),exists),-1);bits=projected.new_zeros(n,2);bits[:,0]=1.
    return torch.stack((first,second),1),bits

def softmax(x):
    x=np.asarray(x,np.float64);p=np.exp(x-x.max(-1,keepdims=True));return p/p.sum(-1,keepdims=True)
def statistics(raw,p,private):
    target=raw['positions'];n=len(target);correct={};actions={};q={};codes={}
    for mask in MASKS:
        if private:
            logits=raw[mask+'_logits'];actions[mask]=logits.argmax(-1);prob=softmax(logits)
            q[mask]=np.take_along_axis(prob,target[...,None],-1)[...,0].prod(-1)
        else:
            r=raw['receiver_logits'];msg=raw[mask+'_tokens'];codes[mask]=msg[:,0]*7+msg[:,1]
            actions[mask]=r.argmax(-1)[codes[mask]];rp=softmax(r)
            goodprob=np.stack([rp[:,g,target[:,g]].T for g in (0,1)],-1).prod(-1)
            q[mask]=(softmax(raw[mask+'_sender_log_probs'])*goodprob).sum(-1)
        correct[mask]=actions[mask]==target
    out={}
    for group,maps in oldworld.partition(p).items():
        ids=np.isin(raw['map_id'],maps);s={}
        for mask in MASKS:
            c=correct[mask][ids]
            s[mask]=dict(J=float(c.all(-1).mean()),food=float(c[:,0].mean()),water=float(c[:,1].mean()),Q=float(q[mask][ids].mean()))
        f,w=s['food_only'],s['water_only']
        s.update(food_visibility_effect=f['food']-w['food'],water_visibility_effect=w['water']-f['water'],
            D_visible=.5*((f['food']-w['food'])+(w['water']-f['water'])),
            visible_accuracy=.5*(f['food']+w['water']),hidden_accuracy=.5*(w['food']+f['water']),
            category_gap_food_only=f['food']-f['water'],category_gap_water_only=w['food']-w['water'],
            stable_correct_J=float((correct['food_only'].all(-1)&correct['water_only'].all(-1))[ids].mean()),
            action_change_rate=float((actions['food_only']!=actions['water_only']).any(-1)[ids].mean()),
            message_change_rate=None if private else float((codes['food_only']!=codes['water_only'])[ids].mean()),n=int(ids.sum()))
        out[group]=s
    return out

def mean_nested(values):
    x=values[0]
    if isinstance(x,dict):return {k:mean_nested([v[k] for v in values]) for k in x}
    if x is None:assert all(v is None for v in values);return None
    return float(np.mean(values))
def aggregate(rows,seeds):
    source=[];means={};systems=['private','social_immediate','social_delayed','social_all']
    for seed,system in itertools.product(seeds,systems):
        rr=[r['scores'] for r in rows if r['seed']==seed and (r['system']==system or system=='social_all' and r['system'].startswith('social_'))]
        source.append(dict(seed=seed,system=system,scores=mean_nested(rr)))
    for system in systems:means[system]=mean_nested([r['scores'] for r in source if r['system']==system])
    primary=[dict(seed=r['seed'],D_visible=r['scores']['common30']['D_visible'],
        category_gap_food_only=r['scores']['common30']['category_gap_food_only'],category_gap_water_only=r['scores']['common30']['category_gap_water_only'])
        for r in source if r['system']=='social_all']
    return dict(rows=rows,seed_rows=source,aggregate=means,primary=primary,primary_mean=means['social_all']['common30']['D_visible'])

@torch.no_grad()
def private_row(seed,p,d,bank,world,out):
    base,_=locations(seed,p);source=base/f's{seed}_p{p}_d{d}_full/final.pt'
    blob=torch.load(source,weights_only=True);prepared=torch.load(base/f'prepared_{seed}.pt',weights_only=True)
    a=remake_agents(seed,prepared,7,2,'identity')[d];a.load_state_dict(blob['agent']);a.requires_grad_(False)
    head=temporal.make_head(oldworld.init_seed(seed,p,d));head.load_state_dict(blob['head']);head.requires_grad_(False)
    before=(state_sha(a.state_dict()),state_sha(head.state_dict()));projected=a.project(bank.features).detach();raw=dict(world)
    for mask in MASKS:
        hs=[];logits=[]
        for lo in range(0,len(world['map_id']),CHUNK):
            sub={k:v[lo:lo+CHUNK] for k,v in world.items()};f,bits=frames(projected,sub,mask)
            h=temporal.observe_sequence(a,f,bits,'full');hs.append(h.numpy());logits.append(head(h).reshape(-1,2,6).numpy())
        raw[mask+'_h']=np.concatenate(hs);raw[mask+'_logits']=np.concatenate(logits)
    name=f'private_s{seed}_p{p}_d{d}.npz';np.savez_compressed(out/name,**raw)
    assert before==(state_sha(a.state_dict()),state_sha(head.state_dict()))
    assert all(x.grad is None and not x.requires_grad for x in list(a.parameters())+list(head.parameters()))
    return dict(seed=seed,partition=p,direction=d,system='private',raw=name,source=str(source),source_sha256=sha(source),
        frozen_state_sha256=before[0],frozen_head_sha256=before[1],scores=statistics(raw,p,True)),raw

@torch.no_grad()
def social_pair(seed,p,mode,bank,world,private_cache,out):
    base,communal=locations(seed,p);source=communal/f's{seed}_p{p}_{mode}/final.pt'
    states=torch.load(source,weights_only=True);prepared=torch.load(base/f'prepared_{seed}.pt',weights_only=True)
    agents=remake_agents(seed,prepared,7,2,'identity')
    for d,(a,state) in enumerate(zip(agents,states)):
        a.load_state_dict(state);a.requires_grad_(False)
        ref=torch.load(base/f's{seed}_p{p}_d{d}_full/final.pt',weights_only=True)['agent']
        assert all(torch.equal(value,ref[k]) for k,value in a.state_dict().items() if k.startswith(('project.','memory.','slot_phi.','input_transform')))
    before=[state_sha(a.state_dict()) for a in agents];rows=[]
    for d in (0,1):
        raw=dict(world);raw['receiver_logits']=social.receiver_logits(agents[1-d]).numpy()
        for mask in MASKS:
            hs=torch.from_numpy(private_cache[d][mask+'_h']);lp=[];tokens=[]
            for lo in range(0,len(hs),CHUNK):
                h=hs[lo:lo+CHUNK];lp.append(social.message_log_probs(agents[d],h).numpy());tokens.append(social.greedy_messages(agents[d],h).numpy())
            raw[mask+'_sender_log_probs']=np.concatenate(lp);raw[mask+'_tokens']=np.concatenate(tokens)
        name=f'social_{mode}_s{seed}_p{p}_d{d}.npz';np.savez_compressed(out/name,**raw)
        rows.append(dict(seed=seed,partition=p,direction=d,system='social_'+mode,raw=name,source=str(source),source_sha256=sha(source),
            frozen_sender_sha256=before[d],frozen_receiver_sha256=before[1-d],scores=statistics(raw,p,False)))
    assert before==[state_sha(a.state_dict()) for a in agents]
    assert all(x.grad is None and not x.requires_grad for a in agents for x in a.parameters())
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True,type=Path);ap.add_argument('--dev',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1);seeds=[99520] if args.dev else SEEDS;parts=[1] if args.dev else [1,2,3];hashes=sources()
    preflight=None
    if not args.dev:
        gate=ROOT/'preflight_qa.json';g=read(gate);assert g['passed'] and g['source_hashes']==hashes;preflight=dict(path=str(gate),sha256=sha(gate))
    inputs={};files=[PROJECT/'redesign_v0.4/data'/n for n in ['features.npz','manifest.json','encoder_report.json']]
    files += [PROJECT/'redesign_v0.20/results/temporal_001/completion_manifest.json',PROJECT/'redesign_v0.21/results/communication_001/completion_manifest.json']
    for seed,p in itertools.product(seeds,parts):
        base,communal=locations(seed,p);files.append(base/f'prepared_{seed}.pt')
        files += [base/f's{seed}_p{p}_d{d}_full/final.pt' for d in (0,1)]
        files += [communal/f's{seed}_p{p}_{mode}/final.pt' for mode in ('immediate','delayed')]
    inputs={str(f):sha(f) for f in files};out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    for path in hashes:
        target=out/'frozen_sources'/Path(path).relative_to(PROJECT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    write(out/'invocation.json',dict(formal=not args.dev,seeds=seeds,partitions=parts,masks=list(MASKS),chunk=CHUNK,source_hashes=hashes,input_hashes=inputs,
        preflight=preflight,started_utc=datetime.now(timezone.utc).isoformat(),torch_version=str(torch.__version__),numpy_version=np.__version__,threads=1,
        no_grad=True,all_parameter_requires_grad=False,new_training=0,new_dino_inferences=0,
        primary='common30 social_all D_visible with modes/directions/partitions nested within source; no source filtering'))
    bank=ImageBank();w=table(bank);np.savez_compressed(out/'worlds.npz',**w);started=time.monotonic();rows=[];cache={}
    # Entire private support first. This is applicability description, not a
    # rule for removing any already-trained social endpoints from evaluation.
    for seed,p,d in itertools.product(seeds,parts,(0,1)):
        row,raw=private_row(seed,p,d,bank,w,out);rows.append(row);cache[seed,p,d]=raw
        print(json.dumps(dict(completed=row['raw'],kind='private')),flush=True)
    private_scores=mean_nested([r['scores'] for r in rows]);old={mask:private_scores['old'][mask]['J'] for mask in MASKS}
    write(out/'private_applicability.json',dict(passed=all(v>=.8 for v in old.values()),threshold=.8,support='old',private_heads=len(rows),old_J=old,
        effect='Descriptive applicability only; no model filtering, new training or changed primary outcome regardless of result'))
    for seed,p,mode in itertools.product(seeds,parts,('immediate','delayed')):
        new=social_pair(seed,p,mode,bank,w,{d:cache[seed,p,d] for d in (0,1)},out);rows.extend(new)
        print(json.dumps(dict(completed=f'social_{seed}_{p}_{mode}',directions=2)),flush=True)
    write(out/'summary.json',aggregate(rows,seeds))
    for path,h in {**hashes,**inputs}.items():assert sha(path)==h,path
    outputs={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()}
    write(out/'evaluation_complete.json',dict(status='complete',formal=not args.dev,private_heads=len(seeds)*len(parts)*2,social_directions=len(seeds)*len(parts)*4,
        private_world_forwards=len(seeds)*len(parts)*2*960,social_sender_world_forwards=len(seeds)*len(parts)*4*960,new_training=0,new_dino_inferences=0,
        source_hashes=hashes,input_hashes=inputs,files=outputs,seconds=time.monotonic()-started))
    print(json.dumps(dict(status='complete',rows=len(rows),seconds=time.monotonic()-started)),flush=True)
if __name__=='__main__':main()
