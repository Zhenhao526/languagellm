"""Fixed-budget private temporal preparation, before any new social training."""
from __future__ import annotations
import argparse,copy,hashlib,itertools,json,platform,shutil,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.8'))
from camp import ImageBank,make_agents,individual_practice,remake_agents
sys.path.insert(0,str(ROOT))
import temporal_model as model
import temporal_world as world

SEEDS=[32101,32102,32103,32104]
TIMES=[0,100,300,600,1200,1800,2100,2400]
PREFIXES=('memory.','slot_phi.')
CHUNK=256

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def arrays_sha(x):
    h=hashlib.sha256()
    for k in sorted(x):
        a=np.ascontiguousarray(x[k]);h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def state_sha(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def snapshot(agent,head):return dict(agent=copy.deepcopy(agent.state_dict()),head=copy.deepcopy(head.state_dict()))
def sources():
    files=[ROOT/x for x in ('run_temporal.py','temporal_model.py','temporal_world.py','固定执行方案.md','环境与能力验收_独立审查.md')]
    files += [PROJECT/x for x in ('redesign_v0.8/camp.py','redesign_v0.4/run_pilot.py','redesign_v0.4/agents.py','redesign_v0.4/resource_env.py')]
    return {str(p):sha(p) for p in files}
def inputs():
    paths=[PROJECT/'redesign_v0.4/data'/x for x in ('features.npz','manifest.json','encoder_report.json')]
    enc=read(paths[-1]);assert sha(paths[0])==enc['features_sha256'] and sha(paths[1])==enc['manifest_sha256']
    paths.append(PROJECT/'redesign_v0.19/results/readout_001/completion_manifest.json')
    return {str(p):sha(p) for p in paths}

def prepare(seed,bank,out):
    agents=make_agents(seed)
    evidence=individual_practice(agents,bank,seed,updates=200,batch=64)
    states=[copy.deepcopy(a.state_dict()) for a in agents]
    torch.save(states,out/f'prepared_{seed}.pt')
    write(out/f'preparation_{seed}.json',dict(seed=seed,updates_per_person=200,batch=64,
        evidence=evidence,no_source_filter=True,new_projection_preparation=True,no_spatial_map_labels=True))
    return states

@torch.no_grad()
def evaluate(agent,head,projected,table,p,arm,folder,step,erase=False):
    modes=['delayed_erase'] if erase else ['immediate','delayed'];scores={}
    for mode in modes:
        blocks=[]
        for lo in range(0,len(table['target_map']),CHUNK):
            subset={k:v[lo:lo+CHUNK] for k,v in table.items()}
            frames,bits=world.frames(projected,subset,'delayed' if erase else mode)
            h=model.observe_sequence(agent,frames,bits,arm,erase_history=erase)
            blocks.append(head(h).reshape(-1,2,6).numpy())
        logits=np.concatenate(blocks)
        assert np.isfinite(logits).all()
        np.savez_compressed(folder/f'evaluation_{step:04d}_{mode}.npz',**table,logits=logits)
        scores[mode]=world.metrics(logits,table,p)
    return scores

def fit(seed,p,who,arm,prepared,bank,out,args,hashes):
    folder=out/f's{seed}_p{p}_d{who}_{arm}';folder.mkdir()
    agent=remake_agents(seed,prepared,7,2,'identity')[who]
    head=model.make_head(world.init_seed(seed,p,who))
    for key,parameter in agent.named_parameters():parameter.requires_grad_(key.startswith(PREFIXES))
    params=[parameter for parameter in agent.parameters() if parameter.requires_grad]+list(head.parameters())
    assert sum(x.numel() for x in params)==155598
    opt=torch.optim.Adam(params,lr=.0007)
    frozen={k:v for k,v in agent.state_dict().items() if not k.startswith(PREFIXES)};frozen_sha=state_sha(frozen)
    with torch.no_grad():projected=agent.project(bank.features).detach()
    np.save(folder/'projected.npy',projected.numpy())
    table=world.evaluation_worlds(bank,p)
    times=sorted({0,args.updates,*[t for t in TIMES if t<args.updates]})
    initial=snapshot(agent,head);torch.save(initial,folder/'initial.pt');torch.save(opt.state_dict(),folder/'initial_optimizer.pt')
    write(folder/'config.json',dict(seed=seed,partition=p,direction=who,arm=arm,updates=args.updates,
        batch_events=256,action_rows=512,eval_chunk=CHUNK,checkpoints=times,head_seed=world.init_seed(seed,p,who),
        trainable_agent_names=[k for k,v in agent.named_parameters() if v.requires_grad],trainable_parameters=155598,
        initial_agent_sha256=state_sha(initial['agent']),initial_head_sha256=state_sha(initial['head']),frozen_sha256=frozen_sha,
        source_hashes=hashes,prepared_sha256=sha(out/f'prepared_{seed}.pt'),projected_sha256=sha(folder/'projected.npy'),
        optimizer='Adam',learning_rate=.0007,gradient_clip=2.,entropy_on_updates=2100,entropy_weight=.02,
        group_maps={k:v.tolist() for k,v in world.partition(p).items()},only_private_preparation=True,no_communication=True))
    curve=[];started=time.monotonic()
    with (folder/'training.jsonl').open('w') as log:
        for step in range(args.updates+1):
            if step in times:
                scores=evaluate(agent,head,projected,table,p,arm,folder,step)
                torch.save(snapshot(agent,head),folder/f'checkpoint_{step:04d}.pt')
                torch.save(opt.state_dict(),folder/f'optimizer_{step:04d}.pt')
                curve.append(dict(update=step,scores=scores,agent_sha256=state_sha(agent.state_dict()),head_sha256=state_sha(head.state_dict())))
                write(folder/'curve.json',curve)
                print(json.dumps(dict(run=folder.name,update=step,seconds=round(time.monotonic()-started,2))),flush=True)
            if step==args.updates:break
            w=world.fixture(bank,seed,p,who,step)
            frames,bits,goals,positions=world.training_inputs(projected,w)
            opt.zero_grad(set_to_none=True)
            loss,components,trace=model.loss_terms(agent,head,frames,bits,goals,positions,w['action_uniform'],arm,step)
            loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(params,2.));assert np.isfinite(norm)
            assert all(v.grad is None for v in agent.parameters() if not v.requires_grad)
            opt.step()
            if step in (0,2100):
                data={'world__'+k:np.asarray(v) for k,v in w.items()}
                data.update({'trace__'+k:np.asarray(v) for k,v in trace.items()})
                data.update(frames=frames.numpy(),view_bits=bits.numpy(),goals=goals,positions=positions)
                np.savez_compressed(folder/f'train_{step+1:04d}.npz',**data)
                torch.save(snapshot(agent,head),folder/f'after_{step+1:04d}.pt')
                torch.save(opt.state_dict(),folder/f'after_{step+1:04d}_optimizer.pt')
            log.write(json.dumps(dict(update=step+1,world_sha256=arrays_sha(w),trace_sha256=arrays_sha(trace),
                components=components,gradient_norm=norm,clip_coefficient=min(1.,2./(norm+1e-6))))+'\n')
            if (step+1)%100==0:log.flush()
    assert state_sha({k:v for k,v in agent.state_dict().items() if not k.startswith(PREFIXES)})==frozen_sha
    erased=evaluate(agent,head,projected,table,p,arm,folder,args.updates,erase=True)['delayed_erase']
    torch.save(snapshot(agent,head),folder/'final.pt');torch.save(opt.state_dict(),folder/'final_optimizer.pt')
    write(folder/'result.json',dict(status='complete',seed=seed,partition=p,direction=who,arm=arm,updates=args.updates,
        seconds=time.monotonic()-started,scores=curve[-1]['scores'],erase_scores=erased,
        selected_goal_actions=args.updates*512,no_communication=True,frozen_verified=True))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--dev',action='store_true');ap.add_argument('--updates',type=int,default=2400)
    args=ap.parse_args();assert args.updates>0;torch.set_num_threads(1)
    seeds=[99520] if args.dev else SEEDS;parts=[1] if args.dev else [1,2,3]
    hashes=sources();input_hashes=inputs();preflight=None
    if not args.dev:
        assert args.updates==2400
        gate=ROOT/'preflight_qa.json';g=read(gate);assert g['passed'] and g['source_hashes']==hashes
        preflight=dict(path=str(gate),sha256=sha(gate))
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);snap=out/'frozen_sources';snap.mkdir()
    for p in hashes:
        target=snap/Path(p).relative_to(PROJECT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write(out/'invocation.json',dict(formal=not args.dev,started_utc=datetime.now(timezone.utc).isoformat(),
        seeds=seeds,partitions=parts,directions=[0,1],arms=['full','detach'],updates=args.updates,
        source_hashes=hashes,input_hashes=input_hashes,preflight=preflight,torch_version=str(torch.__version__),
        numpy_version=np.__version__,platform=platform.platform(),device='cpu',threads=1,
        new_dino_inferences=0,new_social_training=0,primary='common30 delayed J full minus detach at2400',
        capability_gate_support='old only; descriptive wholebatch; no source selection'))
    bank=ImageBank();started=time.monotonic();completed=[]
    for seed in seeds:
        prepared=prepare(seed,bank,out)
        for p,who,arm in itertools.product(parts,(0,1),('full','detach')):
            fit(seed,p,who,arm,prepared,bank,out,args,hashes);completed.append(f's{seed}_p{p}_d{who}_{arm}')
    for p,h in {**hashes,**input_hashes}.items():assert sha(p)==h,p
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    write(out/'training_complete.json',dict(status='complete',formal=not args.dev,head_fits=len(completed),runs=completed,
        updates=args.updates,private_updates=args.updates*len(completed),selected_goal_actions=args.updates*len(completed)*512,
        initial_resource_preparation_updates=len(seeds)*2*200,initial_resource_preparation_actions=len(seeds)*2*200*64,
        seconds=time.monotonic()-started,source_hashes=hashes,input_hashes=input_hashes,files=files,
        completed_utc=datetime.now(timezone.utc).isoformat(),new_social_training=0,new_dino_inferences=0))
    print(json.dumps(dict(status='complete',head_fits=len(completed),seconds=time.monotonic()-started)),flush=True)

if __name__=='__main__':main()
