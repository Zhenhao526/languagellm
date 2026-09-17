"""Train and audit recovery of a staged dual-token protocol."""
from __future__ import annotations
import argparse, hashlib, json, platform, shutil, time
from pathlib import Path
import numpy as np
from . import design, environment, policy
from research_program.action_dependent_signaling_study import design as base_design, policy as base_policy

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_bytes(x): return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
def finite(x):
    if isinstance(x,dict): return all(finite(v) for v in x.values())
    if isinstance(x,list): return all(finite(v) for v in x)
    if isinstance(x,(float,int,np.number)): return bool(np.isfinite(float(x)))
    return True

def source_hashes():
    rels=('__init__.py','design.py','policy.py','environment.py','runner.py','aggregate.py','audit.py','plan.md','tests/test_game.py')
    return {str((HERE/x).relative_to(ROOT)):sha(HERE/x) for x in rels}
def dependency_hashes():
    rels=('research_program/action_dependent_signaling_study/design.py','research_program/action_dependent_signaling_study/policy.py','research_program/action_dependent_signaling_study/environment.py')
    return {x:sha(ROOT/x) for x in rels}

def parent_path(parent_root,seed): return Path(parent_root)/f'seed_{seed}_{design.PARENT_CONDITION}'/f'checkpoint_{design.CHECKPOINTS[-1]:04d}.npz'
def parent_hashes(parent_root):
    out={}
    for seed in design.SEEDS:
        p=parent_path(parent_root,seed); design.require(p.is_file(),f'missing parent checkpoint {p}'); out[str(seed)]=sha(p)
    return out

def prepare(out,parent_root):
    out=Path(out).resolve(); design.require(not out.exists(),'refuse overwrite'); parent_root=Path(parent_root).resolve(); ph=parent_hashes(parent_root)
    cfg=design.prepare(str(parent_root),ph); cfg.update({'runs':len(design.SEEDS)*len(design.CONDITIONS),'parent_runs':len(design.SEEDS),'child_runs':len(design.SEEDS)*len(design.CONDITIONS),'evaluation_episodes_per_worker':4096})
    src=source_hashes(); deps=dependency_hashes(); out.mkdir(parents=True)
    for rel in src:
        q=out/'source_snapshot'/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/rel,q)
    for rel in deps:
        q=out/'dependency_snapshot'/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/rel,q)
    (out/'prepared.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    plan={'schema':'generation_compositional_transmission_v1','prepared_sha256':sha(out/'prepared.json'),'sources':src,'dependencies':deps,'runtime':{'python':platform.python_version(),'numpy':np.__version__},'config':cfg}
    (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n'); (out/'freeze.json').write_text(json.dumps({'plan_sha256':sha(out/'plan.json'),'prepared_sha256':sha(out/'prepared.json')},indent=2)+'\n'); verify(out)
    return {'status':'prepared','out':str(out),'plan_sha256':sha(out/'plan.json'),'prepared_sha256':sha(out/'prepared.json')}

def verify(out):
    out=Path(out); plan=json.loads((out/'plan.json').read_text()); cfg=json.loads((out/'prepared.json').read_text()); fr=json.loads((out/'freeze.json').read_text())
    design.require(sha(out/'plan.json')==fr['plan_sha256'] and sha(out/'prepared.json')==fr['prepared_sha256']==plan['prepared_sha256'],'freeze hash mismatch')
    design.require(plan['config']==cfg and plan['sources']==source_hashes() and plan['dependencies']==dependency_hashes(),'source changed')
    for rel,d in plan['sources'].items(): design.require(sha(out/'source_snapshot'/rel)==d,'snapshot changed '+rel)
    for rel,d in plan['dependencies'].items(): design.require(sha(out/'dependency_snapshot'/rel)==d,'dependency snapshot changed '+rel)
    design.require(parent_hashes(cfg['parent_root'])==cfg['parent_checkpoint_sha256'],'parent checkpoint changed')
    return plan,cfg

def entropy_grad(prob):
    lp=np.log(np.maximum(prob,1e-300)); ent=-(prob*lp).sum(axis=-1,keepdims=True); return -prob*(lp+ent)
def center(values,keys):
    values=np.asarray(values,dtype=np.float64); keys=np.asarray(keys); out=np.zeros_like(values)
    for key in np.unique(keys):
        idx=np.flatnonzero(keys==key)
        if len(idx)>1: out[idx]=values[idx]-(values[idx].sum()-values[idx])/(len(idx)-1)
    return out
def future_returns(rewards): return np.flip(np.cumsum(np.flip(rewards,axis=1),axis=1),axis=1)
def arrival_times(): return base_design.message_arrival_times('staged','dual2')

def rollout(p,ep,channel,role,*,sample,mode='natural',message_override=None):
    actual='silent' if channel=='silent' else 'live'
    filt=design.TARGET_WORKER if role=='worker' else None
    return environment.rollout(p,ep,'dual2','rotating','hidden','factorized',actual,'staged',message_mode=mode,sample=sample,partner_filter=filt,message_override=message_override)

def child_gradient(p,ep,tr,role,channel,beta):
    g={k:np.zeros_like(v) for k,v in p.items()}; future=future_returns(tr['rewards'])
    if role=='worker':
        idx=np.flatnonzero(tr['active'])
        for t in range(design.ACTION_START,design.HORIZON):
            if len(idx)==0: break
            ap=tr['action_probs'][t][idx]; st=tr['state'][t][idx]; local=tr['local'][t][idx]; inv=tr['inventory'][t][idx]
            adv=center(future[idx,t],st*1000+t*100+local*10+inv); oa=np.zeros_like(ap); oa[np.arange(len(idx)),tr['actions'][idx,t]]=1
            d=-(adv[:,None]*(oa-ap))/len(idx)-beta*entropy_grad(ap)/len(idx)
            np.add.at(g['worker_logits'],(np.zeros(len(idx),dtype=np.int64),st,np.full(len(idx),t),local,inv),d)
    elif channel!='silent':
        adv_base=future[:,design.ACTION_START:].sum(axis=1); context=base_design.goal_index(ep['goal'])
        arr=arrival_times()
        for slot in range(2):
            sp=tr['message_probs'][slot]; one=np.zeros_like(sp); one[np.arange(len(ep['goal'])),tr['selected_messages'][:,slot]]=1
            adv=center(future[:,arr[slot]+1:].sum(axis=1),context); d=-(adv[:,None]*(one-sp))/len(ep['goal'])-beta*entropy_grad(sp)/len(ep['goal'])
            np.add.at(g['sender_logits_hidden'][:,slot,:],context,d)
    return g

def save(path,p,update):
    np.savez_compressed(path,update=np.array(update,dtype=np.int64),sender_logits_hidden=p['sender_logits_hidden'],sender_logits_visible=p['sender_logits_visible'],worker_logits=p['worker_logits']); return sha(path)

def sequence(p,goal):
    idx=int(goal[0])*2+int(goal[1]); return np.array([base_policy.softmax(p['sender_logits_hidden'][idx,slot]).argmax() for slot in range(2)],dtype=np.int64)

def recombination_override(p,ep):
    out=np.zeros((len(ep['goal']),2),dtype=np.int64); donors=np.asarray([[0,0],[0,1],[1,0],[1,1]],dtype=np.int8)
    for i,g in enumerate(ep['goal']):
        target=np.asarray(g,dtype=np.int8); d0=donors[np.flatnonzero(donors[:,0]==target[0])[0]]; d1=donors[np.flatnonzero(donors[:,1]==target[1])[0]]
        out[i,0]=sequence(p,d0)[0]; out[i,1]=sequence(p,d1)[1]
    return out

def metrics(p,ep,role,channel,mode,*,override=None):
    tr=rollout(p,ep,channel,role,sample=False,mode=mode,message_override=override); mask=tr['active']; r=tr['team_return'][mask];
    return {'episodes':int(mask.sum()),'team_return_mean':float(r.mean()) if len(r) else None,'team_return_sd':float(r.std()) if len(r) else None,'positive_episode_rate':float((r>0).mean()) if len(r) else None,'action_histogram':np.bincount(tr['actions'][mask].ravel(),minlength=design.ACTION_COUNT).tolist() if len(r) else []}

def semantic(p,role):
    out=[]
    for goal in ((0,0),(0,1),(1,0),(1,1)):
        hits=[]
        for local in ((0,0),(0,1),(1,0),(1,1)):
            ep=design.episode_stream(880000+goal[0]*10+goal[1],1); ep['goal'][:]=goal; ep['target_bits'][:]=base_design.target_bits(ep['goal'],'factorized'); ep['partner_id'][:]=0
            tr=rollout(p,ep,'live',role,sample=False); hit=0
            for t in range(design.ACTION_START,design.HORIZON):
                stage=(t-design.ACTION_START)//design.STEPS_PER_SUBTASK; act=int(tr['actions'][0,t]); site=act-1
                hit+=int(act>0 and int(ep['site_type'][0,stage,site])==int(ep['target_bits'][0,stage]))
            hits.append(hit/(design.SUBTASKS*design.STEPS_PER_SUBTASK))
        out.append(float(np.mean(hits)))
    return {'semantic_success_by_goal':out,'semantic_success_mean':float(np.mean(out)),'sender_sequences':[sequence(p,g).tolist() for g in ((0,0),(0,1),(1,0),(1,1))]}

def evaluate(p,seed,role,*,evaluation):
    ep=design.episode_stream(seed,4096*design.WORKERS,evaluation=evaluation); modes={}
    for mode in ('natural','closed','permuted'):
        modes[mode]=metrics(p,ep,role,'live',mode)
    modes['recombined']=metrics(p,ep,role,'live','natural',override=recombination_override(p,ep))
    return {'split':'heldout' if evaluation else 'training_support','modes':modes,'codebook':semantic(p,role)}

def train_one(seed,condition,execution,parent_checkpoint,updates=None):
    role,channel=design.parse_condition(condition); updates=design.UPDATES if updates is None else int(updates); run=Path(execution)/f'seed_{seed}_{condition}'; run.mkdir(parents=True,exist_ok=False)
    parent=policy.load(parent_checkpoint); p=policy.clone(parent); replacement=policy.make_replacement(seed)
    if role=='worker': p['worker_logits'][design.TARGET_WORKER]=replacement['worker_logits'][0]
    else: p['sender_logits_hidden']=replacement['sender_logits_hidden']; p['sender_logits_visible']=replacement['sender_logits_visible']
    parent_hash=policy.combined_hash(parent); init_hash=policy.combined_hash(p); checkpoints=sorted(set([u for u in design.CHECKPOINTS if u<=updates]+[updates]));
    if 0 in checkpoints: save(run/'checkpoint_0000.npz',p,0)
    rows=[]; start=time.perf_counter()
    for update in range(1,updates+1):
        ep=design.episode_stream(seed,design.BATCH_SIZE,update=update); tr=rollout(p,ep,channel,role,sample=True,mode='permuted' if channel=='permuted' else 'natural'); beta=design.entropy_coefficient(update); g=child_gradient(p,ep,tr,role,channel,beta)
        norm=float(np.sqrt(sum(float((x*x).sum()) for x in g.values()))); scale=min(1.0,5.0/max(norm,1e-12))
        if role=='worker': p['worker_logits'][design.TARGET_WORKER]-=design.LEARNING_RATE*scale*g['worker_logits'][design.TARGET_WORKER]
        elif channel!='silent': p['sender_logits_hidden']-=design.LEARNING_RATE*scale*g['sender_logits_hidden']; p['sender_logits_visible']-=design.LEARNING_RATE*scale*g['sender_logits_visible']
        rows.append({'update':update,'seed':seed,'condition':condition,'role':role,'channel':channel,'world_sha256':design.array_sha(ep['site_type']),'goal_sha256':design.array_sha(ep['goal']),'partner_sha256':design.array_sha(ep['partner_id']),'message_uniform_sha256':design.array_sha(ep['message_uniforms']),'action_uniform_sha256':design.array_sha(ep['action_uniforms']),'return_mean':float(tr['team_return'][tr['active']].mean()) if tr['active'].any() else 0.0,'active_count':int(tr['active'].sum()),'gradient_norm':norm,'gradient_clip_scale':scale,'parameter_sha256':policy.combined_hash(p),'elapsed_seconds':time.perf_counter()-start})
        if update in checkpoints and update>0: rows[-1]['checkpoint_sha256']=save(run/f'checkpoint_{update:04d}.npz',p,update)
    result={'seed':seed,'condition':condition,'role':role,'channel':channel,'parent_checkpoint':str(parent_checkpoint),'parent_checkpoint_sha256':sha(parent_checkpoint),'parent_parameter_sha256':parent_hash,'initial_parameter_sha256':init_hash,'updates':updates,'final':{'training_support':evaluate(p,seed,role,evaluation=False),'heldout':evaluate(p,seed,role,evaluation=True)},'checkpoints':checkpoints,'trajectory':rows}
    log=run/'training.jsonl'; log.write_bytes(b''.join(json_bytes(x) for x in rows)); result['training_log_sha256']=sha(log); result['final_parameter_sha256']=policy.combined_hash(p); (run/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n'); return result

def run_grid(prepared,out,updates=None,seeds=None,conditions=None):
    plan,cfg=verify(prepared); out=Path(out); execution=out/'execution'; execution.mkdir(parents=True); seeds=tuple(design.SEEDS if seeds is None else seeds); conditions=tuple(design.CONDITIONS if conditions is None else conditions); design.require(set(seeds)<=set(design.SEEDS) and set(conditions)<=set(design.CONDITIONS),'invalid subset'); allr=[]
    for seed in seeds:
        for c in conditions:
            allr.append(train_one(seed,c,execution,parent_path(cfg['parent_root'],seed),updates)); (execution/'progress.json').write_text(json.dumps({'completed':len(allr),'total':len(seeds)*len(conditions),'seeds':list(seeds),'conditions':list(conditions)},indent=2))
    (execution/'results.json').write_text(json.dumps({'results':allr},ensure_ascii=False,indent=2)+'\n'); return allr

if __name__=='__main__':
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True); a=sub.add_parser('prepare'); a.add_argument('--out',required=True); a.add_argument('--parent-root',required=True); a=sub.add_parser('execute'); a.add_argument('--out',required=True); a.add_argument('--prepared',required=True); a.add_argument('--updates',type=int,default=None); a.add_argument('--seeds'); a.add_argument('--conditions'); args=ap.parse_args()
    if args.cmd=='prepare': print(json.dumps(prepare(args.out,args.parent_root),ensure_ascii=False))
    else:
        seeds=None if args.seeds is None else tuple(int(x) for x in args.seeds.split(',') if x); conditions=None if args.conditions is None else tuple(x for x in args.conditions.split(',') if x); print(json.dumps({'status':'completed','runs':len(run_grid(args.prepared,args.out,args.updates,seeds,conditions))},ensure_ascii=False))
