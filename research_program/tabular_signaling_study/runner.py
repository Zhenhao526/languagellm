"""Train, evaluate and freeze the low-variance tabular signaling control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np

from . import design, policy, environment

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CHECKPOINTS=(0,500,1000,2000,3000)


def require(ok,msg):
    if not ok: raise ValueError(msg)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(x): return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
def json_bytes(x): return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()


def make_prepared():
    d=design.prepare(); d.update({'checkpoints':list(CHECKPOINTS),'runtime_policy':'float64 NumPy only',
        'runs':len(design.SEEDS)*len(design.CONDITIONS),'episode_batches_per_run':design.UPDATES,
        'training_episodes_per_run':design.UPDATES*design.BATCH_SIZE,
        'training_steps_per_run':design.UPDATES*design.BATCH_SIZE*design.HORIZON,
        'message_samples_per_run':design.UPDATES*design.BATCH_SIZE,
        'evaluation_episodes_per_condition':4096,'automatic_followon_experiment':False})
    return d


def source_hashes():
    files=[HERE/'__init__.py',HERE/'README.md',HERE/'design.py',HERE/'policy.py',
           HERE/'environment.py',HERE/'runner.py',HERE/'plan.md',HERE/'tests/test_game.py']
    require(all(p.is_file() for p in files),'missing source file')
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def prepare(out):
    out=Path(out).resolve(); require(not out.exists(),'refuse overwrite')
    d=make_prepared(); sources=source_hashes(); out.mkdir(parents=True)
    for rel in sources:
        q=out/'source_snapshot'/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/rel,q)
    (out/'prepared.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    plan={'schema':'tabular_signaling_v1','prepared_sha256':sha(out/'prepared.json'),'sources':sources,
          'runtime':{'python':platform.python_version(),'numpy':np.__version__},
          'no_training_or_model_initialization_by_prepare':True,'config':d}
    (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    (out/'freeze.json').write_text(json.dumps({'plan_sha256':sha(out/'plan.json'),'prepared_sha256':sha(out/'prepared.json')},indent=2)+'\n')
    verify(out); return {'status':'prepared','output':str(out),'plan_sha256':sha(out/'plan.json'),'prepared_sha256':sha(out/'prepared.json')}


def verify(out):
    out=Path(out); plan=json.loads((out/'plan.json').read_text()); d=json.loads((out/'prepared.json').read_text()); fr=json.loads((out/'freeze.json').read_text())
    require(sha(out/'plan.json')==fr['plan_sha256'],'plan hash mismatch')
    require(sha(out/'prepared.json')==fr['prepared_sha256']==plan['prepared_sha256'],'prepared hash mismatch')
    require(plan['config']==make_prepared(),'design changed'); require(plan['sources']==source_hashes(),'source changed')
    for rel,digest in plan['sources'].items(): require(sha(out/'source_snapshot'/rel)==digest,'snapshot changed '+rel)
    return plan,d


def save_checkpoint(path,p,update):
    np.savez_compressed(path,update=np.array(update,dtype=np.int64),sender_logits=p['sender_logits'],worker_logits=p['worker_logits'])
    return sha(path)


def _center(values,keys,mask=None):
    values=np.asarray(values,dtype=np.float64); keys=np.asarray(keys)
    mask=np.ones(len(values),dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    out=np.zeros_like(values)
    for key in np.unique(keys):
        idx=np.flatnonzero((keys==key)&mask)
        if len(idx)>1: out[idx]=values[idx]-(values[idx].sum()-values[idx])/(len(idx)-1)
    return out


def _entropy(arr,K):
    arr=np.asarray(arr); x=np.bincount(arr.ravel(),minlength=K).astype(float); x/=max(x.sum(),1); x=x[x>0]
    return float(-(x*np.log2(x)).sum())


def _entropy_grad(prob):
    lp=np.log(np.maximum(prob,1e-300)); ent=-(prob*lp).sum(axis=-1,keepdims=True)
    return -prob*(lp+ent)


def _message_mi(messages,ep):
    m=messages[:,0,0]; vals=[]
    for state in (ep['goal'][:,0,0],ep['context'],ep['site_type'][:,0]):
        joint=np.zeros((design.ALPHABET_SIZE+1, int(np.max(state))+1 if np.max(state)>1 else 2),float)
        for i in range(len(m)): joint[m[i],state[i]]+=1
        joint/=len(m); pm=joint.sum(1); ps=joint.sum(0); q=0.
        for i in range(joint.shape[0]):
            for j in range(joint.shape[1]):
                if joint[i,j]>0 and pm[i]>0 and ps[j]>0: q+=joint[i,j]*np.log2(joint[i,j]/(pm[i]*ps[j]))
        vals.append(q)
    return float(np.mean(vals))


def evaluate(p,seed,scarcity,task,information,channel,*,message_mode='natural',episodes_count=4096,evaluation=True):
    ep=design.episode_stream(seed,scarcity,task,episodes_count,evaluation=evaluation)
    tr=environment.rollout(p,ep,information,channel,message_mode=message_mode,sample=False)
    r=tr['team_return']; oracle=environment.oracle_team_return(ep); valid=oracle>1e-12
    norm=np.divide(r,oracle,out=np.full_like(r,np.nan),where=valid)
    token=tr['messages'][:,0,0]
    result={'episodes':episodes_count,'split':'heldout' if evaluation else 'training_support',
            'team_return_mean':float(r.mean()),'team_return_sd':float(r.std()),'positive_episode_rate':float((r>0).mean()),
            'oracle_team_return_mean':float(oracle.mean()),'oracle_positive_episode_rate':float(valid.mean()),
            'normalized_return_mean_on_oracle_positive':float(np.nanmean(norm)) if valid.any() else None,
            'oracle_regret_mean':float((oracle-r).mean()),'message_mode':message_mode,
            'message_entropy':float(_entropy(token,design.ALPHABET_SIZE+1)),
            'message_type_mi':float(_message_mi(tr['messages'],ep)),
            'messages_per_episode':1,'action_histogram':np.bincount(tr['actions'][:,:,1].ravel(),minlength=design.ACTION_COUNT).tolist(),
            'message_histogram':np.bincount(token,minlength=design.ALPHABET_SIZE+1).tolist(),
            'final_inventory_mean':tr['final_inventory'].mean(axis=0).tolist()}
    # This mapping is a descriptive codebook readout, never fed back into
    # training.  It is reported for contexts present in the evaluation split.
    contexts=np.unique(ep['context']); local=ep['site_type'][:,0]
    result['token_by_context_local']={str(int(c)): [int(p['sender_logits'][int(c),lt].argmax()) for lt in (0,1)]
                                      for c in contexts if int(c)<p['sender_logits'].shape[0]}
    return result


def train_one(seed,condition,out,updates=None):
    memory,scarcity,information,channel,task=design.parse_condition(condition)
    updates=design.UPDATES if updates is None else updates
    run=Path(out)/f'seed_{seed}_{condition}'; run.mkdir(parents=True,exist_ok=False)
    p=policy.make_policy(seed,task,memory); init=policy.parameter_hash(p)
    checkpoint_updates=[u for u in CHECKPOINTS if u<=updates]
    if 0 in checkpoint_updates: save_checkpoint(run/'checkpoint_0000.npz',p,0)
    rows=[]; start=time.perf_counter()
    for update in range(1,updates+1):
        ep=design.episode_stream(seed,scarcity,task,design.BATCH_SIZE,update=update)
        tr=environment.rollout(p,ep,information,channel,message_mode='natural',sample=True)
        B=design.BATCH_SIZE; g=policy.zero_grads(p); beta=design.entropy_coefficient(update)
        # Future worker return is the sender's objective; no message is sent
        # after t=0, so the score target excludes the communication-only step.
        worker_return=np.flip(np.cumsum(np.flip(tr['rewards'][:,:,1],axis=1),axis=1),axis=1)
        msg_value=worker_return[:,1:].sum(axis=1)
        key=design.CONTEXTS[task]*ep['site_type'][:,0]+ep['context']
        msg_adv=_center(msg_value,key)
        sp=tr['message_probs'][0]; one=np.zeros_like(sp)
        tok=np.where(tr['messages'][:,0,0]==design.NULL_MESSAGE,0,tr['messages'][:,0,0])
        one[np.arange(B),tok]=1
        if channel=='live':
            np.add.at(g['sender_logits'],(ep['context'],ep['site_type'][:,0]),
                      -(msg_adv[:,None]*(one-sp))/B - beta*_entropy_grad(sp)/B)
        for t in range(design.ACTION_START,design.HORIZON):
            ap=tr['action_probs'][t]; state=tr['state_idx'][t]; local=tr['local_idx'][t]; inv=tr['inv_idx'][t]; vg=tr['goal_idx'][t]
            act_adv=_center(worker_return[:,t],state*1000+t*100+local*10+inv*2+vg)
            oa=np.zeros_like(ap); oa[np.arange(B),tr['actions'][:,t,1]]=1
            d=-(act_adv[:,None]*(oa-ap))/B - beta*_entropy_grad(ap)/B
            np.add.at(g['worker_logits'],(state,np.full(B,t),local,inv,vg),d)
        norm,scale=policy.step(p,g,design.LEARNING_RATE)
        row={'update':update,'seed':seed,'condition':condition,'memory':memory,'scarcity':scarcity,'information':information,'channel':channel,'task':task,
             'episode_seed':int(update),'world_sha256':array_sha(ep['site_type']),'goal_sha256':array_sha(ep['goal']),
             'message_uniform_sha256':array_sha(ep['message_uniforms']),'action_uniform_sha256':array_sha(ep['action_uniforms']),
             'return_mean':float(tr['team_return'].mean()),'return_sd':float(tr['team_return'].std()),'gradient_norm':norm,
             'gradient_clip_scale':scale,'entropy_coefficient':beta,'parameter_sha256':policy.parameter_hash(p),
             'elapsed_seconds':time.perf_counter()-start}
        rows.append(row)
        if update in checkpoint_updates and update>0: row['checkpoint_sha256']=save_checkpoint(run/f'checkpoint_{update:04d}.npz',p,update)
    final={}
    for evaluation,label in ((False,'training_support'),(True,'heldout')):
        natural=evaluate(p,seed,scarcity,task,information,channel,message_mode='natural',evaluation=evaluation)
        if channel=='live':
            closed=evaluate(p,seed,scarcity,task,information,channel,message_mode='closed',evaluation=evaluation)
            permuted=evaluate(p,seed,scarcity,task,information,channel,message_mode='permuted',evaluation=evaluation)
        else:
            closed=dict(natural,message_mode='closed',reused_natural=True); permuted=dict(natural,message_mode='permuted',reused_natural=True)
        final[label]={'natural':natural,'closed':closed,'permuted':permuted}
    final.update(final['heldout'])
    result={'seed':seed,'condition':condition,'memory':memory,'scarcity':scarcity,'information':information,'channel':channel,'task':task,'updates':updates,
            'initial_parameter_sha256':init,'final_parameter_sha256':policy.parameter_hash(p),'trajectory':rows,'final':final,
            'training_log_sha256':None,'checkpoints':checkpoint_updates,'elapsed_seconds':time.perf_counter()-start}
    (run/'training.jsonl').write_bytes(b''.join(json_bytes(x) for x in rows)); result['training_log_sha256']=sha(run/'training.jsonl')
    (run/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n'); return result


def run_grid(prepared,out,updates=None,seeds=None,conditions=None):
    out=Path(out); execution=out/'execution'; execution.mkdir(parents=True)
    seeds=tuple(design.SEEDS if seeds is None else seeds); conditions=tuple(design.CONDITIONS if conditions is None else conditions)
    require(set(seeds).issubset(set(design.SEEDS)) and seeds,'invalid seed subset'); require(set(conditions).issubset(set(design.CONDITIONS)) and conditions,'invalid condition subset')
    results=[]; total=len(seeds)*len(conditions)
    for seed in seeds:
        for condition in conditions:
            results.append(train_one(seed,condition,execution,updates))
            (execution/'progress.json').write_text(json.dumps({'completed':len(results),'total':total,'seeds':list(seeds),'conditions':list(conditions)},indent=2))
    (execution/'results.json').write_text(json.dumps({'results':results},ensure_ascii=False,indent=2)+'\n'); return results


def cli():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--out',required=True)
    a=sub.add_parser('execute'); a.add_argument('--out',required=True); a.add_argument('--prepared',required=True); a.add_argument('--updates',type=int,default=None)
    a.add_argument('--seeds',default=None); a.add_argument('--conditions',default=None)
    args=p.parse_args()
    if args.cmd=='prepare': print(json.dumps(prepare(args.out),ensure_ascii=False)); return
    verify(args.prepared)
    seeds=None if args.seeds is None else tuple(int(x) for x in args.seeds.split(',') if x)
    conditions=None if args.conditions is None else tuple(x for x in args.conditions.split(',') if x)
    res=run_grid(args.prepared,args.out,args.updates,seeds,conditions)
    print(json.dumps({'status':'completed','runs':len(res),'out':str(Path(args.out).resolve())},ensure_ascii=False))


if __name__=='__main__': cli()
