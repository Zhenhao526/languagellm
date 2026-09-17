"""Train and audit the division-of-labour signaling game."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import platform
import shutil
import time
import numpy as np
from . import design, model, environment

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CHECKPOINTS=(0,500,1000,2000)

def require(ok,msg):
    if not ok: raise ValueError(msg)

def json_hash(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def array_sha(x): return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()

def json_bytes(x): return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()

def make_prepared():
    d=design.prepare(); d.update({'checkpoints':list(CHECKPOINTS),'runtime_policy':'float64 NumPy only',
        'runs':len(design.SEEDS)*len(design.CONDITIONS),'episode_batches_per_run':design.UPDATES,
        'training_episodes_per_run':design.UPDATES*design.BATCH_SIZE,
        'training_steps_per_run':design.UPDATES*design.BATCH_SIZE*design.HORIZON,
        'message_samples_per_run':design.UPDATES*design.BATCH_SIZE*len(design.MESSAGE_ROUNDS)*2,
        'evaluation_episodes_per_condition':4096,'automatic_followon_experiment':False})
    return d

def source_hashes():
    files=[HERE/'__init__.py',HERE/'README.md',HERE/'design.py',HERE/'model.py',HERE/'environment.py',HERE/'runner.py',HERE/'plan.md',HERE/'tests/test_game.py']
    require(all(p.is_file() for p in files),'missing source file')
    return {str(p.relative_to(ROOT)):sha(p) for p in files}

def prepare(out):
    out=Path(out).resolve(); require(not out.exists(),'refuse overwrite')
    d=make_prepared(); sources=source_hashes()
    out.mkdir(parents=True)
    for rel in sources:
        q=out/'source_snapshot'/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/rel,q)
    (out/'prepared.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    plan={'schema':'role_signaling_communication_v1','prepared_sha256':sha(out/'prepared.json'),
          'sources':sources,'runtime':{'python':platform.python_version(),'numpy':np.__version__},
          'no_training_or_model_initialization_by_prepare':True,'config':d}
    (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    (out/'freeze.json').write_text(json.dumps({'plan_sha256':sha(out/'plan.json'),'prepared_sha256':sha(out/'prepared.json')},indent=2)+'\n')
    verify(out)
    return {'status':'prepared','output':str(out),'plan_sha256':sha(out/'plan.json'),'prepared_sha256':sha(out/'prepared.json')}

def verify(out):
    out=Path(out); plan=json.loads((out/'plan.json').read_text()); d=json.loads((out/'prepared.json').read_text()); fr=json.loads((out/'freeze.json').read_text())
    require(sha(out/'plan.json')==fr['plan_sha256'],'plan hash mismatch'); require(sha(out/'prepared.json')==fr['prepared_sha256']==plan['prepared_sha256'],'prepared hash mismatch')
    require(plan['config']==make_prepared(),'design changed'); require(plan['sources']==source_hashes(),'source changed')
    for rel,digest in plan['sources'].items(): require(sha(out/'source_snapshot'/rel)==digest,'snapshot changed '+rel)
    return plan,d

def save_checkpoint(path,nets,opt,update):
    payload={'update':np.array(update,dtype=np.int64)}
    for a,n in enumerate(nets):
        for k,v in n.items(): payload[f'agent{a}_{k}']=v
    for a,s in enumerate(opt):
        for k,q in s.items():
            payload[f'agent{a}_{k}_m']=q['m']; payload[f'agent{a}_{k}_v']=q['v']
    np.savez_compressed(path,**payload); return sha(path)

def _backward(nets,trace,memory,information,channel,update):
    B,H=trace['rewards'].shape[0],design.HORIZON
    # A scout's token changes the worker's future action, and the duplicated
    # worker reward is the scout's objective. Actions are learned only on
    # worker rows; scout actions are structurally forced to wait. State-group
    # baselines reduce variance without conditioning on researcher-only fields.
    agent_returns=np.flip(np.cumsum(np.flip(trace['rewards'],axis=1),axis=1),axis=1)
    episodes=trace['episodes']
    returns=agent_returns.mean(axis=2)
    grads=model.zero_grads(nets); carry=[np.zeros((B,design.HIDDEN)) for _ in design.AGENTS]
    beta=design.entropy_coefficient(max(1,min(update,design.UPDATES)))
    for t in range(H-1,-1,-1):
      for a in design.AGENTS:
        cache,hh,ml,al,x=trace['cache'][t][a]; x0,recv,hprev,pre,h=cache
        pm=trace['probs_msg'][t][a]; pa=trace['probs_act'][t][a]
        sendmask=(episodes['sender']==a)
        workmask=~sendmask
        # Loss is negative return; sampled-score terms exist only when a live
        # token is actually delivered.  After round 1 the protocol is closed.
        dml=np.zeros_like(ml)
        if channel=='live' and t in design.MESSAGE_ROUNDS:
            one=np.zeros_like(pm)
            idx=np.flatnonzero(sendmask)
            if len(idx): one[idx,trace['messages'][idx,t,a]]=1.
            # Condition the score baseline on the sender's private state.  A
            # global batch mean would mix the two request values and leaves a
            # very noisy message credit signal in the partner-demand game.
            msg_key=2*episodes['goal'][:,a,t]+episodes['site_type'][:,a]
            msg_values=agent_returns[:,t,a]-trace['rewards'][:,t,a]
            msg_adv=_group_center(msg_values,msg_key,sendmask)
            dml += -(msg_adv[:,None]*(one-pm))/B
            dml *= sendmask[:,None]
        if channel=='live' and t in design.MESSAGE_ROUNDS:
            dml += -beta*model.entropy_grad(pm)*sendmask[:,None]/len(design.AGENTS)
        onea=np.zeros_like(pa); onea[np.arange(B),trace['actions'][:,t,a]]=1.
        act_key=2*episodes['goal'][:,a,t]+episodes['site_type'][:,a]
        act_adv=_group_center(agent_returns[:,t,a],act_key,workmask)
        dal=-(act_adv[:,None]*(onea-pa))/B
        dal *= workmask[:,None]
        dal += -beta*model.entropy_grad(pa)*workmask[:,None]/len(design.AGENTS)
        g=grads[a]
        g['W_msg'] += np.einsum('bi,bj->ij',h,dml); g['b_msg'] += dml.sum(0)
        g['W_act'] += np.einsum('bi,bj->ij',h,dal); g['b_act'] += dal.sum(0)
        dh=(np.einsum('bk,hk->bh',dml,nets[a]['W_msg'])
            +np.einsum('bk,hk->bh',dal,nets[a]['W_act'])+carry[a])
        dpre=dh*(1-h*h)
        g['W_obs'] += np.einsum('bi,bh->ih',x0,dpre)
        g['W_recv'] += np.einsum('bi,bh->ih',recv,dpre); g['b_h'] += dpre.sum(0)
        # W_h is deliberately disabled in stateless runs by hprev=0 in cache;
        # the caller still receives an exact zero gradient for it.
        g['W_h'] += np.einsum('bi,bh->ih',hprev,dpre)
        carry[a]=(np.einsum('bi,hi->bh',dpre,nets[a]['W_h'])
                  if memory=='recurrent' else np.zeros_like(carry[a]))
    return grads,{'return_mean':float(trace['team_return'].mean()),'return_sd':float(trace['team_return'].std()),
                  'future_return_mean':returns.mean(axis=0).tolist(),'message_entropy_mean':float(np.mean([-(p*np.log(np.maximum(p,1e-300))).sum(-1).mean() for row in trace['probs_msg'] for p in row]))}


def _group_center(values, keys, mask=None):
    """Subtract a leave-one-out mean within small observable state groups."""
    values=np.asarray(values,dtype=np.float64); keys=np.asarray(keys)
    out=np.zeros_like(values)
    mask=np.ones(len(values),dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    for key in np.unique(keys):
        idx=np.flatnonzero((keys==key)&mask)
        if len(idx)>1:
            out[idx]-=(values[idx].sum()-values[idx])/(len(idx)-1)
        else:
            out[idx]=0.
    return out

def episode(seed,scarcity,task,update,B):
    return design.episode_stream(seed,scarcity,task,B,update=update)

def evaluate(nets,seed,scarcity,task,memory,information,channel,*,message_mode='natural',episodes_count=4096,evaluation=True):
    ep=design.episode_stream(seed,scarcity,task,episodes_count,evaluation=evaluation)
    tr=environment.run_episode_batch(nets,ep,memory,information,channel,sample=False,controls={'message_mode':message_mode},collect=False)
    r=tr['team_return']; actions=tr['actions']; messages=tr['messages']
    oracle=environment.oracle_team_return(ep)
    valid=oracle>1e-12
    normalized=np.divide(r,oracle,out=np.full_like(r,np.nan),where=valid)
    # Exact discrete summaries; team return includes scarcity costs and is
    # directly comparable within a scarcity regime.
    result={'episodes':episodes_count,'split':'heldout' if evaluation else 'training_support',
            'team_return_mean':float(r.mean()),'team_return_sd':float(r.std()),
            'positive_episode_rate':float((r>0).mean()),
            'oracle_team_return_mean':float(oracle.mean()),
            'oracle_positive_episode_rate':float(valid.mean()),
            'normalized_return_mean_on_oracle_positive':float(np.nanmean(normalized)) if valid.any() else None,
            'oracle_regret_mean':float((oracle-r).mean()),
            'message_mode':message_mode,'message_entropy':float(_active_message_entropy(messages,ep)),
            'message_type_mi':float(_message_mi(messages,ep)),
            'messages_per_episode':int(len(design.MESSAGE_ROUNDS)),
            'action_histogram':np.bincount(actions.ravel(),minlength=design.ACTION_COUNT).tolist(),
            'message_histogram':np.bincount(_active_messages(messages,ep),minlength=design.ALPHABET_SIZE).tolist(),
            'final_inventory_mean':tr['final_inventory'].mean(axis=0).tolist()}
    return result

def _entropy(arr,K):
    x=np.bincount(np.asarray(arr).ravel(),minlength=K)/max(np.asarray(arr).size,1); x=x[x>0]; return float(-(x*np.log2(x)).sum())

def _active_messages(messages,ep):
    vals=[]
    sender=np.asarray(ep['sender'])
    for a in design.AGENTS:
        idx=np.flatnonzero(sender==a)
        if len(idx):
            vals.extend(np.asarray(messages[idx][:,design.MESSAGE_ROUNDS,a]).ravel().tolist())
    return np.asarray(vals,dtype=np.int64)

def _active_message_entropy(messages,ep):
    vals=_active_messages(messages,ep)
    return _entropy(vals,design.ALPHABET_SIZE) if len(vals) else 0.0

def _message_mi(messages,ep):
    # MI(token; sender own goal) and MI(token; sender local site type), averaged
    vals=[]
    sender=np.asarray(ep['sender'])
    for a in design.AGENTS:
      idx=np.flatnonzero(sender==a)
      if len(idx)==0: continue
      for t in design.MESSAGE_ROUNDS:
       m=messages[idx,t,a]
       for state in (ep['goal'][idx,a,t], ep['site_type'][idx,a]):
        joint=np.zeros((design.ALPHABET_SIZE+1,2),float)
        for i in range(len(m)): joint[m[i],state[i]]+=1
        joint/=len(m); pm=joint.sum(1); ps=joint.sum(0); q=0
        for i in range(joint.shape[0]):
         for j in range(2):
          if joint[i,j]>0:q+=joint[i,j]*np.log2(joint[i,j]/(pm[i]*ps[j]))
        vals.append(q)
    return float(np.mean(vals))

def train_one(seed,condition,out,updates=None):
    memory,scarcity,information,channel,task=design.parse_condition(condition); updates=design.UPDATES if updates is None else updates
    run=Path(out)/f'seed_{seed}_{condition}'; run.mkdir(parents=True,exist_ok=False)
    nets=model.make_networks(seed,memory); init=model.parameter_hash(nets); opt=model.adam_state(nets)
    checkpoint_updates=[u for u in CHECKPOINTS if u <= updates]
    if 0 in checkpoint_updates:
        save_checkpoint(run/'checkpoint_0000.npz',nets,opt,0)
    rows=[]; start=time.perf_counter()
    for update in range(1,updates+1):
        ep=episode(seed,scarcity,task,update,design.BATCH_SIZE)
        tr=environment.run_episode_batch(nets,ep,memory,information,channel,sample=True,collect=True)
        grads,diag=_backward(nets,tr,memory,information,channel,update)
        norm,scale=model.adam_step(nets,grads,opt,update)
        row={'update':update,'seed':seed,'condition':condition,'memory':memory,'scarcity':scarcity,'information':information,'channel':channel,'task':task,
             'episode_seed':int(update),'world_sha256':array_sha(ep['site_type']),
             'goal_sha256':array_sha(ep['goal']),'message_uniform_sha256':array_sha(ep['message_uniforms']),
             'action_uniform_sha256':array_sha(ep['action_uniforms']),'return_mean':diag['return_mean'],'return_sd':diag['return_sd'],
             'gradient_norm':norm,'gradient_clip_scale':scale,'entropy_coefficient':design.entropy_coefficient(update),
             'parameter_sha256':model.parameter_hash(nets),'elapsed_seconds':time.perf_counter()-start}
        rows.append(row)
        if update in checkpoint_updates and update > 0:
            row['checkpoint_sha256']=save_checkpoint(run/f'checkpoint_{update:04d}.npz',nets,opt,update)
    # Causal controls are evaluated from final parameters; no policy updates.
    # Keep both the training-support split and the held-out transition split so
    # acquisition and compositional generalization cannot be conflated.
    final={}
    for evaluation, label in ((False,'training_support'),(True,'heldout')):
        natural=evaluate(nets,seed,scarcity,task,memory,information,channel,
                         message_mode='natural',evaluation=evaluation)
        if channel=='live':
            closed=evaluate(nets,seed,scarcity,task,memory,information,channel,
                            message_mode='closed',evaluation=evaluation)
            permuted=evaluate(nets,seed,scarcity,task,memory,information,channel,
                              message_mode='permuted',evaluation=evaluation)
        else:
            closed=dict(natural,message_mode='closed',reused_natural=True)
            permuted=dict(natural,message_mode='permuted',reused_natural=True)
        final[label]={'natural':natural,'closed':closed,'permuted':permuted}
    # Backwards-compatible aliases point to the held-out split.
    final.update(final['heldout'])
    result={'seed':seed,'condition':condition,'memory':memory,'scarcity':scarcity,'information':information,'channel':channel,'task':task,'updates':updates,
      'initial_parameter_sha256':init,'final_parameter_sha256':model.parameter_hash(nets),'trajectory':rows,'final':final,
      'training_log_sha256':None,'checkpoints':checkpoint_updates,'elapsed_seconds':time.perf_counter()-start}
    (run/'training.jsonl').write_bytes(b''.join(json_bytes(x) for x in rows)); result['training_log_sha256']=sha(run/'training.jsonl')
    (run/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result

def run_grid(prepared,out,updates=None,seeds=None,conditions=None):
    out=Path(out); execution=out/'execution'; execution.mkdir(parents=True)
    seeds=tuple(design.SEEDS if seeds is None else seeds)
    conditions=tuple(design.CONDITIONS if conditions is None else conditions)
    require(set(seeds).issubset(set(design.SEEDS)) and len(seeds)>0,'invalid seed subset')
    require(set(conditions).issubset(set(design.CONDITIONS)) and len(conditions)>0,'invalid condition subset')
    results=[]
    # Sequential execution keeps the small local test deterministic and avoids
    # oversubscribing the Mac.  The frozen grid remains fully paired.
    total=len(seeds)*len(conditions)
    for seed in seeds:
      for condition in conditions:
        results.append(train_one(seed,condition,execution,updates))
        (execution/'progress.json').write_text(json.dumps({'completed':len(results),'total':total,'seeds':list(seeds),'conditions':list(conditions)},indent=2))
    (execution/'results.json').write_text(json.dumps({'results':results},ensure_ascii=False,indent=2)+'\n')
    return results

def cli():
 p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
 a=sub.add_parser('prepare'); a.add_argument('--out',required=True)
 a=sub.add_parser('execute'); a.add_argument('--out',required=True); a.add_argument('--prepared',required=True); a.add_argument('--updates',type=int,default=None)
 a.add_argument('--seeds',default=None,help='comma-separated subset of frozen seeds for smoke tests')
 a.add_argument('--conditions',default=None,help='comma-separated subset of frozen conditions for smoke tests')
 args=p.parse_args()
 if args.cmd=='prepare': print(json.dumps(prepare(args.out),ensure_ascii=False))
 else:
  verify(args.prepared)
  seeds=None if args.seeds is None else tuple(int(x) for x in args.seeds.split(',') if x)
  conditions=None if args.conditions is None else tuple(x for x in args.conditions.split(',') if x)
  res=run_grid(args.prepared,args.out,args.updates,seeds,conditions); print(json.dumps({'status':'completed','runs':len(res),'out':str(Path(args.out).resolve())},ensure_ascii=False))
if __name__=='__main__': cli()
