"""Train and measure the predeclared local pilot.

No natural-language model, no semantic labels in communication policy losses,
no straight-through/differentiable messages, and no shared agent parameters.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, platform, resource, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .env import VisualWorld
from .model import Agent,DIM,empty_history,private_record,append_record,sample_policy


def sync(device):
    if str(device)=='mps': torch.mps.synchronize()


def write_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
    tmp.replace(path)


def device_info():
    return {'torch':torch.__version__,'python':platform.python_version(),
            'mps_built':torch.backends.mps.is_built(),'mps_available':torch.backends.mps.is_available(),
            'mps_fallback_env':os.getenv('PYTORCH_ENABLE_MPS_FALLBACK','unset'),
            'numpy':np.__version__,'cpu_threads':torch.get_num_threads()}


def to_images(batch,device):
    return (torch.from_numpy(batch['target_images']).to(device),
            torch.from_numpy(batch['candidate_images']).to(device))


@torch.no_grad()
def feature_batch(agents,batch,device):
    target,candidate=to_images(batch,device)
    # Simulator can batch computation, but policy calls below select only the
    # acting individual's permitted observations. Encoders are already frozen.
    return [(a.encoder(target),a.encoder(candidate.flatten(0,1)).reshape(-1,4,DIM)) for a in agents]


@torch.no_grad()
def visual_accuracy(agent,seed,device,n=512):
    world=VisualWorld(seed); correct=0
    for offset in range(0,n,128):
        batch=world.sample(min(128,n-offset)); x,c=to_images(batch,device)
        selected=agent.visual_match(x,c).argmax(-1).cpu().numpy()
        correct+=int((selected==batch['correct_indices']).sum())
    return correct/n


def pretrain(seed,out,device,updates=250,batch_size=128):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/f'pretrained_seed{seed}.pt'
    metadata=out/f'pretrained_seed{seed}.json'
    if checkpoint.exists() and metadata.exists(): return checkpoint,json.loads(metadata.read_text())
    agents=[]; infos=[]; start=time.monotonic()
    for index in range(2):
        torch.manual_seed(seed*1000+index*137+17)
        agent=Agent().to(device)
        # Only perception is prepared; communication and history policies retain
        # their independent random initialization.
        opt=torch.optim.Adam(list(agent.encoder.parameters())+[agent.visual_log_temperature],lr=1e-3)
        world=VisualWorld(seed*10000+index*457+31)
        policy_rng=np.random.default_rng(seed*10000+index*457+37)
        before=visual_accuracy(agent,seed*10000+index*31+800000,device)
        train_curve=[]
        for update in range(updates):
            batch=world.sample(batch_size); target,candidate=to_images(batch,device)
            logits=agent.visual_match(target,candidate)
            logp=F.log_softmax(logits,-1); probs=logp.exp()
            u=torch.from_numpy(policy_rng.random(batch_size).astype(np.float32)).to(device)
            choices=(probs.detach().cumsum(-1)<u[:,None]).sum(-1).clamp(max=3)
            # Hidden metadata is used exclusively to compute scalar rewards.
            correct=torch.from_numpy(batch['correct_indices']).to(device)
            reward=(choices==correct).float()
            # A constant baseline has no dependence on a hidden answer.
            loss=-((reward-.25).detach()*logp.gather(1,choices[:,None]).squeeze(1)).mean()
            loss-=.01*(-(probs*logp).sum(-1)).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(agent.parameters(),2.)
            opt.step()
            if (update+1)%50==0 or update==updates-1:
                acc=visual_accuracy(agent,seed*10000+index*31+800001,device,256)
                train_curve.append({'update':update+1,'greedy_accuracy':acc})
                print(json.dumps({'stage':'pretrain','seed':seed,'agent':index,'update':update+1,'accuracy':acc}),flush=True)
        after=visual_accuracy(agent,seed*10000+index*31+800002,device,1024)
        agents.append({k:v.detach().cpu() for k,v in agent.state_dict().items()})
        infos.append({'agent':index,'initial_visual_accuracy':before,'accuracy':after,
                      'updates':updates,'trials':updates*batch_size,'curve':train_curve})
    sync(device)
    torch.save({'agents':agents,'seed':seed},checkpoint)
    info={'seed':seed,'device':str(device),'seconds':time.monotonic()-start,'agents':infos,
          'params_per_agent':sum(p.numel() for p in agent.parameters()),'method':'reward-only visual matching',
          'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    write_json(metadata,info)
    return checkpoint,info


def load_agents(checkpoint,device):
    data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    agents=[]
    for state in data['agents']:
        a=Agent(); a.load_state_dict(state); a.freeze_perception(); a.to(device); agents.append(a)
    return agents


@torch.no_grad()
def evaluate(agents,histories,condition,task,codebook,seed,device,n=512,intervention=False):
    # Estimate meaning and naturally used symbols on a separate reference set.
    # Its observations/rewards never enter a policy or alter either history.
    reference=evaluate(agents,histories,condition,task,codebook,seed+424242,
                       device,n=1024,intervention=False) if intervention else None
    world=VisualWorld(seed)
    role_rng=np.random.default_rng(seed+1)
    correct=0; cleared_correct=0; intervention_correct=0; intervention_follow=0; intervention_n=0
    original_correct_on_intervened=0
    direction_correct=np.zeros(2,dtype=np.int64)
    direction_cleared_correct=np.zeros(2,dtype=np.int64)
    direction_n=np.zeros(2,dtype=np.int64)
    message_counts=np.zeros((2,4,4),dtype=np.int64)
    response_counts=np.zeros((2,4,4),dtype=np.int64)
    all_rows=[]
    enabled=condition=='H1'
    for start in range(0,n,128):
        b=world.sample(min(128,n-start)); z=feature_batch(agents,b,device)
        roles=role_rng.integers(0,2,len(b['target_ids']))
        for sender in range(2):
            idx=np.flatnonzero(roles==sender)
            if len(idx)==0: continue
            receiver=1-sender; ix=torch.as_tensor(idx,device=device)
            target=z[sender][0][ix]; candidates=z[receiver][1][ix]
            hs=histories[sender][None].expand(len(idx),-1,-1)
            hr=histories[receiver][None].expand(len(idx),-1,-1)
            if task=='emergent':
                messages=agents[sender].send(target,hs,enabled).argmax(-1)
                clear_messages=agents[sender].send(target,hs,False).argmax(-1)
            elif task=='fixed':
                messages=torch.as_tensor(codebook[b['target_ids'][idx]],device=device)
                clear_messages=messages
            else:
                messages=torch.zeros(len(idx),dtype=torch.long,device=device); clear_messages=messages
            choices=agents[receiver].receive(candidates,messages,hr,enabled).argmax(-1)
            clear_choices=agents[receiver].receive(candidates,clear_messages,hr,False).argmax(-1)
            msg_np=messages.cpu().numpy(); choice_np=choices.cpu().numpy()
            selected=b['candidate_ids'][idx,choice_np]
            clear_selected=b['candidate_ids'][idx,clear_choices.cpu().numpy()]
            true=b['target_ids'][idx]
            correct+=int((selected==true).sum()); cleared_correct+=int((clear_selected==true).sum())
            direction_correct[sender]+=int((selected==true).sum())
            direction_cleared_correct[sender]+=int((clear_selected==true).sum())
            direction_n[sender]+=len(idx)
            np.add.at(message_counts[sender],(true,msg_np),1)
            np.add.at(response_counts[sender],(msg_np,selected),1)
            if intervention:
                all_rows.append((sender,candidates,hr,messages,b['candidate_ids'][idx].copy(),
                                 true.copy(),selected.copy()))
    reference_counts=np.asarray(reference['message_counts']) if reference else None
    mappings=[reference_counts[s].argmax(axis=0) for s in range(2)] if reference else []
    reference_used=[np.flatnonzero(reference_counts[s].sum(axis=0)>0)
                    for s in range(2)] if reference else []
    # The reference set alone determines eligibility and replacement symbols.
    # Both natural and intervened actions are scored on the same eligible trials.
    for sender,candidates,hr,messages,candidate_ids,true,original_selected in all_rows:
        used=reference_used[sender]
        if len(used)<2: continue
        substitutions={int(m):int(used[(i+1)%len(used)]) for i,m in enumerate(used)}
        natural_messages=messages.cpu().numpy()
        eligible=np.flatnonzero(np.isin(natural_messages,used))
        if len(eligible)==0: continue
        eligible_tensor=torch.as_tensor(eligible,device=device)
        alt_np=np.array([substitutions[int(natural_messages[i])] for i in eligible])
        alt=torch.as_tensor(alt_np,device=device)
        choice=agents[1-sender].receive(candidates[eligible_tensor],alt,
                                       hr[eligible_tensor],enabled).argmax(-1).cpu().numpy()
        chosen=candidate_ids[eligible,choice]
        intervention_correct+=int((chosen==true[eligible]).sum())
        intervention_follow+=int((chosen==mappings[sender][alt_np]).sum())
        original_correct_on_intervened+=int((original_selected[eligible]==true[eligible]).sum())
        intervention_n+=len(choice)
    return {'accuracy':correct/n,'cleared_accuracy':cleared_correct/n,'n':n,'n_trials':n,
            'direction_accuracy':[int(direction_correct[s])/int(direction_n[s])
                                  if direction_n[s] else None for s in range(2)],
            'direction_cleared_accuracy':[int(direction_cleared_correct[s])/int(direction_n[s])
                                          if direction_n[s] else None for s in range(2)],
            'direction_n':direction_n.tolist(),
            'intervention_accuracy':intervention_correct/intervention_n if intervention_n else None,
            'intervention_follow_rate':intervention_follow/intervention_n if intervention_n else None,
            'original_accuracy_on_intervened':original_correct_on_intervened/intervention_n
                                             if intervention_n else None,
            'intervention_n':intervention_n,'intervention_n_trials':intervention_n,
            'intervention_follow_n_trials':intervention_n,'message_counts':message_counts.tolist(),
            'response_counts':response_counts.tolist(),
            'used_symbols':[int((message_counts[s].sum(axis=0)>0).sum()) for s in range(2)],
            'reference_n':reference['n'] if reference else 0,
            'reference_seed':seed+424242 if reference else None,
            'reference_direction_n':reference['direction_n'] if reference else None,
            'reference_used_symbols':[used.tolist() for used in reference_used] if reference else None,
            'reference_message_to_class':[[int(mappings[s][m]) if m in reference_used[s] else None
                                           for m in range(4)] for s in range(2)] if reference else None,
            'reference_message_counts':reference_counts.tolist() if reference else None,
            'evaluation_mode':'greedy; fixed private training-end history; no parameter/history updates or feedback'}


def run(checkpoint,info,seed,condition,task,device,out,steps=12000,rollout=32,
        eval_every=1000,eval_n=512,lr=.002,entropy=.03,wall_limit=900,save=True):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    agents=load_agents(checkpoint,device)
    optimizers=[torch.optim.Adam([p for p in a.parameters() if p.requires_grad],lr=lr) for a in agents]
    histories=[empty_history(device),empty_history(device)]
    world=VisualWorld(seed*100000+1001)
    role_rng=np.random.default_rng(seed*100000+1002)
    policy_rng=np.random.default_rng(seed*100000+1003)
    codebook=np.random.default_rng(seed*100000+1004).permutation(4)
    enabled=condition=='H1'; evaluations=[]; train_rewards=[]; done=0; start=time.monotonic()
    baselines=np.full((2,2),.25,dtype=np.float64)
    params=sum(p.numel() for p in agents[0].parameters())
    config={'seed':seed,'condition':condition,'task':task,'planned_steps':steps,'rollout':rollout,
            'eval_every':eval_every,'eval_n':eval_n,'lr':lr,'entropy':entropy,'wall_limit':wall_limit,
            'device':str(device),'codebook_for_fixed_control':codebook.tolist() if task=='fixed' else None,
            'params_per_agent':params,'pretrained_checkpoint_sha256':info['checkpoint_sha256'],
            'pretrain_accuracy':[a['accuracy'] for a in info['agents']],
            'visual_encoder':'prepared with reward-only matching, then frozen',
            'history':'last 4 completed private episodes, updated chronologically after decisions',
            'learning':'local REINFORCE with detached shared 0/1 reward; no cross-message gradients'}
    if save: write_json(out/'config.json',config)
    log=(out/'interactions.jsonl').open('w') if save else None
    initial=evaluate(agents,histories,condition,task,codebook,seed+9000000,device,eval_n)
    initial['step']=0; evaluations.append(initial)
    next_eval=eval_every
    last_history=None
    try:
        while done<steps:
            if time.monotonic()-start>wall_limit: break
            # End a rollout exactly at an evaluation boundary to make all
            # conditions use the same interaction checkpoints.
            count=min(rollout,steps-done,next_eval-done)
            batch=world.sample(count); z=feature_batch(agents,batch,device)
            roles=role_rng.integers(0,2,count)
            uniforms=policy_rng.random((count,2))
            terms=[[],[]]
            batch_rewards=[]
            for t in range(count):
                sender=int(roles[t]); receiver=1-sender
                hs=histories[sender][None]; hr=histories[receiver][None]
                sender_lp=sender_ent=None
                if task=='emergent':
                    logits=agents[sender].send(z[sender][0][t:t+1],hs,enabled)[0]
                    message,sender_lp,sender_ent=sample_policy(logits,uniforms[t,0])
                elif task=='fixed': message=int(codebook[batch['target_ids'][t]])
                else: message=0
                msg=torch.tensor([message],dtype=torch.long,device=device)
                logits=agents[receiver].receive(z[receiver][1][t:t+1],msg,hr,enabled)[0]
                choice,receiver_lp,receiver_ent=sample_policy(logits,uniforms[t,1])
                # The ONLY use of hidden correctness in policy optimization.
                reward=float(choice==int(batch['correct_indices'][t]))
                if sender_lp is not None:
                    terms[sender].append(-(reward-baselines[sender,0])*sender_lp-entropy*sender_ent)
                terms[receiver].append(-(reward-baselines[receiver,1])*receiver_lp-entropy*receiver_ent)
                baselines[sender,0]=.99*baselines[sender,0]+.01*reward
                baselines[receiver,1]=.99*baselines[receiver,1]+.01*reward
                # No target label or receiver choice enters the sender's event;
                # no hidden target feature enters the receiver's event.
                histories[sender]=append_record(histories[sender],private_record(0,z[sender][0][t],message,None,reward))
                histories[receiver]=append_record(histories[receiver],private_record(1,z[receiver][1][t],message,choice,reward))
                batch_rewards.append(reward)
                if log:
                    log.write(json.dumps({'step':done+t+1,'sender':sender,'target_id':int(batch['target_ids'][t]),
                      'candidate_ids':batch['candidate_ids'][t].tolist(),'message':message,'choice':choice,
                      'selected_id':int(batch['candidate_ids'][t,choice]),'reward':reward})+'\n')
            for i in range(2):
                if terms[i]:
                    loss=torch.stack(terms[i]).mean()
                    optimizers[i].zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(agents[i].parameters(),2.)
                    optimizers[i].step()
            done+=count; train_rewards.extend(batch_rewards)
            if done==next_eval or done==steps:
                result=evaluate(agents,histories,condition,task,codebook,seed+9000000+done,device,eval_n)
                result.update(step=done,train_accuracy=float(np.mean(train_rewards[-eval_every:])),seconds=time.monotonic()-start)
                evaluations.append(result)
                print(json.dumps({'stage':'train','seed':seed,'condition':condition,'task':task,
                                  'step':done,'accuracy':result['accuracy'],'cleared':result['cleared_accuracy'],
                                  'seconds':round(result['seconds'],2)}),flush=True)
                next_eval+=eval_every
                if save: write_json(out/'progress.json',{'config':config,'evaluations':evaluations})
    finally:
        if log: log.close()
    sync(device)
    main_loop_seconds=time.monotonic()-start
    final=evaluate(agents,histories,condition,task,codebook,seed+9900000,device,max(eval_n,2048),True)
    sync(device)
    summary=dict(config,steps=done,train_seconds=time.monotonic()-start,main_loop_seconds=main_loop_seconds,evaluations=evaluations,
                 final_eval=final,stopped_by_time=done<steps,
                 process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if str(device)=='mps':
        summary['mps_driver_allocated_bytes']=torch.mps.driver_allocated_memory()
        summary['mps_current_allocated_bytes']=torch.mps.current_allocated_memory()
    if save:
        write_json(out/'summary.json',summary)
        torch.save({'agents':[{k:v.detach().cpu() for k,v in a.state_dict().items()} for a in agents],
                    'histories':[h.cpu() for h in histories],'config':config},out/'final.pt')
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['pretrain','benchmark','run'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('pilot_round1/results/pretrained'))
    p.add_argument('--device',choices=['cpu','mps'],default='cpu')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--condition',choices=['H0','H1'],default='H0')
    p.add_argument('--task',choices=['emergent','fixed','no_comm'],default='emergent')
    p.add_argument('--steps',type=int,default=12000)
    p.add_argument('--pretrain-updates',type=int,default=250)
    p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--eval-n',type=int,default=512)
    p.add_argument('--wall-limit',type=float,default=900)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True,exist_ok=True)
    write_json(args.out/'runtime.json',device_info())
    checkpoint,info=pretrain(args.seed,args.cache,args.device,args.pretrain_updates)
    if args.stage=='pretrain':
        print(json.dumps(info),flush=True); return
    if min(a['accuracy'] for a in info['agents'])<.9:
        raise RuntimeError('Visual calibration below 90%; communication experiment not started')
    if args.stage=='benchmark':
        results=[]
        for device in ['cpu','mps']:
            if device=='mps' and not torch.backends.mps.is_available(): continue
            begin=time.monotonic()
            try:
                result=run(checkpoint,info,args.seed,'H1','emergent',device,args.out/device,
                           steps=1024,eval_every=1024,eval_n=128,wall_limit=150,save=False)
                results.append({'device':device,'seconds':time.monotonic()-begin,'steps':result['steps'],
                                'accuracy':result['final_eval']['accuracy'],
                                'main_loop_seconds':result['main_loop_seconds'],
                                'peak_rss_bytes':result['process_peak_rss_bytes']})
            except Exception as e:
                results.append({'device':device,'error':str(e)})
        write_json(args.out/'benchmark.json',results); print(json.dumps(results),flush=True)
    else:
        result=run(checkpoint,info,args.seed,args.condition,args.task,args.device,args.out,args.steps,
                   eval_every=args.eval_every,eval_n=args.eval_n,wall_limit=args.wall_limit)
        print(json.dumps({'complete':True,'out':str(args.out),'accuracy':result['final_eval']['accuracy'],
                          'seconds':result['train_seconds']}),flush=True)


if __name__=='__main__': main()
