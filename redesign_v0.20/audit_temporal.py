"""Bounded independent temporal audit; no production loss/observer calls.

Standard PyTorch GRU/LayerNorm/Adam primitives are reused, but the observation
routing, loss graph and world generator below are independent implementations.
All-step checks concern saved inputs; exact update replays are first and2101.
"""
from pathlib import Path
from itertools import product,combinations
from functools import lru_cache
import argparse,hashlib,importlib.util,json,sys,traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
spec=importlib.util.spec_from_file_location('readout_audit_helpers',PROJECT/'redesign_v0.19/audit_readout.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
read,sha,load,arrays,same,write,Audit=b.read,b.sha,b.load,b.arrays,b.same,b.write,b.Audit
MAPS=b.MAPS

@lru_cache(None)
def events(p,train=False):
    old=set(b.groups(p)['old']);rows=[]
    for a,t,m in product(sorted(old),sorted(old) if train else range(30),(0,1)):
        if MAPS[a,1-m]==MAPS[t,1-m] and MAPS[t,m] not in MAPS[a]:rows.append((a,t,m))
    return np.asarray(rows,np.int64)

def world(seed,p,who,step,n,pools):
    seq=lambda purpose:np.random.SeedSequence([20020,seed,p,who,purpose,step])
    rng=np.random.Generator(np.random.PCG64(seq(1)));index=rng.integers(72,size=n);e=events(p,True)[index]
    photos=np.column_stack([rng.choice(pools[k],n) for k in (0,1)]);goals=rng.integers(2,size=n)
    u=np.random.Generator(np.random.PCG64(seq(2))).random((2*n,1)).astype(np.float32)
    return dict(event_id=index,source_map=e[:,0],target_map=e[:,1],mover=e[:,2],photo_ids=photos,positions=MAPS[e[:,1]],goals=goals,action_uniform=u)

def eval_world(p,pools):
    e=events(p);e=np.repeat(e,np.where(np.isin(e[:,1],b.groups(p)['old']),3,2),axis=0)
    photos=np.asarray(list(product(*[np.sort(pools[k])[:4] for k in (0,1)])),np.int64);x=np.repeat(e,16,axis=0)
    return dict(source_map=x[:,0],target_map=x[:,1],mover=x[:,2],positions=MAPS[x[:,1]],photo_ids=np.tile(photos,(len(e),1)))

def frames(projected,w,mode):
    n=len(w['positions']);out=projected.new_zeros(n,2,6,65);rr=torch.arange(n)
    for k in (0,1):
        for t in (0,1):
            mask=np.ones(n,bool) if t==0 or mode=='immediate' else w['mover']==k
            pos=MAPS[w['source_map'],k] if t==0 else w['positions'][:,k];ii=np.flatnonzero(mask)
            out[rr[ii],t,torch.from_numpy(pos[ii]),:64]=projected[torch.from_numpy(w['photo_ids'][ii,k])]
            out[rr[ii],t,torch.from_numpy(pos[ii]),64]=1
    visual=torch.cat((out[...,:64].flatten(2),out[...,64]),-1);bits=projected.new_ones(n,2);bits[:,1]=int(mode=='immediate')
    return visual,bits

def observe(agent,x,bits,arm,erase=False):
    h=x.new_zeros(len(x),96);saved=[]
    for t in (0,1):
        slots=torch.cat((x[:,t,:384].reshape(-1,6,64),x[:,t,384:,None]),-1)
        mixed=torch.einsum('ij,bjd->bid',agent.input_transform,slots)
        z=torch.tanh(F.linear(mixed,agent.slot_phi[0].weight,agent.slot_phi[0].bias)).flatten(1)
        if t==1:h=torch.zeros_like(h) if erase else h.detach() if arm=='detach' else h
        h=agent.memory(torch.cat((z,bits[:,t,None]),-1),h);saved.append(h)
    return F.layer_norm(h,(96,),eps=1e-5),saved[0],saved[1]

def loss(agent,head,projected,w,arm,step):
    views=[frames(projected,w,m) for m in ('immediate','delayed')]
    x=torch.cat([v[0] for v in views]);bits=torch.cat([v[1] for v in views]);goal=np.tile(w['goals'],2);pos=np.tile(w['positions'],(2,1));n=len(goal)
    h,h0,h1=observe(agent,x,bits,arm);logits=head(h).reshape(n,2,6);selected=logits[torch.arange(n),torch.from_numpy(goal)]
    lp=F.log_softmax(selected,-1);action=(lp.detach().exp().cumsum(-1)<torch.from_numpy(w['action_uniform'])).sum(-1).clamp(max=5)
    target=pos[np.arange(n),goal];reward=(action.detach().numpy()==target).astype(np.float32);chosen=lp.gather(1,action[:,None]).squeeze(1)
    policy=-(chosen*torch.from_numpy(reward-.5)).mean();entropy=-(lp.exp()*lp).sum(-1).mean();weight=.02 if step<2100 else 0.;value=policy-weight*entropy
    stats=dict(loss=float(value.detach()),policy_loss=float(policy.detach()),entropy=float(entropy.detach()),entropy_coefficient=weight,mean_reward=float(reward.mean()),arm=arm,rows=n)
    trace=dict(h0=h0.detach().numpy(),raw_h1=h1.detach().numpy(),hfinal=h.detach().numpy(),full_logits=logits.detach().numpy(),goalselected_logits=selected.detach().numpy(),selected_probabilities=lp.detach().exp().numpy(),action_uniform=w['action_uniform'],action=action.detach().numpy(),reward=reward,selected_target=target)
    return value,stats,trace

def environment_check(au,bank):
    import temporal_world as production
    for p in (1,2,3):
        expected=eval_world(p,[bank.pools['test',k] for k in (0,1)])
        au.check(np.array_equal(events(p),production.events(p)) and np.array_equal(events(p,True),production.events(p,True)),'independent_legal_event_support',p)
        au.check(all(np.array_equal(v,production.evaluation_worlds(bank,p)[k]) for k,v in expected.items()),'exact_balanced_evaluation_table',p)
        e=expected;au.check(np.all(np.bincount(e['source_map'],minlength=30)[b.groups(p)['old']]==320) and np.all(np.bincount(e['target_map'],minlength=30)==192),'source_and_terminal_marginals',p)
        for group,upper,classes,pairs in [('all',.2,5,360),('old',1/3,3,108),('added',1.,1,0),('sealed',1.,1,0)]:
            by={}
            for i in np.flatnonzero(np.isin(e['target_map'],b.groups(p)[group])):
                m=e['mover'][i];key=(m,e['positions'][i,m],*e['photo_ids'][i]);by.setdefault(key,[]).append(e['positions'][i,1-m])
            counts=[np.unique(v,return_counts=True)[1] for v in by.values()]
            au.check(all(len(c)==classes and np.all(c==6) and sum(a*z for a,z in combinations(c,2))==pairs for c in counts),'same_last_frame_ambiguity_and_history_pairs',(p,group,upper))
        projected=torch.arange(60*64,dtype=torch.float32).reshape(60,64)
        for mode in ('immediate','delayed'):
            x,bits=frames(projected,e,mode);xx,bb=production.frames(projected,e,mode)
            au.check(torch.equal(x,xx) and torch.equal(bits,bb),'independent_full_or_local_visible_frame',(p,mode))

def construct(camp,prepared,seed,p,who):
    with torch.random.fork_rng(devices=[]):
        project=nn.Sequential(nn.Linear(1024,64),nn.Tanh());project.load_state_dict({k[8:]:v for k,v in prepared[who].items() if k.startswith('project.')})
        torch.manual_seed(seed*1000+701+who);agent=camp.CampAgent(project,7,2,transform=torch.eye(6))
        init=int(np.random.SeedSequence([20020,seed,p,who,0,0]).generate_state(1,np.uint64)[0]>>np.uint64(1))
        torch.manual_seed(init);head=nn.Sequential(nn.Linear(96,96),nn.Tanh(),nn.Linear(96,12))
    for k,v in agent.named_parameters():v.requires_grad_(k.startswith(('memory.','slot_phi.')))
    return agent,head,init

def state(agent,head):return dict(agent=agent.state_dict(),head=head.state_dict())
def restore(agent,head,x):agent.load_state_dict(x['agent']);head.load_state_dict(x['head'])
def parameters(agent,head):return [v for v in agent.parameters() if v.requires_grad]+list(head.parameters())

def audit(out,au):
    sys.path.insert(0,str(PROJECT/'redesign_v0.8'));import camp
    inv=read(out/'invocation.json');done=read(out/'training_complete.json');formal=inv['formal']
    sourcefiles=[ROOT/n for n in ('run_temporal.py','temporal_model.py','temporal_world.py','固定执行方案.md','环境与能力验收_独立审查.md')]+[PROJECT/n for n in ('redesign_v0.8/camp.py','redesign_v0.4/run_pilot.py','redesign_v0.4/agents.py','redesign_v0.4/resource_env.py')]
    hashes={str(p):sha(p) for p in sourcefiles};au.check(inv['source_hashes']==done['source_hashes']==hashes,'exact_frozen_source_inventory')
    for file,h in hashes.items():au.check(sha(out/'frozen_sources'/Path(file).relative_to(PROJECT))==h,'source_snapshot',file)
    au.check(inv['input_hashes']==done['input_hashes'] and all(sha(p)==h for p,h in inv['input_hashes'].items()),'data_and_previous_manifest_binding')
    au.check(done['status']=='complete' and all(sha(out/p)==h for p,h in done['files'].items()),'all_completed_artifact_hashes')
    au.check(inv['seeds']==([32101,32102,32103,32104] if formal else [99520]) and inv['partitions']==([1,2,3] if formal else [1]) and inv['arms']==['full','detach'] and inv['directions']==[0,1] and inv['updates']==2400,'preselected_complete_matrix_and_budget')
    if formal:
        gate=read(inv['preflight']['path']);au.check(gate['passed'] and gate['source_hashes']==hashes and sha(inv['preflight']['path'])==inv['preflight']['sha256'],'formal_successful_source_matched_gate')
    bank=camp.ImageBank();environment_check(au,bank);counts=au.counts;prepared_hashes=[]
    for seed in inv['seeds']:
        prepared=load(out/f'prepared_{seed}.pt');prepared_hashes.append(sha(out/f'prepared_{seed}.pt'));prep=read(out/f'preparation_{seed}.json')
        au.check(prep['seed']==seed and prep['updates_per_person']==200 and prep['batch']==64 and prep['no_source_filter'],'recorded_new_resource_preparation',seed)
        for p,who in product(inv['partitions'],inv['directions']):
            expectedworlds=[b.array_hash(world(seed,p,who,t,256,[bank.pools['train',k] for k in (0,1)])) for t in range(inv['updates'])];au.count('independent_world_steps',len(expectedworlds))
            initial_pair=[];first_trace=[];table=eval_world(p,[bank.pools['test',k] for k in (0,1)])
            for arm in inv['arms']:
                folder=out/f's{seed}_p{p}_d{who}_{arm}';cfg=read(folder/'config.json');agent,head,initseed=construct(camp,prepared,seed,p,who);initial=load(folder/'initial.pt');initial_pair.append(initial)
                au.check(same(state(agent,head),initial) and cfg['head_seed']==initseed,'independent_fresh_agent_head_initialization',folder.name)
                au.check(cfg['trainable_agent_names']==[k for k,v in agent.named_parameters() if v.requires_grad] and sum(v.numel() for v in parameters(agent,head))==cfg['trainable_parameters']==155598,'only_memory_slot_and_head_trainable',folder.name)
                au.check(cfg['source_hashes']==hashes and cfg['prepared_sha256']==sha(out/f'prepared_{seed}.pt') and cfg['projected_sha256']==sha(folder/'projected.npy') and cfg['eval_chunk']==256 and cfg['batch_events']==256,'config_source_and_batch_binding',folder.name)
                au.check((cfg['updates'],cfg['learning_rate'],cfg['gradient_clip'],cfg['entropy_on_updates'],cfg['entropy_weight'])==(2400,.0007,2.,2100,.02),'fixed_training_hyperparameters',folder.name)
                with torch.no_grad():projected=agent.project(bank.features).detach()
                au.check(np.array_equal(projected.numpy(),np.load(folder/'projected.npy')),'exact_frozen_projection',folder.name)
                logs=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()];au.check(len(logs)==inv['updates'] and [x['world_sha256'] for x in logs]==expectedworlds,'all_step_paired_world_and_action_RNG',folder.name)
                au.check([x['update'] for x in logs]==list(range(1,inv['updates']+1)) and all(x['components']['entropy_coefficient']==(.02 if i<2100 else 0.) and np.isfinite(x['gradient_norm']) for i,x in enumerate(logs)),'complete_budget_entropy_boundary',folder.name)
                au.check(not load(folder/'initial_optimizer.pt')['state'],'fresh_Adam_no_inherited_moments',folder.name)
                for step in (0,2100):
                    before=load(folder/f'checkpoint_{step:04d}.pt');restore(agent,head,before);params=parameters(agent,head);opt=torch.optim.Adam(params,lr=.0007);opt.load_state_dict(load(folder/f'optimizer_{step:04d}.pt'))
                    w=world(seed,p,who,step,256,[bank.pools['train',k] for k in (0,1)]);raw=arrays(folder/f'train_{step+1:04d}.npz')
                    au.check(all(np.array_equal(raw['world__'+k],v) for k,v in w.items()),'forensic_exact_world',(folder.name,step))
                    v=[frames(projected,w,m) for m in ('immediate','delayed')]
                    au.check(np.array_equal(raw['frames'],torch.cat([x[0] for x in v]).numpy()) and np.array_equal(raw['view_bits'],torch.cat([x[1] for x in v]).numpy()) and np.array_equal(raw['goals'],np.tile(w['goals'],2)) and np.array_equal(raw['positions'],np.tile(w['positions'],(2,1))),'forensic_legal_inputs',(folder.name,step))
                    opt.zero_grad(set_to_none=True);value,stats,trace=loss(agent,head,projected,w,arm,step)
                    au.check(all(np.array_equal(raw['trace__'+k],v) for k,v in trace.items()) and b.array_hash(trace)==logs[step]['trace_sha256'],'independent_full_trace_replay',(folder.name,step))
                    if step==0:first_trace.append(b.array_hash(trace))
                    au.check(stats==logs[step]['components'],'independent_selected_goal_loss_entropy',(folder.name,step));value.backward();norm=float(torch.nn.utils.clip_grad_norm_(params,2.))
                    au.check(norm==logs[step]['gradient_norm'] and all(v.grad is None for v in agent.parameters() if not v.requires_grad),'exact_gradient_norm_and_frozen_gradients',(folder.name,step));opt.step()
                    au.check(same(state(agent,head),load(folder/f'after_{step+1:04d}.pt')) and same(opt.state_dict(),load(folder/f'after_{step+1:04d}_optimizer.pt')),'exact_independent_parameter_and_Adam_update',(folder.name,step));au.count('exact_update_replays')
                curve=read(folder/'curve.json');au.check([x['update'] for x in curve]==cfg['checkpoints']==[0,100,300,600,1200,1800,2100,2400],'all_fixed_checkpoints',folder.name)
                for point in curve:
                    t=point['update'];cp=load(folder/f'checkpoint_{t:04d}.pt');restore(agent,head,cp);optim=load(folder/f'optimizer_{t:04d}.pt')
                    au.check(all(torch.equal(v,initial['agent'][k]) for k,v in cp['agent'].items() if not k.startswith(('memory.','slot_phi.'))),'all_checkpoint_frozen_parameters',(folder.name,t))
                    au.check((not optim['state']) if t==0 else len(optim['state'])==10 and all(float(v['step'])==t for v in optim['state'].values()),'checkpoint_Adam_steps',(folder.name,t))
                    au.check(len(optim['param_groups'])==1 and all((g['lr'],tuple(g['betas']),g['eps'],g['weight_decay'],g['amsgrad'])==(.0007,(.9,.999),1e-8,0,False) for g in optim['param_groups']),'checkpoint_fixed_Adam_rule',(folder.name,t))
                    for mode in ('immediate','delayed')+ (('delayed_erase',) if t==2400 else ()):
                        raw=arrays(folder/f'evaluation_{t:04d}_{mode}.npz');au.check(all(np.array_equal(raw[k],v) for k,v in table.items()),'complete_evaluation_world_photo_table',(folder.name,t,mode))
                        with torch.no_grad():
                            for lo in (0,5632):
                                sub={k:v[lo:lo+256] for k,v in table.items()};x,bits=frames(projected,sub,'immediate' if mode=='immediate' else 'delayed');h,_,_=observe(agent,x,bits,arm,mode=='delayed_erase');z=head(h).reshape(-1,2,6).numpy()
                                au.check(np.array_equal(z,raw['logits'][lo:lo+256]),'original_chunk_shape_sampled_evaluation_forward',(folder.name,t,mode,lo));au.count('evaluation_replayed_rows',len(z))
                        au.count('complete_evaluation_tables')
                au.check(same(load(folder/'final.pt'),load(folder/'checkpoint_2400.pt')) and same(load(folder/'final_optimizer.pt'),load(folder/'optimizer_2400.pt')),'final_exact_last_checkpoint',folder.name);au.count('fits');au.count('logged_updates',len(logs))
            au.check(same(*initial_pair),'full_detach_identical_complete_initial_state',(seed,p,who))
            au.check(len(set(first_trace))==1,'full_detach_identical_initial_forward_action_reward',(seed,p,who))
    au.check(len(set(prepared_hashes))==len(prepared_hashes),'distinct_new_source_preparations')
    au.check(counts['fits']==done['head_fits'] and counts['logged_updates']==done['private_updates'] and done['selected_goal_actions']==counts['logged_updates']*512,'complete_batch_totals')
    return hashes

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path);ap.add_argument('--preflight',action='store_true');args=ap.parse_args();torch.set_num_threads(1)
    out=(args.out or ROOT/'results/smoke_002' if args.preflight else args.out or ROOT/'results/temporal_001').resolve();au=Audit();hashes={}
    try:hashes=audit(out,au)
    except Exception:au.check(False,'auditor_exception',traceback.format_exc())
    result=dict(passed=not au.failures,source_hashes=hashes,script_sha256=sha(__file__),helper_sha256=sha(PROJECT/'redesign_v0.19/audit_readout.py'),checks=au.checks,failures=au.failures,counts=au.counts,output=str(out),new_training=0,production_loss_or_observe_called=False,
        scope='All source/artifact bindings, legal world/photo tables and training RNG hashes; common initial states and frozen checkpoint tensors; independent first/2101 exact loss/gradient/Adam replay; first256 and last128 evaluation rows per mode/checkpoint at original batch shape.',
        exclusions='No full re-training, no independent replay of resource-preparation optimization, no claim that every update or evaluation forward was replayed. Full raw metric/aggregation recount belongs to the separate analyzer.')
    target=ROOT/'preflight_qa.json' if args.preflight else out/'audit_execution.json';write(target,result);(out/'audit_temporal_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes',)},ensure_ascii=False),flush=True)
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
