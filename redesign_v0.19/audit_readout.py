"""Independent v19 readout auditor; does not import the production loss/fixture.

Only old frozen-model loading may be reused. This file's finalized CLI and
receipt are added once the root runner's saved-file schema is available.
"""
from pathlib import Path
from itertools import product,permutations
from functools import lru_cache
from datetime import datetime,timezone
import argparse,hashlib,json,sys,traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
MAPS=np.asarray(list(permutations(range(6),2)),np.int64)
PERMS=np.asarray(list(permutations(range(6))),np.int64)
MATCHINGS=(((0,1),(2,3),(4,5)),((0,2),(1,4),(3,5)),((0,3),(1,5),(2,4)))
HEAD_NAMES=('0.weight','0.bias','2.weight','2.bias')
HEAD_SHAPES=((96,96),(96,),(12,96),(12,))
WORLD_KEYS=('triple_id','source_map','target_map','permutation','positions','photo_ids','goals','action_uniform')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p):return torch.load(p,weights_only=True)
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def array_hash(data,keys=None):
    h=hashlib.sha256()
    for k in sorted(data) if keys is None else keys:
        x=np.ascontiguousarray(data[k]);h.update(k.encode());h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
    return h.hexdigest()

def same(x,y):
    if torch.is_tensor(x):return torch.is_tensor(y) and x.dtype==y.dtype and x.shape==y.shape and torch.equal(x,y)
    if isinstance(x,dict):return isinstance(y,dict) and x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
    if isinstance(x,(tuple,list)):return isinstance(y,(tuple,list)) and len(x)==len(y) and all(same(a,b) for a,b in zip(x,y))
    return type(x)==type(y) and x==y

class Audit:
    def __init__(self):self.checks=0;self.failures=[];self.counts={};self.max_errors={}
    def check(self,truth,label,context=None):
        self.checks+=1
        if not bool(truth):self.failures.append(dict(check=label,context=context))
    def close(self,x,y,label,context=None,atol=2e-12,rtol=2e-12):
        x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
        err=float(np.max(np.abs(x-y),initial=0)) if x.shape==y.shape else 1e300
        self.max_errors[label]=max(err,self.max_errors.get(label,0.))
        self.check(x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all() and np.allclose(x,y,atol=atol,rtol=rtol),label,context)
    def count(self,k,n=1):self.counts[k]=self.counts.get(k,0)+n

def groups(p):
    def matching(i):
        pairs={(a,b) for a,b in MATCHINGS[i]}|{(b,a) for a,b in MATCHINGS[i]}
        return np.asarray([i for i,x in enumerate(MAPS) if tuple(x) in pairs],np.int64)
    added=matching(p-1);sealed=matching(p%3);old=np.setdiff1d(np.arange(30),np.r_[added,sealed])
    return dict(old=old,added=added,sealed=sealed,all=np.arange(30))

@lru_cache(None)
def triples(p):
    allowed=set(groups(p)['old']);lookup={tuple(x):i for i,x in enumerate(MAPS)};rows=[]
    for source in sorted(allowed):
        for gi,g in enumerate(PERMS):
            target=lookup[tuple(g[MAPS[source]])]
            if target in allowed:rows.append((source,gi,target))
    out=np.asarray(rows,np.int64);assert out.shape==(7776,3)
    return out

def sequence(seed,p,who,purpose,step=0):return np.random.SeedSequence([19019,seed,p,who,purpose,step])
def fresh_head_seed(seed,p,who):return int(sequence(seed,p,who,0).generate_state(1,np.uint64)[0]>>np.uint64(1))
def fresh_head(seed,p,who):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(fresh_head_seed(seed,p,who));return nn.Sequential(nn.Linear(96,96),nn.Tanh(),nn.Linear(96,12))

def world(seed,p,who,step,batch,pools):
    rng=np.random.Generator(np.random.PCG64(sequence(seed,p,who,1,step)))
    index=rng.integers(7776,size=batch);selected=triples(p)[index]
    photos=np.stack([rng.choice(pools[k],size=batch) for k in (0,1)],axis=1)
    goals=rng.integers(2,size=batch)
    action=np.random.Generator(np.random.PCG64(sequence(seed,p,who,2,step))).random((2*batch,1)).astype(np.float32)
    return dict(triple_id=index,source_map=selected[:,0],target_map=selected[:,2],permutation=PERMS[selected[:,1]],
      positions=np.stack((MAPS[selected[:,0]],MAPS[selected[:,2]]),1),photo_ids=photos,goals=goals,action_uniform=action)

def reconstruct_loss(head,h,w,signal,step):
    """Standard library loss primitives, independently assembled, exact graph order."""
    n=len(w['goals']);goal=np.r_[w['goals'],w['goals']];position=np.concatenate((w['positions'][:,0],w['positions'][:,1]))
    logits=head(h).reshape(2*n,2,6);selected=logits[torch.arange(2*n),torch.from_numpy(goal)]
    lp=F.log_softmax(selected,-1)
    action=(lp.detach().exp().cumsum(-1)<torch.from_numpy(w['action_uniform'])).sum(-1).clamp(max=5)
    truth=position[np.arange(2*n),goal];reward=(action.detach().numpy()==truth).astype(np.float32)
    chosen=lp.gather(1,action[:,None]).squeeze(1)
    rl=-(chosen*torch.from_numpy(reward-.5)).mean()
    ce=F.cross_entropy(selected,torch.from_numpy(truth))
    entropy=-(lp.exp()*lp).sum(-1).mean();weight=.02 if step<2100 else 0.
    task=rl if signal=='reward' else ce;loss=task-weight*entropy
    stats=dict(loss=float(loss.detach()),task_loss=float(task.detach()),reward_loss=float(rl.detach()),ce_loss=float(ce.detach()),
      entropy=float(entropy.detach()),entropy_coefficient=weight,mean_reward=float(reward.mean()),signal=signal,rows=2*n,batch_pairs=n)
    trace=dict(full_logits=logits.detach().numpy(),goalselected_logits=selected.detach().numpy(),selected_probabilities=lp.detach().exp().numpy(),
      action_uniform=w['action_uniform'],action=action.detach().numpy(),reward=reward,selected_target=truth,flat_goals=goal,flat_positions=position)
    return loss,stats,trace

def independent_metrics(logits,positions,map_ids,p):
    x=np.asarray(logits,np.float64);logp=x-x.max(-1,keepdims=True);logp-=np.log(np.exp(logp).sum(-1,keepdims=True));prob=np.exp(logp)
    chosen=x.argmax(-1);success=chosen==positions
    expected=np.take_along_axis(prob,positions[:,:,None],-1).squeeze(-1)
    entropy=-np.sum(prob*logp,-1);ties=(x==x.max(-1,keepdims=True)).sum(-1)>1
    out={}
    for g,maps in groups(p).items():
        mask=np.isin(map_ids,maps);good=success[mask]
        out[g]=dict(worlds=int(mask.sum()),correct_both=int(good.all(-1).sum()),correct_goals=int(good.sum()),J=float(good.all(-1).mean()),
          single=float(good.mean()),food=float(good[:,0].mean()),water=float(good[:,1].mean()),stochastic_J=float(expected[mask].prod(-1).mean()),
          stochastic_single=float(expected[mask].mean()),policy_entropy=float(entropy[mask].mean()),food_entropy=float(entropy[mask,0].mean()),
          water_entropy=float(entropy[mask,1].mean()),exact_max_tie_goal_count=int(ties[mask].sum()),exact_max_tie_rate=float(ties[mask].mean()),
          food_tie_rate=float(ties[mask,0].mean()),water_tie_rate=float(ties[mask,1].mean()))
    return out

def module_sha(m):
    d=hashlib.sha256()
    for k,v in m.state_dict().items():d.update(k.encode());d.update(v.detach().cpu().numpy().tobytes())
    return d.hexdigest()

def state_digest(state):
    d=hashlib.sha256()
    for k,v in state.items():d.update(k.encode());d.update(v.detach().cpu().numpy().tobytes())
    return d.hexdigest()

def raw_array_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def expected_table(p,split,entries):
    pools=[np.asarray([i for i,e in enumerate(entries) if e['split']==split and e['category']==category],np.int64) for category in ('food','water')]
    pool=groups(p)['old'] if split=='train' else np.arange(30)
    pairs=np.asarray(list(product(*pools)),np.int64);ids=np.tile(pairs,(len(pool),1));mids=np.repeat(pool,len(pairs))
    return dict(map_ids=mids,positions=MAPS[mids],photo_ids=ids),pools

@torch.no_grad()
def observer_replay(agent,projected,table,scale):
    out=[];raw_out=[]
    for start in range(0,len(table['map_ids']),512):
        pos=table['positions'][start:start+512];ids=table['photo_ids'][start:start+512];n=len(pos)
        slots=projected.new_zeros(n,6,65)
        for k in (0,1):
            slots[torch.arange(n),torch.from_numpy(pos[:,k]),:64]=projected[torch.from_numpy(ids[:,k])]
            slots[torch.arange(n),torch.from_numpy(pos[:,k]),64]=1.
        mixed=torch.einsum('ij,bjd->bid',agent.input_transform,slots)
        encoded=torch.tanh(F.linear(mixed,agent.slot_phi[0].weight,agent.slot_phi[0].bias)).flatten(1)
        seen=torch.cat((encoded,encoded.new_ones(n,1)),1)
        raw=agent.memory(seen,seen.new_zeros(n,96))
        raw_out.append(raw.numpy());out.append((raw*raw.new_tensor(scale)).numpy())
    return np.concatenate(raw_out),np.concatenate(out)


def source_and_cache(out,inv,au):
    sys.path.insert(0,str(PROJECT/'redesign_v0.15'))
    import run_scaled as old
    source=Path(inv['source']);scaled=Path(inv['scaled_source']);formal=inv['formal']
    expected=(PROJECT/'redesign_v0.13/results/spatial_001',PROJECT/'redesign_v0.15/results/scaled_001') if formal else (PROJECT/'redesign_v0.13/results/smoke_002',PROJECT/'redesign_v0.15/results/smoke_002')
    au.check((source,scaled)==expected,'preselected_exact_inherited_sources')
    inherited=read(scaled/'invocation.json')['source_hashes'];hashes={**inherited,**{str(ROOT/n):sha(ROOT/n) for n in ('run_readout.py','readout_interface.py','固定执行方案.md','前置审查.md')}}
    au.check(hashes==inv['source_hashes'],'exact_production_source_inventory')
    for file,digest in hashes.items():au.check(sha(file)==digest and sha(out/'frozen_sources'/Path(file).relative_to(PROJECT))==digest,'frozen_source_and_archive_hash',file)
    for file,digest in inv['input_hashes'].items():au.check(sha(file)==digest,'consumed_input_hash',file)
    if formal:
        for parent in (source,scaled):
            m=read(parent/'completion_manifest.json');au.check(m['status']=='complete','historical_source_manifest_complete',str(parent))
            for file,digest in inv['input_hashes'].items():
                f=Path(file)
                if f.is_relative_to(parent) and f.name!='completion_manifest.json':au.check(m['artifacts'].get(str(f.relative_to(PROJECT)))==digest,'consumed_original_source_in_manifest',file)
        gate=inv['preflight'];g=read(gate['path']);au.check(g['passed'] and g['source_hashes']==hashes and sha(gate['path'])==gate['sha256'],'formal_has_exact_successful_gate')
    bank=old.ImageBank();entries=bank.entries;caches={};cache_records={}
    for seed,p in product(inv['seeds'],inv['partitions']):
        prepared=load(source/f'prepared_{seed}.pt');au.check(same(prepared,load(scaled/f'prepared_{seed}.pt')),'same_resource_projection_preparation',(seed,p))
        private=source/f'private_s{seed}_p{p}_control';initial=load(private/'initial.pt')['agents'];final=load(private/'final.pt')['agents']
        retained=load(source/f'social_s{seed}_p{p}_control/initial.pt');reset=load(scaled/f'social_s{seed}_p{p}_reset_scaled/initial.pt')
        transfer=load(private/'transferred.pt');au.check(same(retained,transfer),'retained_social_start_exact_private_transfer',(seed,p))
        cal=read(scaled/'calibration_receipt.json');cfg=read(Path(cal['path']).parent/f's{seed}_p{p}/calibration.json')
        au.check(sha(cal['path'])==cal['sha256'] and sha(Path(cal['path']).parent/f's{seed}_p{p}/calibration.json')==cal['files'][f's{seed}_p{p}/calibration.json'],'inherited_exact_calibration_not_refit',(seed,p))
        cr=read(out/f'cache_s{seed}_p{p}/cache_receipt.json')
        au.check(cr['chunk']==512 and cr['dtype']=='float32' and cr['all_parameters_frozen'] and cr['grad_enabled'] is False and cr['new_dino_inferences']==0,'uniform_frozen_cache_flags',(seed,p))
        rec={(r['interface'],r['direction']):r for r in cr['records']};au.check(len(rec)==4,'four_interface_person_cache_records',(seed,p))
        agents=old.remake_agents(seed,prepared,7,2,'identity')
        for who in (0,1):
            au.check(all(torch.equal(retained[who][k],final[who][k] if k.startswith(('memory.','slot_phi.')) else initial[who][k]) for k in retained[who]),'retained_only_private_visual_learning_transferred',(seed,p,who))
            au.check(all(torch.equal(reset[who][k],v) for k,v in initial[who].items()) and set(reset[who])==set(initial[who])|{'visual_scale'},'reset_exact_private_initial_plus_only_scale',(seed,p,who))
            au.check(float(reset[who]['visual_scale'])==cfg['persons'][who]['alpha_float32'],'once_inherited_float32_scale',(seed,p,who))
            for interface,state,scale in [('retained',retained[who],1.),('reset_scaled',reset[who],float(reset[who]['visual_scale']))]:
                agent=agents[who];agent.load_state_dict({k:v for k,v in state.items() if k!='visual_scale'},strict=True);agent.requires_grad_(False);agent.train()
                snapshot=module_sha(agent)
                with torch.no_grad():projected=agent.project(bank.features)
                r=rec[interface,who];expected_source=(source/f'social_s{seed}_p{p}_control/initial.pt' if interface=='retained' else scaled/f'social_s{seed}_p{p}_reset_scaled/initial.pt')
                au.check(r['source_checkpoint']==str(expected_source) and r['source_sha256']==sha(expected_source) and r['state_sha256']==state_digest(state) and r['scale']==scale,'cache_bound_to_complete_original_state',(seed,p,who,interface))
                for split in ('train','test'):
                    table,pools=expected_table(p,split,entries);item=r['entries'][split];path=out/item['file'];data=arrays(path)
                    au.check(item['sha256']==sha(path) and item['worlds']==len(table['map_ids']) and item['photo_pools']==[x.tolist() for x in pools],'cache_manifest_hash_and_full_support',(seed,p,who,interface,split))
                    au.check(all(np.array_equal(data[k],v) for k,v in table.items()),'independent_full_map_photo_order',(seed,p,who,interface,split))
                    raw_h,h=observer_replay(agent,projected,table,scale)
                    au.close(data['h'],h,'independent_all_cache_observer_replay',(seed,p,who,interface,split),atol=0,rtol=0)
                    if 'raw_h' in data:au.close(data['raw_h'],raw_h,'independent_unscaled_observer_cache',(seed,p,who,interface,split),atol=0,rtol=0)
                    au.check(raw_array_sha(data['h'])==item['h_sha256'],'cache_effective_h_sha',(seed,p,who,interface,split))
                    au.close(item['mean_l2'],np.linalg.norm(h.astype(np.float64),axis=1).mean(),'cache_mean_scale_statistic',(seed,p,who,interface,split))
                    caches[seed,p,who,interface,split]=data;cache_records[seed,p,who,interface,split]=item
                    au.count('cache_tables');au.count('cache_worlds',len(h))
                au.check(module_sha(agent)==snapshot and all(v.grad is None for v in agent.parameters()),'cache_audit_no_interface_weight_or_gradient_mutation',(seed,p,who,interface))
    return hashes,caches,entries


def cache_lookup(cache):
    return {(int(m),int(f),int(w)):i for i,(m,(f,w)) in enumerate(zip(cache['map_ids'],cache['photo_ids']))}

def index_world(w,lookup):
    return np.asarray([lookup[int(m),int(f),int(wa)] for key in ('source_map','target_map') for m,(f,wa) in zip(w[key],w['photo_ids'])],np.int64)


def replay_update(folder,cfg,cache,w,ix,step,record,au):
    head=fresh_head(cfg['seed'],cfg['partition'],cfg['direction']);start=0 if step==0 else 2100
    before=folder/('initial.pt' if start==0 else f'checkpoint_{start:04d}.pt')
    oldopt=folder/('initial_optimizer.pt' if start==0 else f'optimizer_{start:04d}.pt')
    head.load_state_dict(load(before));opt=torch.optim.Adam(head.parameters(),lr=.0007);opt.load_state_dict(load(oldopt))
    opt.zero_grad(set_to_none=True)
    h=torch.from_numpy(cache['h'][ix]).detach();loss,stats,trace=reconstruct_loss(head,h,w,cfg['signal'],step)
    saved=arrays(folder/f'train_{step+1:04d}.npz')
    au.check(all(np.array_equal(saved['world__'+k],v) for k,v in w.items()) and np.array_equal(saved['cache_indices'],ix),'forensic_world_and_cache_indices_exact',[folder.name,step+1])
    for key,value in trace.items():au.close(saved['trace__'+key],value,'forensic_head_trace_exact',[folder.name,step+1,key],atol=0,rtol=0)
    for key,value in stats.items():
        if isinstance(value,(int,float)):au.close(record['components'][key],value,'forensic_loss_scalar',[folder.name,step+1,key],atol=0,rtol=0)
        else:au.check(record['components'][key]==value,'forensic_signal_label',[folder.name,step+1,key])
    au.check(array_hash(trace)==record['trace_sha256'],'forensic_trace_sha_exact',[folder.name,step+1])
    # Formula check in float64 from saved raw logits is independent of autograd
    # graph-order requirements for the subsequent bitwise Adam replay.
    x=trace['goalselected_logits'].astype(np.float64);lp=x-x.max(-1,keepdims=True);lp-=np.log(np.exp(lp).sum(-1,keepdims=True))
    ce=-lp[np.arange(len(x)),trace['selected_target']].mean()
    rl=-(lp[np.arange(len(x)),trace['action']]*(trace['reward']-.5)).mean()
    entropy=-np.sum(np.exp(lp)*lp,-1).mean()
    au.close([ce,rl,entropy],[stats['ce_loss'],stats['reward_loss'],stats['entropy']],'independent_NumPy_selected_goal_loss',[folder.name,step+1],atol=2e-6,rtol=2e-6)
    loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),2.));opt.step()
    au.close(norm,record['gradient_norm'],'exact_forensic_preclip_gradient_norm',[folder.name,step+1],atol=0,rtol=0)
    au.check(same(head.state_dict(),load(folder/f'after_{step+1:04d}.pt')),'exact_forensic_post_update_parameters',[folder.name,step+1])
    au.check(same(opt.state_dict(),load(folder/f'after_{step+1:04d}_optimizer.pt')),'exact_forensic_post_update_Adam',[folder.name,step+1])
    au.check(not h.requires_grad and h.grad is None and len(opt.param_groups)==1 and len(opt.param_groups[0]['params'])==4,'only_four_head_tensors_in_optimizer',[folder.name,step+1])
    au.count('exact_update_replays')


def runner_metrics(logits,table,p):
    own=independent_metrics(logits,table['positions'],table['map_ids'],p)
    x=logits.astype(np.float64);lp=x-x.max(-1,keepdims=True);lp-=np.log(np.exp(lp).sum(-1,keepdims=True))
    truth=lp[np.arange(len(x))[:,None],np.arange(2)[None,:],table['positions']]
    result={}
    for group,pool in {**groups(p),'common30':np.arange(30)}.items():
        v=own['all' if group=='common30' else group];mask=np.isin(table['map_ids'],pool)
        result[group]=dict(n=v['worlds'],J=v['J'],single=v['single'],Q=v['stochastic_J'],native_single=v['stochastic_single'],entropy=v['policy_entropy'],nll=float(-truth[mask].mean()),correct_joint=v['correct_both'],correct_goals=v['correct_goals'],
          exact_max_tie_goal_count=v['exact_max_tie_goal_count'],exact_max_tie_rate=v['exact_max_tie_rate'])
    return result


def audit_fits(out,inv,caches,entries,au):
    updates=inv['updates'];pools=[np.asarray([i for i,e in enumerate(entries) if e['split']=='train' and e['category']==k]) for k in ('food','water')]
    for seed,p,who in product(inv['seeds'],inv['partitions'],(0,1)):
        init=fresh_head(seed,p,who).state_dict();folders=[];configs=[];logs=[]
        for interface,signal in product(('retained','reset_scaled'),('reward','ce')):
            folder=out/f's{seed}_p{p}_d{who}_{interface}_{signal}';cfg=read(folder/'config.json')
            records=[json.loads(s) for s in (folder/'training.jsonl').read_text().splitlines()]
            folders.append(folder);configs.append(cfg);logs.append(records)
            au.check(same(load(folder/'initial.pt'),init) and same(load(folder/'checkpoint_0000.pt'),init),'all_conditions_same_new_head_init',folder.name)
            au.check(cfg['initial_sha256']==state_digest(init) and cfg['head_initialization_seed']==fresh_head_seed(seed,p,who),'initial_hash_from_new_namespace',folder.name)
            au.check(len(records)==updates and [r['update'] for r in records]==list(range(1,updates+1)),'complete_fixed_update_log',folder.name)
            au.check(cfg['updates']==updates and cfg['batch_pairs']==512 and cfg['worlds_per_update']==1024 and cfg['learning_rate']==.0007 and cfg['gradient_clip']==2 and cfg['only_head_trainable'] and cfg['correct_labels_available']==(signal=='ce') and not cfg['new_communication_training'],'fixed_training_and_label_budget',folder.name)
            au.check(cfg['training_pool']==groups(p)['old'].tolist() and cfg['test_world_count']==1920,'train_old_only_test_full_balanced_support',folder.name)
            cr=out/f'cache_s{seed}_p{p}/cache_receipt.json';au.check(Path(cfg['cache_receipt'])==cr and cfg['cache_receipt_sha256']==sha(cr) and cfg['source_hashes']==inv['source_hashes'],'fit_uses_exact_cache_and_sources',folder.name)
        train=caches[seed,p,who,'retained','train'];lookup=cache_lookup(train)
        for step in range(updates):
            w=world(seed,p,who,step,512,pools);ix=index_world(w,lookup);wh=array_hash(w);ih=raw_array_sha(ix)
            for folder,cfg,records in zip(folders,configs,logs):
                r=records[step];comp=r['components']
                au.check(r['world_sha256']==wh and r['cache_indices_sha256']==ih,'all_steps_same_independent_paired_worlds_and_cache_indexes',[folder.name,step+1])
                au.check(comp['entropy_coefficient']==(.02 if step<2100 else 0.) and comp['rows']==1024 and comp['batch_pairs']==512 and comp['signal']==cfg['signal'],'all_steps_selected_goal_signal_and_entropy_schedule',[folder.name,step+1])
                expected=comp['reward_loss'] if cfg['signal']=='reward' else comp['ce_loss']
                au.close(comp['task_loss'],expected,'logged_only_selected_task_loss',[folder.name,step+1],atol=0,rtol=0)
                au.close(r['clip_coefficient'],min(1.,2./(r['gradient_norm']+1e-6)),'logged_gradient_clip_coefficient',[folder.name,step+1],atol=0,rtol=0)
                if step in (0,2100):replay_update(folder,cfg,caches[seed,p,who,cfg['interface'],'train'],w,ix,step,r,au)
            au.count('unique_world_steps');au.count('paired_head_updates',4)
        for folder,cfg in zip(folders,configs):
            result=read(folder/'result.json');curve=read(folder/'curve.json');test=caches[seed,p,who,cfg['interface'],'test']
            times=sorted({0,updates,*[t for t in (0,100,300,600,1200,1800,2100,2400) if t<updates]})
            au.check(cfg['checkpoints']==times and [r['update'] for r in curve]==times,'all_fixed_checkpoints_and_evaluation_times',folder.name)
            au.check(result['status']=='complete' and result['simulated_selected_goal_actions']==updates*1024 and result['ce_labels']==(updates*1024 if cfg['signal']=='ce' else 0) and result['only_head_trained'] and result['no_communication'],'fit_terminal_budget_and_scope',folder.name)
            for row,t in zip(curve,times):
                head=fresh_head(seed,p,who);state=load(folder/f'checkpoint_{t:04d}.pt');head.load_state_dict(state)
                opt=load(folder/f'optimizer_{t:04d}.pt')
                au.check(list(state)==list(HEAD_NAMES) and [tuple(v.shape) for v in state.values()]==list(HEAD_SHAPES),'only_readout_checkpoint_parameters',[folder.name,t])
                au.check(len(opt['param_groups'])==1 and len(opt['param_groups'][0]['params'])==4 and opt['param_groups'][0]['lr']==.0007 and (len(opt['state'])==4 if t else len(opt['state'])==0),'head_only_fresh_Adam_budget',[folder.name,t])
                if t:au.check(all(int(v['step'])==t for v in opt['state'].values()),'Adam_step_matches_update',[folder.name,t])
                file=folder/f'evaluation_{t:04d}.npz';raw=arrays(file)
                au.check(row['raw_sha256']==sha(file) and row['head_sha256']==state_digest(state),'evaluation_bound_to_checkpoint',[folder.name,t])
                au.check(all(np.array_equal(raw[k],test[k]) for k in ('map_ids','positions','photo_ids')),'all_evaluation_worlds_exact_common_cache',[folder.name,t])
                with torch.no_grad():pred=torch.cat([head(torch.from_numpy(test['h'][lo:lo+512])).reshape(-1,2,6) for lo in range(0,len(test['h']),512)]).numpy()
                au.close(raw['logits'],pred,'all_evaluation_logits_exact_head_replay',[folder.name,t],atol=0,rtol=0)
                stats=runner_metrics(raw['logits'],test,p)
                for group,metrics in row['scores'].items():
                    for name,value in metrics.items():au.close(value,stats[group][name],'all_raw_evaluation_metrics_independent_NumPy',[folder.name,t,group,name])
                au.count('checkpoint_evaluations');au.count('evaluation_worlds',len(pred))
            au.check(same(load(folder/'final.pt'),load(folder/f'checkpoint_{updates:04d}.pt')) and result['scores']==curve[-1]['scores'],'final_state_scores_exact_last_checkpoint',folder.name)
            au.check(result['final_sha256']==state_digest(load(folder/'final.pt')),'final_state_hash',folder.name)
            au.count('head_fits')
            print(json.dumps(dict(audited=folder.name,failures=len(au.failures))),flush=True)


def synthetic_checks(au):
    seeds=[fresh_head_seed(s,p,d) for s,p,d in product((31101,31102,31103,31104),(1,2,3),(0,1))]
    au.check(len(set(seeds))==24 and not (set(seeds)&{s*1000+881+d for s,d in product((31101,31102,31103,31104),(0,1))}),'new_head_seed_unique_and_not_old_private_seed')
    for p in (1,2,3):
        t=triples(p);au.check(all(np.bincount(t[:,j],minlength=30)[groups(p)['old']].tolist()==[432]*18 for j in (0,2)),'independent_7776_uniform_old_map_marginal',p)
    head=fresh_head(99513,1,0);h=torch.from_numpy(np.random.default_rng(91619).normal(size=(64,96)).astype(np.float32));w=world(99513,1,0,0,32,[np.arange(22),np.arange(22,44)])
    ce,_,_=reconstruct_loss(head,h,w,'ce',0);g1=torch.autograd.grad(ce,list(head.parameters()))
    changed=dict(w,action_uniform=1-w['action_uniform']);ce2,_,_=reconstruct_loss(head,h,changed,'ce',0);g2=torch.autograd.grad(ce2,list(head.parameters()))
    au.check(float(ce.detach())==float(ce2.detach()) and all(torch.equal(a,b) for a,b in zip(g1,g2)),'CE_independent_of_sampled_actions')
    for signal in ('ce','reward'):
        for step in (0,2099,2100,2399):
            loss,stats,trace=reconstruct_loss(head,h,w,signal,step)
            au.check(stats['entropy_coefficient']==(.02 if step<2100 else 0.) and torch.isfinite(loss),'independent_entropy_transition',(signal,step))
    uniform=np.zeros((30,2,6));metrics=runner_metrics(uniform,dict(positions=MAPS,map_ids=np.arange(30)),1)
    au.close(metrics['all']['Q'],1/36,'independent_uniform_two_action_baseline')
    au.check(metrics['all']['J']==0 and metrics['all']['exact_max_tie_rate']==1,'uniform_greedy_tie_rule_not_random_policy')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    torch.set_num_threads(1);out=args.out.resolve();au=Audit();hashes={};fatal=None
    try:
        inv=read(out/'invocation.json');done=read(out/'training_complete.json')
        au.check(done['status']=='complete' and done['formal']==inv['formal'],'complete_declared_batch')
        for file,digest in done['files'].items():au.check(sha(out/file)==digest,'completion_artifact_hash',file)
        au.check(done['source_hashes']==inv['source_hashes'] and done['input_hashes']==inv['input_hashes'],'completion_invocation_sources_consistent')
        au.check(inv['new_dino_inferences']==inv['new_interface_training']==inv['new_communication_training']==0,'no_DINO_interface_or_social_learning')
        synthetic_checks(au);hashes,caches,entries=source_and_cache(out,inv,au);audit_fits(out,inv,caches,entries,au)
        expected=len(inv['seeds'])*len(inv['partitions'])*2*2*2
        au.check(done['head_fits']==au.counts['head_fits']==expected and done['head_updates']==expected*inv['updates'],'all_declared_fits_updates_complete')
        au.check(done['simulated_selected_goal_actions']==expected*inv['updates']*1024 and done['supervised_labels']==expected*inv['updates']*512,'full_action_and_CE_supervision_budget')
        if args.preflight:
            au.check(not inv['formal'] and inv['updates']==2400,'preflight_full_budget_development_only')
            inherited=read(PROJECT/'redesign_v0.15/results/scaled_001/invocation.json')['source_hashes'];formalhash={**inherited,**{str(ROOT/n):sha(ROOT/n) for n in ('run_readout.py','readout_interface.py','固定执行方案.md','前置审查.md')}}
            au.check(hashes==formalhash,'development_and_formal_production_sources_identical')
    except Exception:
        fatal=traceback.format_exc();au.check(False,'audit_exception',fatal)
    source=out/'audit_readout_source.py'
    if source.exists() and sha(source)!=sha(__file__):source.rename(out/f'audit_readout_source_{sha(source)[:12]}.py')
    source.write_bytes(Path(__file__).read_bytes())
    receipt=dict(passed=not au.failures,created_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),output=str(out),
      source_hashes=hashes,checks=au.checks,failures=au.failures,counts=au.counts,max_errors=au.max_errors,
      scope='All frozen sources, complete train/test cache observer replay, paired exogenous world/index streams, fresh head initializations, head-only checkpoints/Adam counts, exact first/2101 update replay, all checkpoint logits and private action metrics.',
      exclusions='No new communication or DINO training, no claim of absent representation information or new independent source seeds; final four-source effect analysis is independently reviewed after this execution audit.',
      production_loss_or_fixture_called=False,new_training=0)
    path=ROOT/'preflight_qa.json' if args.preflight else out/'audit_execution.json'
    if path.exists():path.with_name(path.stem+'_previous_'+sha(path)[:12]+'.json').write_bytes(path.read_bytes())
    write(path,receipt)
    if args.preflight:write(out/'audit_preflight.json',receipt)
    print(json.dumps(dict(passed=receipt['passed'],checks=au.checks,failures=len(au.failures),counts=au.counts,receipt=str(path))),flush=True)
    if fatal:print(fatal,flush=True)
    if au.failures:raise SystemExit(1)

if __name__=='__main__':main()
