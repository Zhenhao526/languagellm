"""Independent saved-state, world and outcome audit; does not train or infer models."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
from itertools import permutations,product
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
V8=ROOT.parent/'redesign_v0.8'
SOURCE=V8/'results/complementarity_001'
SEEDS=(27101,27102,27103,27104)
ARMS=('sender_only','receiver_only','both')
SENDER=('slot_phi.','memory.','send_')
RECEIVER=('receive_embedding.','actor.','receive_value.')
MAPS=np.asarray(list(permutations(range(6),2)),np.int64)
MATCHINGS={1:((0,1),(2,3),(4,5)),2:((0,2),(1,4),(3,5)),3:((0,3),(1,5),(2,4))}
MODES=('normal','shuffle','blank','stochastic','erase_memory')


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return torch.load(path,weights_only=True,map_location='cpu')
def state_hash(states):
    h=hashlib.sha256()
    for who,state in enumerate(states):
        for k,v in sorted(state.items()):h.update(f'{who}/{k}'.encode());h.update(v.numpy().tobytes())
    return h.hexdigest()
def same(a,b,keys=None):
    return len(a)==len(b) and all(set(x)==set(y) and all(torch.equal(x[k],y[k]) for k in (keys or x))
                                  for x,y in zip(a,b))
def old_new(split):
    held={p for a,b in MATCHINGS[split] for p in ((a,b),(b,a))}
    return tuple(i for i,p in enumerate(MAPS) if tuple(p) not in held),tuple(i for i,p in enumerate(MAPS) if tuple(p) in held)
def seed_for(seed,split,purpose,step=0):
    # Same declared SeedSequence recipe, independent from the experiment module.
    return int(np.random.SeedSequence([9009,seed,split,purpose,step]).generate_state(1,dtype=np.uint64)[0])//2

def active_key(key,arm):
    return (arm in ('sender_only','both') and key.startswith(SENDER)) or (arm in ('receiver_only','both') and key.startswith(RECEIVER))
def metrics(counts):
    a,b,c,d=(int(counts[k]) for k in ('00','01','10','11'))
    n=a+b+c+d;single=b+c+2*d;total=(b+c)/4+d
    return dict(n=n,decisions=2*n,reward_sum=total,mean_reward=total/n if n else None,
        reward_variance=((b+c)/16+d)/n-(total/n)**2 if n else None,positive_rewards=b+c+d,
        single_correct=single,single_accuracy=single/(2*n) if n else None,both_correct=d,both_accuracy=d/n if n else None,
        food_correct=c+d,water_correct=b+d,outcome_counts=dict(zip(('00','01','10','11'),(a,b,c,d))))
def metric_match(actual,expected):
    for k,v in expected.items():
        if isinstance(v,float):
            if not np.isclose(actual.get(k,np.nan),v,rtol=0,atol=1e-12):return False
        elif actual.get(k)!=v:return False
    return True


class Audit:
    def __init__(self):self.counts=Counter();self.failures=[];self.world_cache={};self.eval_cache={};self.trace_rows=0
    def check(self,value,name,context=None):
        self.counts[name]+=1
        if not bool(value):self.failures.append(dict(check=name,context=context))


def get_pools(audit):
    entries=read(ROOT.parent/'redesign_v0.4/data/manifest.json')['images']
    pools={(s,k):np.asarray([i for i,e in enumerate(entries) if e['split']==s and e['category']==name],np.int64)
        for s in ('train','test') for k,name in enumerate(('food','water'))}
    audit.check([len(pools[s,k]) for s in ('train','test') for k in (0,1)]==[22,22,8,8],'photo_pool_sizes')
    audit.check(not({e['sha256'] for e in entries if e['split']=='train'}&{e['sha256'] for e in entries if e['split']=='test'}),
                'photos_disjoint_by_sha')
    return pools


def world(seed,n,pools,training):
    rng=np.random.default_rng(seed);rows=[];h=hashlib.sha256()
    for who in (0,1):
        size=n//2
        if training:mapid=rng.choice(np.arange(30),size);first=rng.integers(2,size=size)
        else:
            cells=np.tile(np.asarray(list(product(range(30),range(2)))),(size//60,1))[rng.permutation(size)]
            mapid,first=cells.T
        row=dict(positions=MAPS[mapid].copy(),photo_ids=np.column_stack([rng.choice(pools[k],size) for k in (0,1)]),
                 goals=np.column_stack((first,1-first)),menu=np.argsort(rng.random((size,2,6)),axis=2),
                 scout=np.full(size,who),episode=np.arange(size))
        for key in ('positions','photo_ids','goals','menu'):h.update(np.ascontiguousarray(row[key]).tobytes())
        rows.append(row)
    return h.hexdigest(),{k:np.concatenate([r[k] for r in rows]) for k in rows[0]}


def trace_audit(audit,path,stats,seed,split,mode,n,pools):
    label=[path.parent.name,mode];key=(seed,n)
    if key not in audit.eval_cache:audit.eval_cache[key]=world(seed,n,{k:pools['test',k] for k in (0,1)},False)
    digest,external=audit.eval_cache[key]
    audit.check(stats['world_sha256']==digest and stats['episodes']==n and stats['horizon']==1,
                'evaluation_world_hash_and_budget',label)
    with np.load(path) as z:
        audit.trace_rows+=len(z['reward'])
        audit.check(all(np.array_equal(z[k],v) for k,v in external.items()),'independent_evaluation_world_arrays',label)
        audit.check(z['inventory'].shape==(n,2) and (z['inventory']==0).all() and
                    z['history'].shape==(n,18) and (z['history']==0).all(),'zero_private_history_and_inventory',label)
        audit.check(z['sent'].shape==(n,2) and z['sent'].dtype==np.int64 and ((z['sent']>=0)&(z['sent']<7)).all(),
                    'two_integer_symbols',label)
        delivered=z['sent'].copy()
        if mode=='blank':delivered[:]=0
        elif mode=='shuffle':
            for scout in (0,1):
                rng=np.random.default_rng(seed+10001+scout)
                for first in (0,1):
                    ix=np.flatnonzero((z['scout']==scout)&(z['goals'][:,0]==first))
                    delivered[ix]=z['sent'][rng.permutation(ix)]
        audit.check(np.array_equal(z['delivered'],delivered),'correct_message_intervention',label)
        audit.check(z['action'].shape==(n,2) and z['action'].dtype==np.int64 and
                    ((z['action']>=0)&(z['action']<6)).all(),'two_legal_actions',label)
        place=np.take_along_axis(z['menu'],z['action'][:,:,None],2)[:,:,0]
        success=(place==np.take_along_axis(z['positions'],z['goals'],1)).astype(np.int64)
        reward=(success[:,0]+success[:,1])/4+(success[:,0]*success[:,1])/2
        audit.check(np.array_equal(z['place'],place) and np.array_equal(z['successes'],success) and
                    np.array_equal(z['reward'],reward),'independent_actions_and_mixed_reward',label)
        by_resource=np.zeros_like(success)
        for query in (0,1):by_resource[np.arange(n),z['goals'][:,query]]=success[:,query]
        code=2*by_resource[:,0]+by_resource[:,1]
        mapid=z['positions'][:,0]*5+z['positions'][:,1]-(z['positions'][:,1]>z['positions'][:,0])
        old,new=old_new(split)
        def count(mask):return {k:int((code[mask]==i).sum()) for i,k in enumerate(('00','01','10','11'))}
        audit.check(metric_match(stats,metrics(count(np.ones(n,bool)))),'complete_outcome_metrics_recount',label)
        audit.check(stats['new_maps_in_adaptation_training'] is True,'new_maps_labelled_as_training',label)
        for group,pool in (('old',old),('new',new)):
            ix=np.isin(mapid,pool)
            audit.check(metric_match(stats['map_groups'][group],metrics(count(ix))),'old_new_metrics_recount',[*label,group])
            for scout in (0,1):
                audit.check(metric_match(stats['direction_groups'][scout][group],metrics(count(ix&(z['scout']==scout)))),
                            'direction_old_new_metrics_recount',[*label,scout,group])
        for scout in (0,1):
            ix=z['scout']==scout;expected=metrics(count(ix))
            audit.check(metric_match(stats['direction_metrics'][scout],expected) and
                        stats['direction_means'][scout]==expected['mean_reward'],'direction_metrics_recount',[*label,scout])
            audit.check(Counter(zip(mapid[ix].tolist(),z['goals'][ix,0].tolist()))==
                        Counter({(m,g):n//120 for m in range(30) for g in (0,1)}),'balanced_map_query_direction_cells',[*label,scout])


def audit_one(audit,folder,seed,split,arm,updates,batch,eval_n,hashes,pools):
    label=folder.name;cfg=read(folder/'config.json');result=read(folder/'result.json')
    old,new=old_new(split);boundary=updates-min(100,max(1,updates//6))
    checkpoints=sorted(set([0,updates]+[x for x in (25,50,100,200,400,500) if x<updates]))
    plan=dict(split=split,representation='identity',schedule='direct',vocab=7,length=2,known=False,
              blocked=False,reward_kind='mixed',complementarity=.5)
    for key,value in dict(seed=seed,split=split,arm=arm,plan=plan,training_plan=dict(plan,map_pool=list(range(30)),allowed_sites=list(range(6))),
        updates=updates,batch=batch,eval_n=eval_n,checkpoints=checkpoints,learning_rate=.0007,entropy_coefficient=.02,
        entropy_off_after=boundary,gradient_clip='norm 2 separately within sender and receiver modules in every arm',
        optimizer='fresh Adam in all arms',role_loss_reduction='mean of both roles including frozen constant',
        evaluation_seed=seed_for(seed,split,2),seed_formula='SeedSequence([9009,seed,split,purpose,step]) uint64 >> 1',
        training_purpose=1,evaluation_purpose=2,source_hashes=hashes,old_map_ids=list(old),new_map_ids=list(new),
        inherited_seeds=True,new_maps_are_adaptation_training=True).items():audit.check(cfg.get(key)==value,'fixed_run_configuration',[label,key])
    source=SOURCE/f's{seed}_split{split}_mixed/final.pt';source_states=load(source)
    audit.check(cfg['source_checkpoint']==dict(path=str(source),sha256=sha(source)),'source_checkpoint_path_and_digest',label)
    initial=load(folder/'initial.pt');final=load(folder/'final.pt')
    audit.check(same(initial,source_states),'initial_is_exact_v8_checkpoint',label)
    audit.check(state_hash(initial)==cfg['initial_sha256']==result['initial_sha256'] and
                state_hash(initial)==read(source.parent/'result.json')['final_sha256'],'initial_hash_and_original_result',label)
    audit.check(state_hash(final)==result['final_sha256'],'final_hash',label)
    for key,value in dict(seed=seed,split=split,arm=arm,updates=updates,batch=batch,source_checkpoint=cfg['source_checkpoint'],
        frozen_modules_verified=True).items():audit.check(result.get(key)==value,'result_configuration',[label,key])
    for who,state in enumerate(initial):
        active=[k for k in state if active_key(k,arm)]
        frozen=[k for k in state if k!='input_transform' and not active_key(k,arm)]
        expected=dict(active=active,frozen=frozen,trainable_count=sum(state[k].numel() for k in active))
        audit.check(cfg['partition'][who]==expected,'exact_active_frozen_partition',[label,who])
        audit.check(expected['trainable_count']=={'sender_only':198234,'receiver_only':11741,'both':209975}[arm],
                    'expected_module_parameter_budget',[label,who])
        audit.check(all(torch.equal(state[k],final[who][k]) for k in (*frozen,'input_transform')),
                    'all_inactive_final_weights_unchanged',[label,who])
        startopt=load(folder/'initial_optimizer.pt')[who];endopt=load(folder/'final_optimizer.pt')[who]
        audit.check(not startopt['state'],'adam_initial_state_empty',[label,who])
        audit.check(len(endopt['state'])==len(active) and all(float(x['step'])==updates for x in endopt['state'].values()),
                    'adam_active_parameters_and_steps',[label,who])
        for opt in (startopt,endopt):
            audit.check(sum(len(g['params']) for g in opt['param_groups'])==len(active) and
                all(g['lr']==.0007 and g['betas']==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0
                    for g in opt['param_groups']),'adam_fixed_hyperparameters',[label,who])
    for point in checkpoints:
        state=load(folder/f'checkpoint_{point:04d}.pt')
        audit.check(all(all(torch.equal(v,state[who][k]) for k,v in before.items() if not active_key(k,arm))
                    for who,before in enumerate(initial)), 'all_checkpoints_frozen_partition',[label,point])
        if point in (0,updates):audit.check(same(state,initial if point==0 else final),'checkpoint_endpoint_match',[label,point])
    rows=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
    audit.check(len(rows)==updates,'exact_training_budget',label)
    old_count=new_count=0
    for step,row in enumerate(rows):
        derived=seed_for(seed,split,1,step);key=(seed,split,step,batch)
        if key not in audit.world_cache:
            digest,external=world(derived,batch,{k:pools['train',k] for k in (0,1)},True)
            mids=external['positions'][:,0]*5+external['positions'][:,1]-(external['positions'][:,1]>external['positions'][:,0])
            audit.world_cache[key]=(digest,int(np.isin(mids,old).sum()),int(np.isin(mids,new).sum()))
        digest,oldn,newn=audit.world_cache[key];old_count+=oldn;new_count+=newn
        audit.check(row['update']==step+1 and row['rng_seed']==derived and row['world_sha256']==digest,
                    'independent_training_world_hash',[label,step+1])
        weight=.02 if step<boundary else 0
        audit.check(row['entropy_weight']==weight,'fixed_entropy_schedule',[label,step+1])
        audit.check(0<=row['both_accuracy']<=row['single_accuracy']<=1 and
                    np.isclose(row['reward'],.5*(row['single_accuracy']+row['both_accuracy']),rtol=0,atol=1e-12),
                    'training_mixed_return_consistency',[label,step+1])
        for who,info in enumerate(row['agents']):
            roles=('sender','receiver') if who==0 else ('receiver','sender')
            expected_loss=0
            for role,c in zip(roles,info['components']):
                expected_loss+=(c['policy_loss']+c['value_loss']-weight*c['entropy'])/2
                audit.check(c['trainable']==(arm=='both' or arm==role+'_only'),'role_gradient_selection',[label,step+1,who,role])
            audit.check(np.isfinite(info['loss']) and np.isclose(info['loss'],expected_loss,atol=2e-6,rtol=1e-6),
                        'mean_of_two_role_losses',[label,step+1,who])
            expected_roles={'sender','receiver'} if arm=='both' else {arm.removesuffix('_only')}
            audit.check(set(info['gradient_norm_by_role'])==expected_roles and
                        all(np.isfinite(v) and v>=0 for v in info['gradient_norm_by_role'].values()),
                        'separate_active_module_clipping',[label,step+1,who])
    curve=read(folder/'curve.json')
    audit.check([x['update'] for x in curve]==checkpoints,'fixed_curve_checkpoints',label)
    for point in curve:
        audit.check(point['frozen_modules_verified'] and set(point['scores'])=={'normal'},'checkpoint_modes_and_frozen_flag',[label,point['update']])
        stats=point['scores']['normal']
        audit.check(stats['n']==eval_n and stats['decisions']==2*eval_n and stats['map_groups']['old']['n']==4*eval_n//5 and
                    stats['map_groups']['new']['n']==eval_n//5,'checkpoint_old_new_world_budget',[label,point['update']])
        for item in [stats,*stats['map_groups'].values(),*stats['direction_metrics'],
                     *[x for d in stats['direction_groups'] for x in d.values()]]:
            audit.check(metric_match(item,metrics(item['outcome_counts'])),'checkpoint_metrics_from_counts',[label,point['update']])
    audit.check(set(result['scores'])==set(MODES) and result['scores']['normal']==curve[-1]['scores']['normal'],
                'final_modes_and_last_curve_match',label)
    for mode in MODES:trace_audit(audit,folder/f'final_{mode}.npz',result['scores'][mode],seed_for(seed,split,2),split,mode,eval_n,pools)
    return dict(seed=seed,split=split,arm=arm,rows=rows,initial=initial,final=final,curve=curve,
                old_training_worlds=old_count,new_training_worlds=new_count,source_checkpoint=str(source))


def audit_sources(audit,root,formal,updates,batch,eval_n):
    paths=sorted(root.glob('invocation_*.json'));audit.check(bool(paths),'invocations_present')
    if not paths:raise RuntimeError('No invocation')
    hashes=read(paths[0])['source_hashes']
    for path in paths:
        record=read(path)
        audit.check(record['source_hashes']==hashes and record['formal']==formal,'invocation_source_and_mode',path.name)
        for key,value in dict(updates=updates,batch=batch,eval_n=eval_n).items():audit.check(record[key]==value,'invocation_budget',[path.name,key])
        expected=[seed_for(s,k,1,u) for s in record['seeds'] for k in record['splits'] for u in range(updates)]
        expected += [seed_for(s,k,2) for s in record['seeds'] for k in record['splits']]
        audit.check(len(expected)==len(set(expected))==record['derived_rng_count']==record['unique_derived_rng_count'],
                    'derived_rng_uniqueness',path.name)
        if formal:audit.check(record['seeds']==list(SEEDS) and record['splits']==[1,2,3] and record['arms']==list(ARMS),
                              'all_fixed_sources_and_arms',path.name)
    snapshot=root/f"sources_{hashes[str(ROOT/'run_adaptation.py')][:12]}"
    for name,digest in hashes.items():
        audit.check(Path(name).exists() and sha(name)==digest,'current_source_and_asset_hash',name)
        if Path(name).suffix in ('.py','.md'):
            audit.check(sha(snapshot/Path(name).name)==digest,'executed_snapshot_hash',name)
    return hashes


def paired_audit(audit,groups):
    for key,runs in groups.items():
        if len(runs)!=3:continue
        base=runs['both']
        for arm in ('sender_only','receiver_only'):
            other=runs[arm]
            audit.check(same(base['initial'],other['initial']) and base['curve'][0]==other['curve'][0],
                        'shared_source_and_zero_step_evaluation',[*key,arm])
            audit.check(all(all(a[k]==b[k] for k in ('update','rng_seed','world_sha256','entropy_weight'))
                            for a,b in zip(base['rows'],other['rows'])),'paired_world_and_exploration_stream',[*key,arm])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'results/adaptation_001')
    p.add_argument('--out',type=Path);args=p.parse_args();root=args.root.resolve();out=args.out or root/'audit_adaptation.json'
    if out.exists():raise FileExistsError(out)
    audit=Audit();pools=get_pools(audit);hashes=audit_sources(audit,root,True,600,512,9600)
    qa=read(ROOT/'preflight_qa.json')
    audit.check(qa['passed'] and qa['source_sha256']==hashes[str(ROOT/'run_adaptation.py')],'preflight_gate_valid')
    groups={};results=[];pending=[]
    expected={f's{s}_split{k}_{a}' for s in SEEDS for k in (1,2,3) for a in ARMS}
    actual={p.name for p in root.glob('s*') if p.is_dir() and p.name[1:2].isdigit()}
    audit.check(actual==expected,'exact_thirty_six_run_directories')
    for seed in SEEDS:
      for split in (1,2,3):
        groups[seed,split]={}
        for arm in ARMS:
            folder=root/f's{seed}_split{split}_{arm}'
            if not(folder/'result.json').exists():pending.append(folder.name);continue
            result=audit_one(audit,folder,seed,split,arm,600,512,9600,hashes,pools)
            groups[seed,split][arm]=result
            results.append({k:v for k,v in result.items() if k not in ('rows','initial','final','curve')})
    paired_audit(audit,groups)
    audit.check(len(results)==36 and not pending,'all_thirty_six_complete',pending)
    report=dict(passed=not audit.failures,status='failed' if audit.failures else 'passed',completed=len(results),pending=pending,
        checks=dict(audit.counts),total_checks=sum(audit.counts.values()),failures=audit.failures,
        training_world_hashes_verified=audit.counts['independent_training_world_hash'],unique_training_batches=len(audit.world_cache),
        evaluation_world_rows=audit.trace_rows,evaluation_primitive_decisions=2*audit.trace_rows,
        sources_and_exposure=results,source_hashes=hashes,preflight_qa_sha256=sha(ROOT/'preflight_qa.json'),
        audit_script_sha256=sha(__file__),completed_utc=datetime.now(timezone.utc).isoformat(),
        boundaries=['Four inherited independent subject pairs, with three repeated split contexts each; no new independent preparations.',
                    'New maps enter adaptation training from step one. This audits adaptation, not zero-shot generalization.',
                    'Saved weights, RNG worlds, recorded role losses and outcomes are audited; no full optimizer replay or new model inference.'])
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'],completed=len(results),failures=len(audit.failures),checks=report['total_checks'],
        world_hashes=report['training_world_hashes_verified'],trace_rows=audit.trace_rows,out=str(out)),ensure_ascii=False))
    if audit.failures:sys.exit(1)


if __name__=='__main__':main()
